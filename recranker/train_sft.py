# -*- coding: utf-8 -*-
"""
recranker/train_sft.py
SFT fine-tuning of Qwen3.5-4B for journal reranking.

GPU requirement: T4 (sm_75) or better.
P100 (sm_60) is NOT supported by current PyTorch — will crash on model load.
On Kaggle: Settings → Accelerator → GPU T4 x1

Install cell (run FIRST, then RESTART kernel):
  !pip uninstall -y unsloth
  !pip install -q "trl>=1.6.0" transformers datasets peft accelerate bitsandbytes
  !pip install -q causal-conv1d mamba-ssm
"""

import sys

if "unsloth" in sys.modules:
    raise RuntimeError(
        "unsloth is active — it patches transformers and causes CUDA errors.\n"
        "Fix: run `!pip uninstall -y unsloth` then RESTART the kernel."
    )

import gc
import json
import os
import re

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Chunked cross-entropy patch
# Standard F.cross_entropy materializes a workspace equal to the full logit tensor
# (seq_len × vocab_size) inside the CUDA kernel — on a 262K-vocab model this
# allocates 7+ GB and OOMs a T4.  Replacing fixed_cross_entropy with a version
# that processes 256 tokens at a time keeps peak workspace at ~300 MB.
# ─────────────────────────────────────────────────────────────────────────────
import transformers.loss.loss_utils as _loss_utils

def _chunked_fixed_cross_entropy(
    source, target, num_items_in_batch=None, ignore_index=-100, chunk_size=256, **kwargs
):
    total = source.new_zeros(())
    n = 0
    for i in range(0, source.shape[0], chunk_size):
        s = source[i : i + chunk_size]
        t = target[i : i + chunk_size]
        valid = t != ignore_index
        if valid.any():
            total = total + F.cross_entropy(s[valid], t[valid], reduction="sum")
            n += int(valid.sum())
    loss = total / max(n, 1)
    if num_items_in_batch is not None:
        loss = loss * max(n, 1) / num_items_in_batch
    return loss

_loss_utils.fixed_cross_entropy = _chunked_fixed_cross_entropy
print("Patched fixed_cross_entropy → chunked (256-token chunks per F.cross_entropy call)")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen3.5-4B"
MAX_SEQ_LENGTH = 1536          # 5 candidates × ~195 tok + paper + framing ≈ 1,425 tok → fits with headroom
LORA_R         = 8            # 16 → 8: halves LoRA param count, saves ~200 MB peak VRAM
LORA_ALPHA     = 16           # keep ratio 2×R
OUTPUT_DIR     = "/kaggle/working/recranker-sft-adapter"
HF_REPO        = None          # e.g. "your-hf-username/recranker-qwen35-4b-sft"
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
TRAIN_JSONL    = "/kaggle/input/datasets/khathih/recranker-data/sft_train.jsonl"
VAL_JSONL      = "/kaggle/input/datasets/khathih/recranker-data/sft_val.jsonl"
MAX_STEPS      = 800           # 4759 examples / eff_batch(8) ≈ 595 steps/epoch → ~1.3 epochs
BATCH_SIZE     = 1
GRAD_ACCUM     = 8             # effective batch = 1×8 = 8
LR             = 2e-4

# T4 does NOT support bfloat16 natively — keep fp16
USE_FP16 = True
USE_BF16 = False

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,
)

n_gpus = torch.cuda.device_count()
# Pin all layers to GPU 0. PEFT LoRA gradients that cross device boundaries
# (device_map="balanced") create large PCIe-buffered copies that OOM individual GPUs.
# A 4B model in 4-bit (~2 GB) fits easily on a single 16 GB T4.
device_map = {"": "cuda:0"}   # BitsAndBytes requires string device name, not integer 0
print(f"GPUs detected: {n_gpus}  →  device_map={device_map!r} (single-GPU LoRA)")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_config,
    device_map          = device_map,
    trust_remote_code   = True,
    attn_implementation = "eager",  # flash-attn requires sm_80+
)
model = prepare_model_for_kbit_training(model)

_vram_gb = torch.cuda.memory_allocated() / 1e9
print(f"VRAM after model load: {_vram_gb:.2f} GB  (expect ≤ 3 GB for 4-bit; >6 GB = BNB not active)")
if _vram_gb > 6:
    raise RuntimeError(
        f"BitsAndBytes 4-bit did NOT activate — model loaded in fp16 ({_vram_gb:.1f} GB).\n"
        "Fix: run `!pip install -U bitsandbytes` then RESTART kernel and re-run."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# Qwen3.5 has thinking mode ON by default — disable so loss targets only
# the ranking output (training data has no <think> content).
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_item(item: dict) -> dict:
    text = tokenizer.apply_chat_template(
        item["messages"],
        tokenize              = False,
        add_generation_prompt = False,
        enable_thinking       = False,
    )
    return {"text": text}


raw_train = load_jsonl(TRAIN_JSONL)
raw_val   = load_jsonl(VAL_JSONL)

train_ds = Dataset.from_list([format_item(x) for x in raw_train])
val_ds   = Dataset.from_list([format_item(x) for x in raw_val])

print(f"Train examples : {len(train_ds)}")
print(f"Val   examples : {len(val_ds)}")
print(f"\nSample (first 400 chars):\n{train_ds[0]['text'][:400]}")

# ─────────────────────────────────────────────────────────────────────────────
# Training  — SFTConfig replaces TrainingArguments in TRL >= 0.12
# SFT-specific args (max_length, dataset_text_field, packing, dataset_num_proc)
# live in SFTConfig, NOT in SFTTrainer constructor.
# ─────────────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model            = model,
    processing_class = tokenizer,
    train_dataset    = train_ds,
    eval_dataset     = val_ds,
    args = SFTConfig(
        # ── SFT-specific ──────────────────────────────────────────────────────
        max_length             = MAX_SEQ_LENGTH,
        dataset_text_field     = "text",
        dataset_num_proc       = 1,
        packing                = False,   # packing always fills 1024 tokens → maximizes lm_head spike; disabled for OOM fix
        # ── Training ──────────────────────────────────────────────────────────
        per_device_train_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUM,
        warmup_steps                 = 50,
        max_steps                    = MAX_STEPS,
        learning_rate                = LR,
        fp16                         = USE_FP16,
        bf16                         = USE_BF16,
        logging_steps                = 20,
        eval_strategy                = "steps",
        eval_steps                   = 100,
        save_strategy                = "steps",
        save_steps                   = 100,    # frequent saves — Kaggle cuts session at 12h
        output_dir                   = OUTPUT_DIR,
        optim                        = "adamw_8bit",
        weight_decay                 = 0.01,
        lr_scheduler_type            = "cosine",
        seed                         = 42,
        report_to                    = "none",
        push_to_hub                  = bool(HF_REPO),
        hub_model_id                 = HF_REPO,
        hub_token                    = HF_TOKEN,
        gradient_checkpointing       = True,
        gradient_checkpointing_kwargs= {"use_reentrant": False},
    ),
)

gc.collect()
torch.cuda.empty_cache()
print(f"VRAM before training: {torch.cuda.memory_allocated()/1e9:.1f} GB / "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

print("\nStarting SFT training ...")
trainer_stats = trainer.train()
print(f"Training complete. Steps: {trainer_stats.global_step}")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved -> {OUTPUT_DIR}/")

if HF_REPO and HF_TOKEN:
    merged = model.merge_and_unload()
    merged.save_pretrained(OUTPUT_DIR + "-merged", safe_serialization=True)
    merged.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"Merged model pushed -> https://huggingface.co/{HF_REPO}")

# ─────────────────────────────────────────────────────────────────────────────
# Quick val evaluation
# ─────────────────────────────────────────────────────────────────────────────

EVAL_SYSTEM = (
    "You are an academic journal recommendation system for biomedical research. "
    "Based on a paper's content, rank the candidate journals from most to least "
    "suitable for publication."
)

model.eval()


def generate_ranking(prompt_text: str, max_new_tokens: int = 600) -> str:
    messages = [
        {"role": "system", "content": EVAL_SYSTEM},
        {"role": "user",   "content": prompt_text},
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = True,
        enable_thinking       = False,
    )
    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            use_cache      = True,
            do_sample      = False,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_top1_journal(response: str) -> int | None:
    m = re.search(r"Rank\s*1\s*:\s*Journal\s+(\d+)", response, re.IGNORECASE)
    return int(m.group(1)) if m else None


print("\nRunning quick val evaluation (first 50 examples) ...")
correct = 0
total   = 0
for ex in raw_val[:50]:
    user_prompt = ex["messages"][1]["content"]
    m = re.search(r"Rank\s*1\s*:\s*Journal\s+(\d+)", ex["messages"][2]["content"])
    true_id = int(m.group(1)) if m else None

    response = generate_ranking(user_prompt)
    pred_id  = parse_top1_journal(response)
    if true_id and pred_id == true_id:
        correct += 1
    total += 1

print(f"Val Acc@1 (first 50): {correct}/{total} = {correct/total:.1%}")
