# -*- coding: utf-8 -*-
"""
recranker/train_sft.py
SFT fine-tuning of Qwen2.5-4B-Instruct for journal reranking.

Uses Qwen/Qwen3.5-4B (4B params, ~8 GB VRAM — fits comfortably on Kaggle P100 16GB).
Thinking mode is disabled (enable_thinking=False) so the model outputs rankings
directly without <think> traces, matching the SFT training data format.

NOTE: "Qwen3.5:4b" in Ollama = "Qwen/Qwen3.5-4B" on HuggingFace.
Ollama is for local inference only; Kaggle training pulls from HuggingFace.

GPU requirement: T4 (sm_75) or better — P100 (sm_60) is NOT supported by
current PyTorch builds and will crash immediately.
On Kaggle: Settings → Accelerator → GPU T4 x1

Install cell (run FIRST, then restart kernel):
  !pip uninstall -y unsloth
  !pip install -q trl transformers datasets peft accelerate bitsandbytes
  !pip install -q causal-conv1d mamba-ssm   # Qwen3.5 hybrid-Mamba deps
"""

import sys

if "unsloth" in sys.modules:
    raise RuntimeError(
        "unsloth is active in this kernel — it patches transformers and breaks P100.\n"
        "Fix: run `!pip uninstall -y unsloth` then RESTART the kernel."
    )

import gc
import json
import os
import re

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen3.5-4B"
MAX_SEQ_LENGTH  = 3072   # safe on P100 with 4B model (was 4096 → OOM with 7B)
LORA_R          = 16
LORA_ALPHA      = 32
OUTPUT_DIR      = "/kaggle/working/recranker-sft-adapter"
HF_REPO         = None   # e.g. "your-hf-username/recranker-qwen25-4b-sft"
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
TRAIN_JSONL     = "/kaggle/input/recranker-data/sft_train.jsonl"
VAL_JSONL       = "/kaggle/input/recranker-data/sft_val.jsonl"
MAX_STEPS       = 2000
BATCH_SIZE      = 2      # safe with 4B model (was 1 with 7B)
GRAD_ACCUM      = 4      # effective batch = 2×4 = 8
LR              = 2e-4

# P100 does NOT support bfloat16
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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_config,
    device_map          = "auto",
    trust_remote_code   = True,
    attn_implementation = "eager",  # flash-attn requires sm_75+; P100 is sm_60
)
model = prepare_model_for_kbit_training(model)

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
# Dataset  — Qwen3.5 ChatML via apply_chat_template
# ─────────────────────────────────────────────────────────────────────────────
# Qwen3.5 thinks by default (<think>…</think> before every response).
# enable_thinking=False disables it so training targets only the ranking output.

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_examples(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize            = False,
            add_generation_prompt = False,
            enable_thinking     = False,   # suppress <think> in training data
        )
        texts.append(text)
    return {"text": texts}


raw_train = load_jsonl(TRAIN_JSONL)
raw_val   = load_jsonl(VAL_JSONL)

train_ds = Dataset.from_list(raw_train).map(format_examples, batched=True)
val_ds   = Dataset.from_list(raw_val).map(format_examples, batched=True)

print(f"Train examples : {len(train_ds)}")
print(f"Val   examples : {len(val_ds)}")
print(f"\nSample (first 400 chars):\n{train_ds[0]['text'][:400]}")

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model              = model,
    processing_class   = tokenizer,
    train_dataset      = train_ds,
    eval_dataset       = val_ds,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    dataset_num_proc   = 1,
    packing            = False,
    args = TrainingArguments(
        per_device_train_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUM,
        warmup_steps                 = 50,
        max_steps                    = MAX_STEPS,
        learning_rate                = LR,
        fp16                         = USE_FP16,
        bf16                         = USE_BF16,
        logging_steps                = 20,
        eval_strategy                = "steps",
        eval_steps                   = 200,
        save_strategy                = "steps",
        save_steps                   = 400,
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

model.eval()

EVAL_SYSTEM = (
    "You are an academic journal recommendation system for biomedical research. "
    "Based on a paper's content, rank the candidate journals from most to least "
    "suitable for publication."
)


def generate_ranking(prompt_text: str, max_new_tokens: int = 600) -> str:
    messages = [
        {"role": "system", "content": EVAL_SYSTEM},
        {"role": "user",   "content": prompt_text},
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = True,
        enable_thinking       = False,   # no <think> during eval either
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
