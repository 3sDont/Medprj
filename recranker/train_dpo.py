# -*- coding: utf-8 -*-
"""
recranker/train_dpo.py
DPO fine-tuning on top of the SFT model for journal reranking.

Requires:   train_sft.py already run → adapter at OUTPUT_DIR or merged on HF Hub.

Run on Google Colab (A100/T4 GPU).
"""

# !pip install -q unsloth trl transformers datasets peft accelerate bitsandbytes

import json
import os
import re

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer

PatchDPOTrainer()

# ─────────────────────────────────────────────────────────────────────────────
# Config — load the SFT checkpoint, not the base model
# ─────────────────────────────────────────────────────────────────────────────
SFT_MODEL       = "recranker-sft-adapter"   # local path OR HF repo after SFT
MAX_SEQ_LENGTH  = 4096
LOAD_IN_4BIT    = True
LORA_R          = 64
LORA_ALPHA      = 64
OUTPUT_DIR      = "recranker-dpo-adapter"
HF_REPO         = None
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
DPO_JSONL       = "/content/dpo_train.jsonl"
MAX_STEPS       = 1000
BATCH_SIZE      = 2
GRAD_ACCUM      = 4
LR              = 5e-6
BETA            = 0.1       # DPO temperature


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = SFT_MODEL,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,
    load_in_4bit   = LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_R,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = 0,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an academic journal recommendation system for biomedical research. "
    "Based on a paper's content, rank the candidate journals from most to least "
    "suitable for publication."
)


def load_dpo_jsonl(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Format prompt as chat string
            system = next(
                (m["content"] for m in obj["prompt"] if m["role"] == "system"),
                SYSTEM_PROMPT,
            )
            user = next(
                (m["content"] for m in obj["prompt"] if m["role"] == "user"),
                "",
            )
            prompt_str = (
                f"<|system|>\n{system}</s>\n"
                f"<|user|>\n{user}</s>\n"
                f"<|assistant|>\n"
            )
            chosen_str   = obj["chosen"][0]["content"]   + "</s>\n"
            rejected_str = obj["rejected"][0]["content"] + "</s>\n"
            records.append({
                "prompt"  : prompt_str,
                "chosen"  : chosen_str,
                "rejected": rejected_str,
            })
    return Dataset.from_list(records)


dpo_ds = load_dpo_jsonl(DPO_JSONL)
print(f"DPO examples: {len(dpo_ds)}")
print(f"\nSample prompt (first 300 chars):\n{dpo_ds[0]['prompt'][:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# DPO training
# ─────────────────────────────────────────────────────────────────────────────

dpo_trainer = DPOTrainer(
    model     = model,
    ref_model = None,         # unsloth handles the reference internally
    args = DPOConfig(
        per_device_train_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUM,
        warmup_ratio                 = 0.1,
        max_steps                    = MAX_STEPS,
        learning_rate                = LR,
        fp16                         = not torch.cuda.is_bf16_supported(),
        bf16                         = torch.cuda.is_bf16_supported(),
        logging_steps                = 20,
        save_steps                   = 200,
        output_dir                   = OUTPUT_DIR,
        optim                        = "adamw_8bit",
        weight_decay                 = 0.05,
        lr_scheduler_type            = "cosine",
        seed                         = 42,
        report_to                    = "none",
        push_to_hub                  = bool(HF_REPO),
        hub_model_id                 = HF_REPO,
        hub_token                    = HF_TOKEN,
    ),
    beta             = BETA,
    train_dataset    = dpo_ds,
    tokenizer        = tokenizer,
    max_length       = 1024,
    max_prompt_length= 800,
)

print("Starting DPO training ...")
dpo_trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"DPO adapter saved -> {OUTPUT_DIR}/")

if HF_REPO and HF_TOKEN:
    model.push_to_hub_merged(
        HF_REPO, tokenizer,
        save_method = "merged_16bit",
        token       = HF_TOKEN,
    )
    print(f"Merged DPO model pushed -> https://huggingface.co/{HF_REPO}")
