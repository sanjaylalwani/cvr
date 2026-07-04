#!/usr/bin/env python3
"""
QLoRA fine-tune of Gemma 2 9B IT on Marathi instruction data.

Run this on the rented GPU (A100 80GB recommended; 4090/24GB works with
per_device_train_batch_size=1 and gradient_accumulation_steps=16).

Setup on the GPU box:
    pip install "torch>=2.3" transformers trl peft bitsandbytes datasets accelerate
    huggingface-cli login        # token with access to google/gemma-2-9b-it
    python train_qlora.py

Expects train.jsonl / eval.jsonl (from prepare_data.py) in the same directory.
"""
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "google/gemma-2-9b-it"
OUTPUT_DIR = "gemma2-9b-marathi-qlora"
MAX_SEQ_LEN = 2048

# ---------------------------------------------------------------- data
data = load_dataset(
    "json",
    data_files={"train": "train.jsonl", "eval": "eval.jsonl"},
)
# Keep only the messages column; TRL applies the model's chat template to it.
data = data.remove_columns(
    [c for c in data["train"].column_names if c != "messages"]
)
print(data)

# ---------------------------------------------------------------- model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # Gemma 2 uses logit soft-capping; eager attention is the safe choice.
    attn_implementation="eager",
)
model.config.use_cache = False  # incompatible with gradient checkpointing

# ---------------------------------------------------------------- LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# ---------------------------------------------------------------- train
train_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                  # 1 epoch is usually right for SFT
    per_device_train_batch_size=4,       # A100 80GB; drop to 1-2 on 24GB
    gradient_accumulation_steps=4,       # effective batch = 16 conversations
    gradient_checkpointing=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_length=MAX_SEQ_LEN,
    packing=False,
    bf16=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    report_to="none",                    # or "wandb" if you use it
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=train_config,
    train_dataset=data["train"],
    eval_dataset=data["eval"],
    processing_class=tokenizer,
    peft_config=peft_config,
)

# Sanity check: print one fully-templated training example before burning money
sample = trainer.train_dataset[0]
print("=" * 60)
print(tokenizer.decode(sample["input_ids"][:512]))
print("=" * 60)

trainer.train()

trainer.save_model(OUTPUT_DIR)  # saves LoRA adapters only (~100-200 MB)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Adapters in ./{OUTPUT_DIR} — download these before killing the pod!")
