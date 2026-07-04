#!/usr/bin/env python3
"""
Merge the trained LoRA adapters into the Gemma 2 9B base model and smoke-test it.

Run on the GPU box after training (needs ~40GB RAM/VRAM headroom to merge in bf16):
    python merge_and_test.py

Produces ./gemma2-9b-marathi-merged — a standalone model you can:
  - load with transformers / vLLM
  - push to the HF Hub
  - convert to GGUF for Ollama on your Mac (see README)
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "google/gemma-2-9b-it"
ADAPTERS = "gemma2-9b-marathi-qlora"
MERGED = "gemma2-9b-marathi-merged"

PUSH_TO_HUB = False          # set True + repo name to publish
HUB_REPO = "your-username/gemma2-9b-marathi"

print("Loading base model in bf16 (not 4-bit — merging needs full weights)...")
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(ADAPTERS)

print("Attaching and merging adapters...")
model = PeftModel.from_pretrained(base, ADAPTERS)
model = model.merge_and_unload()

model.save_pretrained(MERGED, safe_serialization=True)
tokenizer.save_pretrained(MERGED)
print(f"Merged model saved to ./{MERGED}")

if PUSH_TO_HUB:
    model.push_to_hub(HUB_REPO, private=True)
    tokenizer.push_to_hub(HUB_REPO, private=True)
    print(f"Pushed to https://huggingface.co/{HUB_REPO}")

# ----------------------------------------------------------------- smoke test
prompts = [
    "महाराष्ट्रातील पाच प्रसिद्ध किल्ल्यांची माहिती द्या.",
    "मला चहा बनवण्याची कृती सांगा.",
    "'शिक्षणाचे महत्त्व' या विषयावर एक छोटा निबंध लिहा.",
]

model.eval()
for p in prompts:
    messages = [{"role": "user", "content": p}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    print("\n" + "=" * 70)
    print("PROMPT:", p)
    print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
