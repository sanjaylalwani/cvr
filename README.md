# Gemma 2 9B → Marathi Instruction Model (QLoRA)

End-to-end runbook. Total cost target: **under $25**.

## Step 0 — One-time accounts (Mac, 10 min)

1. HuggingFace account → open https://huggingface.co/google/gemma-2-9b-it → accept the Gemma license (required, gated model).
2. Create an HF access token (read scope): https://huggingface.co/settings/tokens
3. RunPod account (runpod.io) with $25 credit. Lambda or Vast.ai work too.

## Step 1 — Prepare data (Mac, ~30 min)

```bash
pip3 install datasets
python3 prepare_data.py --list            # see available configs first
python3 prepare_data.py --max-examples 60000
```

If a subset fails to load, the script tells you and moves on — adjust
`--subsets` / split names based on what `--list` shows.

**Do not skip:** manually read ~20 samples for translation quality:

```bash
shuf -n 20 train.jsonl > sample.jsonl
```

If a subset looks garbled, drop it and re-run. Data quality > data quantity.

## Step 2 — Rent the GPU (5 min)

- RunPod → Deploy → **A100 80GB SXM** (~$1.6–1.9/hr) — comfortable
- Budget option: **RTX 4090 24GB** (~$0.35–0.70/hr) — set
  `per_device_train_batch_size=1`, `gradient_accumulation_steps=16` in train_qlora.py
- Template: official PyTorch 2.x + CUDA 12 image
- Disk: 100 GB volume

Upload `train.jsonl`, `eval.jsonl`, `train_qlora.py`, `merge_and_test.py`
(via runpodctl, scp, or a private HF dataset repo).

## Step 3 — Train (GPU box, ~2–5 hrs)

```bash
pip install "torch>=2.3" transformers trl peft bitsandbytes datasets accelerate
huggingface-cli login          # paste your HF token
nohup python train_qlora.py > train.log 2>&1 &
tail -f train.log
```

What to watch:
- The script prints one fully-templated example before training — verify the
  Gemma chat format (`<start_of_turn>user` ... `<start_of_turn>model`) wraps
  your Marathi text correctly.
- Train loss should drop steadily (typically ~2.x → ~1.x) in the first 200 steps.
- Eval loss every 200 steps: if it starts rising while train loss falls, stop —
  you have your model (checkpoints are saved every 200 steps).

Rough ETA on 60k conversations, 1 epoch: **~2–4 h on A100, ~6–10 h on 4090.**

## Step 4 — Merge + smoke test (GPU box, ~15 min)

```bash
python merge_and_test.py
```

Read the three Marathi generations. You're checking: fluent Devanagari, follows
the instruction, no random English/code-switching, no repetition loops.

## Step 5 — Get the model off the pod (before stopping it!)

Cheapest: push to a **private HF repo** (set `PUSH_TO_HUB = True` in
merge_and_test.py). The merged model is ~18 GB; adapters alone are ~200 MB.
At minimum, always download the adapter directory — with adapters + base model
name you can reconstruct everything later for free.

Then **terminate the pod**. Billing stops only when the pod is gone.

## Step 6 — Use it

### A. Python / transformers (any GPU box, or CPU slowly)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

m = AutoModelForCausalLM.from_pretrained(
    "your-username/gemma2-9b-marathi", torch_dtype=torch.bfloat16, device_map="auto")
t = AutoTokenizer.from_pretrained("your-username/gemma2-9b-marathi")

msgs = [{"role": "user", "content": "पुण्याबद्दल थोडक्यात माहिती द्या."}]
ids = t.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
print(t.decode(m.generate(ids, max_new_tokens=300)[0][ids.shape[-1]:], skip_special_tokens=True))
```

### B. Locally on your MacBook Pro via Ollama (recommended daily driver)

Convert the merged model to GGUF once (on the GPU box or any machine with
~40 GB disk):

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py ../gemma2-9b-marathi-merged --outfile marathi-gemma.gguf
# quantize to 4-bit for the Mac (~5.5 GB, runs great on Apple Silicon)
cmake -B build && cmake --build build -j
./build/bin/llama-quantize marathi-gemma.gguf marathi-gemma-q4_k_m.gguf Q4_K_M
```

On the Mac, create a `Modelfile`:

```
FROM ./marathi-gemma-q4_k_m.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

```bash
ollama create marathi-gemma -f Modelfile
ollama run marathi-gemma "मला संत तुकारामांबद्दल सांगा."
```

Now the model your Mac couldn't train, it can happily *serve* — Apple Silicon
is genuinely good at 4-bit inference.

### C. Serve for CompareLM

Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1` —
you can register `marathi-gemma` as a model row in CompareLM's models table
and compare it side-by-side against GPT/Claude on Marathi prompts. That's a
great built-in eval harness you already own.

## Step 7 — Evaluate & iterate

- Build a fixed set of ~50 Marathi prompts (mix: factual, creative, how-to,
  reasoning). Run base gemma-2-9b-it vs your fine-tune side by side.
- Common v2 improvements: filter IndicAlign harder (chrF++ back-translation
  filtering like Airavata did), add native (non-translated) Marathi data,
  bump LoRA r to 32, try 2 epochs on the cleaned subset.

## Cost summary

| Item | Est. |
|---|---|
| Data prep | $0 (your Mac) |
| A100 80GB × ~4 h | ~$7–8 |
| Merge/convert × 0.5 h | ~$1 |
| HF storage | $0 (private repo, free) |
| **Total** | **~$10, well under budget** |
