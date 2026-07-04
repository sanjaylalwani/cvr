#!/usr/bin/env python3
"""
Prepare Marathi instruction data from AI4Bharat IndicAlign for QLoRA SFT.

Runs on your Mac (CPU-only, just needs bandwidth + disk).
Output: train.jsonl / eval.jsonl where each line is
    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}, ...]}

Usage:
    pip install datasets
    python prepare_data.py --list                 # discover configs/splits first
    python prepare_data.py --max-examples 60000   # build the dataset
"""
import argparse
import hashlib
import json
import random
import re
import unicodedata

from datasets import get_dataset_config_names, load_dataset

DATASET = "ai4bharat/indic-align"

# Subsets worth trying for Marathi SFT. Verify names with --list first;
# the repo has evolved and config names may differ slightly.
CANDIDATE_SUBSETS = ["Dolly_T", "OpenAssistant_T", "WikiHow", "IndoWordNet", "Anudesh"]

# Common language split spellings across AI4Bharat repos
MARATHI_SPLIT_CANDIDATES = ["mar_Deva", "mar", "marathi", "mr"]

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

MIN_RESPONSE_CHARS = 40
MAX_TOTAL_CHARS = 8000          # keep well under 2048 tokens after templating
MIN_DEVANAGARI_RATIO = 0.40     # response must be mostly Marathi, not English leakage


def devanagari_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if DEVANAGARI.match(c)) / len(letters)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_conversations(example):
    """
    Normalize one IndicAlign record into a list of message dicts.
    Handles both `interactions` (list of {prompt, response}-style turns)
    and flat prompt/response style records.
    Returns None if the record can't be parsed.
    """
    messages = []

    interactions = example.get("interactions")
    if interactions:
        for turn in interactions:
            if not isinstance(turn, dict):
                return None
            user = turn.get("prompt") or turn.get("instruction") or turn.get("input")
            asst = turn.get("response") or turn.get("output") or turn.get("completion")
            if not user or not asst:
                return None
            messages.append({"role": "user", "content": clean(user)})
            messages.append({"role": "assistant", "content": clean(asst)})
        return messages or None

    user = example.get("prompt") or example.get("instruction")
    asst = example.get("response") or example.get("output")
    if user and asst:
        context = example.get("input") or example.get("context")
        if context and context.strip():
            user = f"{user}\n\n{context}"
        return [
            {"role": "user", "content": clean(user)},
            {"role": "assistant", "content": clean(asst)},
        ]
    return None


def passes_filters(messages) -> bool:
    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars > MAX_TOTAL_CHARS:
        return False
    for m in messages:
        if not m["content"]:
            return False
        if m["role"] == "assistant":
            if len(m["content"]) < MIN_RESPONSE_CHARS:
                return False
            if devanagari_ratio(m["content"]) < MIN_DEVANAGARI_RATIO:
                return False
    return True


def fingerprint(messages) -> str:
    # Dedup on the first user turn (translated sets often repeat prompts)
    first_user = next(m["content"] for m in messages if m["role"] == "user")
    return hashlib.md5(first_user.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="List configs and exit")
    ap.add_argument("--subsets", nargs="*", default=CANDIDATE_SUBSETS)
    ap.add_argument("--max-examples", type=int, default=60000)
    ap.add_argument("--eval-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    configs = get_dataset_config_names(DATASET)
    print(f"Available configs in {DATASET}:\n  " + "\n  ".join(configs))
    if args.list:
        return

    random.seed(args.seed)
    seen, records = set(), []
    stats = {}

    for subset in args.subsets:
        if subset not in configs:
            print(f"[skip] {subset}: not a config in this repo")
            continue

        ds = None
        for split in MARATHI_SPLIT_CANDIDATES:
            try:
                ds = load_dataset(DATASET, subset, split=split)
                print(f"[load] {subset} split={split}: {len(ds)} rows")
                break
            except Exception:
                continue
        if ds is None:
            print(f"[skip] {subset}: no Marathi split found "
                  f"(tried {MARATHI_SPLIT_CANDIDATES}) — check its splits manually")
            continue

        kept = 0
        for ex in ds:
            messages = extract_conversations(ex)
            if not messages or not passes_filters(messages):
                continue
            fp = fingerprint(messages)
            if fp in seen:
                continue
            seen.add(fp)
            records.append({"messages": messages, "source": subset})
            kept += 1
        stats[subset] = kept
        print(f"[keep] {subset}: {kept} conversations after filtering")

    if not records:
        raise SystemExit("No records collected — check config/split names with --list")

    random.shuffle(records)
    records = records[: args.max_examples]

    n_eval = max(200, int(len(records) * args.eval_fraction))
    eval_set, train_set = records[:n_eval], records[n_eval:]

    for name, data in [("train.jsonl", train_set), ("eval.jsonl", eval_set)]:
        with open(name, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[write] {name}: {len(data)} conversations")

    print("\nPer-subset kept counts:", json.dumps(stats, indent=2))
    print("\nNow EYEBALL the data: shuf -n 20 train.jsonl | python -m json.tool")


if __name__ == "__main__":
    main()
