"""
R2: ask the model directly whether each action can be undone.

For every labelled row the model sees the same state and action as the probe,
then a three-option question (prompts.build_stated_messages). Saved per row:
the full prompt, the raw reply, the parsed label (None if unparseable; counted,
never dropped) and the probability of each option letter at the first answer token.

Usage (from the project root):
  python requirements/requirement2/r2_stated_answers.py --limit 20          # quick test with Qwen
  python requirements/requirement2/r2_stated_answers.py                     # full run with Qwen
  python requirements/requirement2/r2_stated_answers.py --model llama       # full run with Llama

Resumes automatically: ids already in the output file are skipped. Runs off a
CUDA GPU write to a "_dev" folder so they never mix with reported results.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))  # prompts.py is shared, one level up

from prompts import (  # noqa: E402
    LETTER_TO_LABEL, build_stated_messages, load_toolemu_specs, parse_stated_answer,
)

DATA_FILES = [
    "data/email_outbox/email_outbox_v5.jsonl",
    "data/filesystem/filesystem_v5.jsonl",
    "data/wallet/wallet_v4.jsonl",
    "data/scienceworld/scienceworld_reversibility_v3_shared.jsonl",
    "data/toolemu/toolemu_v3.jsonl",
]
TOOLKITS = ROOT / "external" / "ToolEmu" / "assets" / "all_toolkits.json"
SEED = 0

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",       # main model 
    "llama": "meta-llama/Llama-3.1-8B-Instruct",  # second model family
}

def load_rows(paths: list[str], limit: int | None) -> list[dict]:
    rows = []
    for path in paths:
        with open(ROOT / path, encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    rows = [r for r in rows if r["label"] != "inconclusive"]
    if limit:  # a mixed sample across testbeds for quick tests
        random.Random(SEED).shuffle(rows)
        rows = rows[:limit]
    return rows


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description="R2 stated answers.")
    parser.add_argument("--model", default="qwen",help="qwen (default), llama, or a full Hugging Face model name")
    parser.add_argument("--files", nargs="+", default=DATA_FILES)
    parser.add_argument("--out", type=Path, help="default: results/rq2/<model>[_dev]/stated.jsonl")
    parser.add_argument("--limit", type=int, help="only a random sample of N rows (for testing)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    args.model = MODELS.get(args.model, args.model)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    device, dtype = pick_device()
    short = args.model.split("/")[-1] + ("" if device == "cuda" else "_dev")
    out = args.out or ROOT / "results" / "rq2" / short / "stated.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    rows = [r for r in load_rows(args.files, args.limit) if r["id"] not in done]
    print(f"{len(rows)} rows to run ({len(done)} already in {out}); device={device}, dtype={dtype}")
    if not rows:
        return

    specs = load_toolemu_specs(TOOLKITS)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    letter_ids = {}
    for letter in "ABC":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"'{letter}' is not a single token for this tokenizer: {ids}")
        letter_ids[letter] = ids[0]

    (out.parent / "run_info.json").write_text(json.dumps({
        "model": args.model, "device": device, "dtype": str(dtype), "seed": SEED,
        "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
        "decoding": "greedy", "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "data_files": args.files,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")

    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            texts = [tokenizer.apply_chat_template(build_stated_messages(r, specs),
                                                   add_generation_prompt=True, tokenize=False)
                     for r in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     output_scores=True, return_dict_in_generate=True,
                                     pad_token_id=tokenizer.pad_token_id)
            first = torch.softmax(gen.scores[0].float(), dim=-1)
            prompt_len = enc["input_ids"].shape[1]
            for i, r in enumerate(batch):
                reply = tokenizer.decode(gen.sequences[i, prompt_len:], skip_special_tokens=True)
                raw = {L: first[i, letter_ids[L]].item() for L in "ABC"}
                mass = sum(raw.values())
                fh.write(json.dumps({
                    "id": r["id"], "environment": r["environment"], "scenario_id": r["scenario_id"],
                    "group_id": r["group_id"], "condition": r["condition"],
                    "true_label": r["label"],
                    "stated_label": parse_stated_answer(reply),
                    "option_probs": {LETTER_TO_LABEL[L]: (p / mass if mass else None) for L, p in raw.items()},
                    "letter_mass": mass,
                    "reply": reply,
                    "prompt": texts[i],
                }, ensure_ascii=False) + "\n")
            fh.flush()
            n = start + len(batch)
            rate = (time.time() - t0) / n
            print(f"[{n}/{len(rows)}] {rate:.2f}s/row, about {rate * (len(rows) - n) / 60:.0f} min left", flush=True)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()