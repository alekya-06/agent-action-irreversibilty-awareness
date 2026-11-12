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
from collections import defaultdict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))  # prompts.py is shared, one level up

from prompts import (  
    STATED_VARIANTS, build_stated_messages, load_toolemu_specs, parse_stated_answer,
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

def load_rows(paths: list[str], limit: int | None, stratified: int | None = None) -> list[dict]:
    rows = []
    for path in paths:
        with open(ROOT / path, encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    rows = [r for r in rows if r["label"] != "inconclusive"]
    rng = random.Random(SEED)
    if stratified:  # balanced across testbed x label; the same subset every time
        cells = defaultdict(list)
        for r in rows:
            testbed = r["environment"] if r["environment"] in ("scienceworld", "toolemu") else "sandbox"
            cells[(testbed, r["label"])].append(r)
        per_cell = stratified // len(cells)
        chosen, ids = [], set()
        for key in sorted(cells):
            options = cells[key][:]
            rng.shuffle(options)
            for r in options[:per_cell]:
                chosen.append(r)
                ids.add(r["id"])
        rest = [r for r in rows if r["id"] not in ids]
        rng.shuffle(rest)
        return chosen + rest[:stratified - len(chosen)]
    if limit:
        rng.shuffle(rows)
        rows = rows[:limit]
    return rows

def load_task_rows(path: Path) -> list[dict]:
    """R3 task versions as R2 questions: the state text plus the tempting shortcut action."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = json.loads(line)
            rows.append({
                "id": t["task_id"], "environment": t["environment"],
                "scenario_id": t["pair_id"], "group_id": t["pair_id"],
                "condition": t["condition"], "label": t["shortcut_label"],
                "state_text": t["state_text"], "action_text": t["shortcut_text"],
                "details": {},
            })
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
    parser.add_argument("--variant", choices=sorted(STATED_VARIANTS), default="standard",
                        help="question variant: standard, reversed (option order), polarity (yes = irreversible)")
    parser.add_argument("--stratified", type=int, help="a fixed subset of N rows balanced across testbed x label")
    parser.add_argument("--tasks", type=Path,
                        help="ask about the R3 task shortcuts instead of the dataset (for K)")
    args = parser.parse_args()
    args.model = MODELS.get(args.model, args.model)

    mapping = STATED_VARIANTS[args.variant][1]
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    device, dtype = pick_device()
    short = (args.model.split("/")[-1] + ("" if args.variant == "standard" else f"_{args.variant}")
             + ("_tasks" if args.tasks else "") + ("" if device == "cuda" else "_dev"))   
    out = args.out or ROOT / "results" / "rq2" / short / "stated.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    #rows = [r for r in load_rows(args.files, args.limit, args.stratified) if r["id"] not in done]
    source = load_task_rows(args.tasks) if args.tasks else load_rows(args.files, args.limit, args.stratified)
    rows = [r for r in source if r["id"] not in done]
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
        "variant": args.variant, 
        "stratified": args.stratified,
        "tasks": str(args.tasks) if args.tasks else None,
    }, indent=2), encoding="utf-8")

    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            texts = [tokenizer.apply_chat_template(build_stated_messages(r, specs, variant=args.variant),
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
                    "stated_label": parse_stated_answer(reply, variant=args.variant),
                    "option_probs": {mapping[L]: (p / mass if mass else None) for L, p in raw.items()},
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