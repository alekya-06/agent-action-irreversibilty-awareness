"""
R3 / AB4: run the agent loop on all task rows for one model and one prompt mode.

Each run is written to disk as soon as it finishes (atomic write), so a crash loses at
most one run. --resume skips runs already on disk. After the runs, pairs.jsonl is
rewritten from all runs on disk, and the teacher-forced decision scores are computed.

Output (results/rq3/<model>[/careful]/):
  runs/<task_id>.json        full run record: every prompt, output, call, observation, state
  pairs.jsonl                {"pair_id", "careful_irreversible", "careful_reversible", ...}
  decision_scores.jsonl      log P(shortcut call) and log P(safe call) at the first prompt
results/rq3/env_versions.json  every installed library version, Python, CUDA, GPU, git commit

Runs off CUDA go to "<model>_dev" (CPU only; MPS is never used), so they never mix with
reported results.

Usage (from the project root):
  python requirements/requirement3/run_rq3.py --model qwen --prompt-mode neutral --resume
  python requirements/requirement3/run_rq3.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 4   # CPU smoke test
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agent_loop import (ROOT, build_system_prompt, first_user_message, load_tasks, pair_rows,  # noqa: E402
                        paired_conditions, run_episode)
from decision_points import all_decision_points  # noqa: E402

MODELS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct"}
OUT_ROOT = ROOT / "results" / "rq3"
MAX_NEW_TOKENS = 256


def env_versions(device: str) -> dict:
    import torch
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = None
    return {
        "python": sys.version, "platform": platform.platform(), "device": device,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
        "libraries": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                                 if d.metadata["Name"])),
    }


def seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ids(tokenizer, messages):
    out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
    ids = out["input_ids"]
    return ids[0] if ids and isinstance(ids[0], list) else ids


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> dict:
    """The single model call of the agent loop (greedy). Steering: register a forward hook
    on a decoder layer of `model` before calling this; nothing else needs to change."""
    import torch
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = torch.tensor([_ids(tokenizer, messages)], device=model.device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                             temperature=None, top_p=None, top_k=None, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return {"prompt": prompt, "output": text}


def completion_logprob(model, tokenizer, messages: list[dict], completion: str) -> dict:
    """Teacher-forced log P(completion | prompt) as the assistant reply (sum and per token)."""
    import torch
    prompt_ids = _ids(tokenizer, messages)
    comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prompt_ids + comp_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1:-1].float()
    lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(comp_ids, device=model.device)[:, None]).sum().item()
    return {"logprob": lp, "tokens": len(comp_ids), "mean_logprob": lp / len(comp_ids)}


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def run_file(run_dir: Path, task_id: str) -> Path:
    return run_dir / (task_id.replace(":", "__") + ".json")


def main() -> None:
    ap = argparse.ArgumentParser(description="R3 agent runs (AB4 with --prompt-mode careful).")
    ap.add_argument("--model", default="qwen", help="qwen, llama, or a full Hugging Face model name")
    ap.add_argument("--prompt-mode", choices=("neutral", "careful"), default="neutral")
    ap.add_argument("--limit", type=int, help="only the first N task rows (file order)")
    ap.add_argument("--resume", action="store_true", help="skip runs already on disk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--tasks", type=Path, default=ROOT / "data" / "agent_tasks" / "tasks_v1.jsonl")
    ap.add_argument("--no-scores", action="store_true", help="skip the teacher-forced decision scores")
    args = ap.parse_args()
    model_name = MODELS.get(args.model, args.model)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = ("cuda", torch.bfloat16) if torch.cuda.is_available() else ("cpu", torch.float32)
    short = model_name.split("/")[-1] + ("" if device == "cuda" else "_dev") + \
        (f"_seed{args.seed}" if args.seed else "")
    out_dir = OUT_ROOT / short / ("careful" if args.prompt_mode == "careful" else "")
    run_dir = out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    transformers.set_seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    versions = env_versions(device)
    write_json_atomic(OUT_ROOT / ("env_versions.json" if device == "cuda" else "env_versions_dev.json"), versions)

    all_rows = load_tasks(args.tasks)
    paired = paired_conditions(all_rows)
    rows = all_rows[: args.limit] if args.limit else all_rows
    todo = [r for r in rows if not (args.resume and run_file(run_dir, r["task_id"]).exists())]
    print(f"{model_name} | {args.prompt_mode} | device={device} dtype={dtype} | "
          f"{len(todo)} of {len(rows)} runs to do -> {run_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    careful = args.prompt_mode == "careful"
    backend = lambda messages: generate(model, tokenizer, messages, args.max_new_tokens)  # noqa: E731

    meta = {"model": model_name, "prompt_mode": args.prompt_mode, "seed": args.seed, "device": device,
            "dtype": str(dtype), "decoding": "greedy (do_sample=False)", "max_new_tokens": args.max_new_tokens,
            "tasks_file": str(args.tasks.relative_to(ROOT)) if args.tasks.is_relative_to(ROOT) else str(args.tasks),
            "env_versions": versions}
    failed = []
    for n, row in enumerate(todo, 1):
        t0 = time.time()
        try:
            rec = run_episode(row, backend, careful, paired[row["pair_id"]])
        except Exception as e:  # reported, never hidden; the run is retried by the next --resume
            failed.append((row["task_id"], repr(e)))
            print(f"[{n}/{len(todo)}] {row['task_id']} FAILED: {e!r}", flush=True)
            continue
        rec.update(meta, seconds=round(time.time() - t0, 1))
        write_json_atomic(run_file(run_dir, row["task_id"]), rec)
        print(f"[{n}/{len(todo)}] {row['task_id']} steps={rec['steps_used']} parse_fail={rec['parse_failures']} "
              f"irr={rec['took_irreversible_action']} careful={rec['careful']} success={rec['success']} "
              f"({rec['seconds']}s)", flush=True)

    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("*.json"))]
    pairs = pair_rows(records)
    with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    print(f"{len(records)} runs on disk, {len(pairs)} complete pairs -> {out_dir / 'pairs.jsonl'}")

    if not args.no_scores:
        score_file = out_dir / "decision_scores.jsonl"
        done = set()
        if score_file.exists():
            done = {json.loads(l)["task_id"] for l in score_file.open(encoding="utf-8") if l.strip()}
        points = all_decision_points(all_rows)
        with score_file.open("a", encoding="utf-8") as fh:
            for row in rows:
                if row["task_id"] in done:
                    continue
                p = points[row["pair_id"]]
                messages = [{"role": "system", "content": build_system_prompt(row, careful)},
                            {"role": "user", "content": first_user_message(row)}]
                s = completion_logprob(model, tokenizer, messages, p["shortcut_call"])
                a = completion_logprob(model, tokenizer, messages, p["alt_call"]) if p["alt_call"] else None
                fh.write(json.dumps({
                    "task_id": row["task_id"], "pair_id": row["pair_id"], "version": row["version"],
                    "shortcut_call": p["shortcut_call"], "alt_call": p["alt_call"],
                    "shortcut_logprob": s["logprob"], "alt_logprob": a["logprob"] if a else None,
                    "margin": s["logprob"] - a["logprob"] if a else None, "model": model_name,
                    "prompt_mode": args.prompt_mode}, ensure_ascii=False) + "\n")
        print(f"decision scores -> {score_file}")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    if failed:
        print(f"{len(failed)} runs FAILED (rerun with --resume):")
        for tid, err in failed:
            print(f"  {tid}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
