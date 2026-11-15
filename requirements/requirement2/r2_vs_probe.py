"""
R2, second half: does the model's stated answer agree with what the probe reads
from its hidden state (brief Section 7.1: Cohen's kappa between stated answer and probe)?

Joins, by row id:
  results/rq1/<qwen|llama>/seed_<s>/predictions.jsonl      (R1 pooled binary probe, test rows)
  results/rq2/<stage>/dataset/<model>/standard/stated.jsonl (R2 stated answers)
and reports, per model (mean +/- sd over seeds) and per testbed.

Usage (from the project root):
  python requirements/requirement2/r2_vs_probe.py
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from r2_stated_answers import ROOT

MODELS = {"qwen": "Qwen2.5-7B-Instruct", "llama": "Llama-3.1-8B-Instruct"}
SEEDS = (42, 43, 44)
PRED_KEYS = ("pred", "prediction", "predicted_label", "probe_pred")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def stated_run(model_dir: str) -> Path:
    for stage in ("final", "dev"):  # prefer the GPU rerun if it exists
        p = ROOT / "results" / "rq2" / stage / "dataset" / model_dir / "standard" / "stated.jsonl"
        if p.exists():
            return p
    raise SystemExit(f"No standard R2 run found for {model_dir}")


def to_binary(label) -> str:
    return "irreversible" if label in ("irreversible", 1, True, "1") else "reversible"


def kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in ("reversible", "irreversible"))
    return None if pe == 1 else (po - pe) / (1 - pe)


def balanced_accuracy(pred: list[str], true: list[str]) -> float | None:
    recalls = []
    for c in ("reversible", "irreversible"):
        idx = [i for i, t in enumerate(true) if t == c]
        if idx:
            recalls.append(sum(pred[i] == c for i in idx) / len(idx))
    return sum(recalls) / len(recalls) if recalls else None


def metrics(rows: list[dict]) -> dict:
    t = [r["true"] for r in rows]
    s = [r["stated"] for r in rows]
    p = [r["probe"] for r in rows]
    out = {
        "n": len(rows),
        "agreement": sum(x == y for x, y in zip(s, p)) / len(rows),
        "kappa_stated_probe": kappa(s, p),
        "ba_probe": balanced_accuracy(p, t),
        "ba_stated": balanced_accuracy(s, t),
    }
    for name, cond in [("both_right", lambda r: r["probe"] == r["true"] and r["stated"] == r["true"]),
                       ("probe_right_stated_wrong", lambda r: r["probe"] == r["true"] and r["stated"] != r["true"]),
                       ("stated_right_probe_wrong", lambda r: r["probe"] != r["true"] and r["stated"] == r["true"]),
                       ("both_wrong", lambda r: r["probe"] != r["true"] and r["stated"] != r["true"])]:
        out[name] = sum(cond(r) for r in rows) / len(rows)
    return out


def mean_sd(values: list[float]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "-"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.3f} ± {sd:.3f}"


def main() -> None:
    results = {}
    keys = ["n", "agreement", "kappa_stated_probe", "ba_probe", "ba_stated",
            "both_right", "probe_right_stated_wrong", "stated_right_probe_wrong", "both_wrong"]
    for short, model_dir in MODELS.items():
        stated = {r["id"]: r for r in load_jsonl(stated_run(model_dir))}
        per_seed, per_env = [], defaultdict(list)
        missing = 0
        for seed in SEEDS:
            path = ROOT / "results" / "rq1" / short / f"seed_{seed}" / "predictions.jsonl"
            preds = load_jsonl(path)
            pkey = next(k for k in PRED_KEYS if k in preds[0])
            rows = []
            for p in preds:
                s = stated.get(p["id"])
                if s is None:
                    missing += 1
                    continue
                if s["true_label"] not in ("reversible", "irreversible"):
                    continue
                rows.append({"env": "sandbox" if s["environment"] in ("filesystem", "email_outbox", "wallet")
                             else s["environment"],
                             "true": s["true_label"], "stated": to_binary(s["stated_label"]),
                             "probe": to_binary(p[pkey])})
            per_seed.append(metrics(rows))
            for env in sorted({r["env"] for r in rows}):
                per_env[env].append(metrics([r for r in rows if r["env"] == env]))

        print(f"\n=== {model_dir}  (stated: {stated_run(model_dir).relative_to(ROOT)})")
        if missing:
            print(f"  warning: {missing} probe rows had no stated answer")
        for k in keys:
            print(f"  {k:<26} {mean_sd([m[k] for m in per_seed])}")
        print("  by testbed:  n | agreement | kappa(stated, probe) | BA probe | BA stated | probe right, stated wrong")
        for env, ms in per_env.items():
            print(f"    {env:<13} {statistics.mean(m['n'] for m in ms):>5.0f} | {mean_sd([m['agreement'] for m in ms])} | "
                  f"{mean_sd([m['kappa_stated_probe'] for m in ms])} | {mean_sd([m['ba_probe'] for m in ms])} | "
                  f"{mean_sd([m['ba_stated'] for m in ms])} | {mean_sd([m['probe_right_stated_wrong'] for m in ms])}")
        results[model_dir] = {"per_seed": per_seed, "per_testbed": per_env}

    out = ROOT / "results" / "rq2" / "r2_vs_probe.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()