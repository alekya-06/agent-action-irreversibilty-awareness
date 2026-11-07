"""
Compare R2 runs on the rows they have in common (e.g. standard vs reversed vs polarity).

Usage (from the project root):
  python requirements/requirement2/r2_compare.py results/rq2/Llama-3.1-8B-Instruct_dev \
      results/rq2/Llama-3.1-8B-Instruct_reversed_dev results/rq2/Llama-3.1-8B-Instruct_polarity_dev
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from r2_analyze import auroc, classification


def load(run: Path) -> dict:
    with open(run / "stated.jsonl", encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(l) for l in fh if l.strip())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()

    runs = {run.name: load(run) for run in args.runs}
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    print(f"{len(common)} rows in common\n")
    print(f"{'run':<42} {'accuracy':>8} {'bal acc':>8} {'AUROC':>6}  stated")
    for name, rows in runs.items():
        sub = [rows[i] for i in common]
        cls = classification(sub)
        pairs = [(r["option_probs"]["irreversible"], r["true_label"] == "irreversible") for r in sub
                 if r["true_label"] in ("reversible", "irreversible") and r["option_probs"].get("irreversible") is not None]
        auc = auroc(pairs)
        print(f"{name:<42} {cls['accuracy']:>8.3f} {cls['balanced_accuracy']:>8.3f} "
              f"{(f'{auc:.3f}' if auc is not None else '  -  '):>6}  {dict(Counter(r['stated_label'] for r in sub))}")

    names = list(runs)
    print("\nSame label in both runs:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            same = sum(runs[names[i]][k]["stated_label"] == runs[names[j]][k]["stated_label"] for k in common)
            print(f"  {names[i]} vs {names[j]}: {same / len(common):.0%}")


if __name__ == "__main__":
    main()