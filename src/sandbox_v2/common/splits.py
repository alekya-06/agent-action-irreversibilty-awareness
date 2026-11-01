"""
Group-aware train/validation/test splits for any shared-schema JSONL files.

Splitting is by scenario_id: every row of a scenario (including all its
minimal-pair variants) lands in the same split, so a probe cannot memorise
a scenario (Section 4.3 of the brief). Three seeds by default (Section 7.4).

Usage:
    python splits.py data/filesystem/filesystem_v3.jsonl \\
        data/email_outbox/email_outbox_v3.jsonl data/wallet/wallet_v3.jsonl \\
        --out-dir data/sandbox_splits
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from schema import check_dataset, read_jsonl, write_jsonl


def split_by_scenario(
    rows: list[dict],
    seed: int,
    train: float = 0.8,
    validation: float = 0.1,
) -> dict[str, list[dict]]:
    scenarios = sorted({r["scenario_id"] for r in rows})
    if len(scenarios) < 3:
        raise ValueError("Need at least three scenarios to split.")

    rng = random.Random(seed)
    rng.shuffle(scenarios)

    n_test = max(1, round(len(scenarios) * (1 - train - validation)))
    n_val = max(1, round(len(scenarios) * validation))
    test_ids = set(scenarios[:n_test])
    val_ids = set(scenarios[n_test:n_test + n_val])

    out = {"train": [], "validation": [], "test": []}
    for row in rows:
        sid = row["scenario_id"]
        name = "test" if sid in test_ids else "validation" if sid in val_ids else "train"
        out[name].append(row)

    seen = [{r["scenario_id"] for r in part} for part in out.values()]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-aware splits by scenario_id.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--validation", type=float, default=0.1)
    args = parser.parse_args()

    rows = [row for path in args.inputs for row in read_jsonl(path)]
    check_dataset(rows)

    manifest = {"inputs": [str(p) for p in args.inputs], "seeds": {}}
    for seed in args.seeds:
        parts = split_by_scenario(rows, seed, args.train, args.validation)
        seed_dir = args.out_dir / f"seed_{seed}"
        for name, part in parts.items():
            write_jsonl(part, seed_dir / f"{name}.jsonl")
        manifest["seeds"][seed] = {
            name: {
                "rows": len(part),
                "scenarios": len({r["scenario_id"] for r in part}),
                "labels": dict(Counter(r["label"] for r in part)),
            }
            for name, part in parts.items()
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
