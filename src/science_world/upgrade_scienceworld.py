"""
Convert a ScienceWorld JSONL made by the earlier labeller into the shared
schema, without rerunning ScienceWorld.

- Labels are recomputed from the stored search results with the shared rule
  in labels.py (the earlier file used the old partial rule).
- scenario_id becomes "scienceworld:<task>", so all variations of a task land
  in the same split. Variations of one task are near-duplicates, so splitting
  by variation would leak them (Section 4.3 of the brief: split by task).
- Adds group_id, condition, action_unit and the top-level cost fields.

Place this file in src/scienceworld/. It uses the shared labels.py and
schema.py from src/sandbox/common/ (or src/sandbox_v2/common/).

Usage (from the project root):
    python src/scienceworld/upgrade_scienceworld.py data/scienceworld/scienceworld_reversibility_v2.jsonl
    # writes ..._shared.jsonl next to the input
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _name in ("sandbox", "sandbox_v2"):
    _common = _SRC / _name / "common"
    if (_common / "labels.py").exists():
        sys.path.insert(0, str(_common))
        break
else:
    raise ImportError(
        f"Could not find the shared common/ folder under {_SRC}/sandbox "
        f"or {_SRC}/sandbox_v2."
    )

from labels import HORIZONS, derive_labels  # noqa: E402
from schema import check_dataset, read_jsonl, write_jsonl  # noqa: E402

SEARCH_DEPTH = 3  # depth the ScienceWorld labeller searches to


def upgrade(row: dict) -> dict:
    task = row.get("task_name") or row["scenario_id"].split(":")[0]
    new = dict(row)
    new["scenario_id"] = f"scienceworld:{task}"
    new["group_id"] = row["id"].rsplit(":", 1)[0]
    new["condition"] = "none"
    new["action_unit"] = "single"
    new["horizon"] = 3

    search = row.get("search")
    if row.get("label_method") == "rule" or not search:
        new.update({
            "label": "irreversible",
            "binary_label": "irreversible",
            "labels_by_horizon": {str(k): "irreversible" for k in HORIZONS},
            "extended_horizon": SEARCH_DEPTH,
            "exact_cost": None,
            "approx_cost": None,
            "exhausted_depth": None,
            "label_method": "rule",
        })
        return new

    exhausted = 0 if search.get("branching_truncated") else search["exhausted_depth"]
    labels = derive_labels(
        search.get("exact_cost"),
        search.get("approx_cost"),
        exhausted,
        SEARCH_DEPTH,
    )
    new.update(labels)
    new["label_method"] = "search"
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    out = args.out or args.input.with_name(args.input.stem + "_shared.jsonl")
    old = read_jsonl(args.input)
    rows = [upgrade(r) for r in old]
    check_dataset(rows)
    write_jsonl(rows, out)

    changed = Counter(
        f"{a['label']} -> {b['label']}"
        for a, b in zip(old, rows) if a["label"] != b["label"]
    )
    print(f"Converted {len(rows)} rows from "
          f"{len({r['scenario_id'] for r in rows})} tasks -> {out}")
    print("Labels at k=3:", dict(Counter(r["label"] for r in rows)))
    print("Labels changed by the shared rule:", dict(changed) or "none")


if __name__ == "__main__":
    main()
