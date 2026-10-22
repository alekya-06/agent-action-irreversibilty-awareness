
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "email_outbox" / "email_outbox.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "email_outbox" / "splits"

SEEDS = [42, 43, 44]

TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10


def load_dataset() -> list[dict[str, Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run dataset.py first."
        )

    rows = []
    with DATA_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} must be a JSON object."
                )

            for field in (
                "scenario_id",
                "scenario_group",
                "binary_label",
                "trinary_label",
            ):
                if field not in row:
                    raise ValueError(
                        f"Line {line_number} missing {field!r}."
                    )

            # Stable ID derived from the unique scenario ID.
            row["example_id"] = row["scenario_id"]
            row["scenario_group_id"] = row["scenario_group"]
            rows.append(row)

    if not rows:
        raise ValueError("Dataset is empty.")

    example_ids = [row["example_id"] for row in rows]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Duplicate example IDs found.")

    return rows


def split_groups(
    rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        groups.setdefault(row["scenario_group_id"], []).append(row)

    group_ids = sorted(groups)

    if len(group_ids) < 3:
        raise ValueError(
            "At least three scenario groups are required "
            "for train/validation/test."
        )

    shuffled = group_ids.copy()
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, round(n * TEST_FRACTION))
    n_val = max(1, round(n * VAL_FRACTION))

    if n_test + n_val >= n:
        raise ValueError(
            "Not enough groups for the requested split."
        )

    test_groups = set(shuffled[:n_test])
    val_groups = set(shuffled[n_test:n_test + n_val])
    train_groups = set(shuffled[n_test + n_val:])

    partitions = {
        "train": [
            row for row in rows
            if row["scenario_group_id"] in train_groups
        ],
        "validation": [
            row for row in rows
            if row["scenario_group_id"] in val_groups
        ],
        "test": [
            row for row in rows
            if row["scenario_group_id"] in test_groups
        ],
    }

    # Verify groups are mutually exclusive and fully assigned.
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert train_groups | val_groups | test_groups == set(group_ids)

    all_ids = []
    for name, partition_rows in partitions.items():
        if not partition_rows:
            raise ValueError(f"{name} partition is empty.")
        all_ids.extend(row["example_id"] for row in partition_rows)

    assert len(all_ids) == len(set(all_ids))
    assert len(all_ids) == len(rows)

    return {
        "seed": seed,
        "partitions": partitions,
        "group_ids": {
            "train": sorted(train_groups),
            "validation": sorted(val_groups),
            "test": sorted(test_groups),
        },
    }


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def partition_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "scenario_groups": len({
            row["scenario_group_id"] for row in rows
        }),
        "binary_label_counts": dict(Counter(
            row["binary_label"] for row in rows
        )),
        "trinary_label_counts": dict(Counter(
            row["trinary_label"] for row in rows
        )),
        "example_ids": sorted(
            row["example_id"] for row in rows
        ),
    }


def main() -> None:
    rows = load_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset_path": str(DATA_PATH),
        "num_rows": len(rows),
        "num_scenario_groups": len({
            row["scenario_group_id"] for row in rows
        }),
        "split_method": "scenario_grouped_train_validation_test",
        "fractions": {
            "train": TRAIN_FRACTION,
            "validation": VAL_FRACTION,
            "test": TEST_FRACTION,
        },
        "seeds": SEEDS,
        "splits": {},
    }

    for seed in SEEDS:
        result = split_groups(rows, seed)
        seed_dir = OUTPUT_DIR / f"seed_{seed}"

        split_info = {
            "seed": seed,
            "group_ids": result["group_ids"],
            "partitions": {},
        }

        for name, partition_rows in result["partitions"].items():
            output_path = seed_dir / f"{name}.jsonl"
            write_jsonl(output_path, partition_rows)

            summary = partition_summary(partition_rows)
            summary["path"] = str(output_path)
            split_info["partitions"][name] = summary

            print(
                f"Seed {seed} | {name}: "
                f"{summary['rows']} rows, "
                f"{summary['scenario_groups']} groups"
            )
            print("  Binary:", summary["binary_label_counts"])
            print("  Trinary:", summary["trinary_label_counts"])

        manifest["splits"][str(seed)] = split_info

    manifest_path = OUTPUT_DIR / "email_outbox_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved split files under: {OUTPUT_DIR}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
