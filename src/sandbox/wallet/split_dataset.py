
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "wallet"
INPUT_FILE = DATA_DIR / "wallet.jsonl"
OUTPUT_DIR = DATA_DIR / "splits"
MANIFEST_FILE = OUTPUT_DIR / "wallet_split_manifest.json"

SEEDS = [42, 43, 44]

TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def validate_dataset(examples: list[dict[str, Any]]) -> None:
    if not examples:
        raise ValueError("Dataset is empty")

    required_fields = {
        "example_id",
        "scenario_id",
        "binary_label",
        "trinary_label",
    }

    for index, example in enumerate(examples):
        missing = required_fields - example.keys()
        if missing:
            raise ValueError(
                f"Example at row {index + 1} is missing fields: "
                f"{sorted(missing)}"
            )

    example_ids = [example["example_id"] for example in examples]

    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Duplicate example_id values found")

    scenario_ids = [example["scenario_id"] for example in examples]

    if len(set(scenario_ids)) < 10:
        raise ValueError(
            f"Only {len(set(scenario_ids))} scenarios; "
            "at least 10 are required for this split."
        )


def split_by_scenario(
    examples: list[dict[str, Any]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    Split by scenario so examples from the same scenario cannot
    leak across train, validation, and test.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for example in examples:
        groups[example["scenario_id"]].append(example)

    scenario_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(scenario_ids)

    n_scenarios = len(scenario_ids)

    n_test = max(1, round(n_scenarios * TEST_FRACTION))
    n_val = max(1, round(n_scenarios * VAL_FRACTION))

    # Ensure at least one scenario remains for training.
    while n_test + n_val >= n_scenarios:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError(
                "Not enough scenarios for train/validation/test splits"
            )

    test_scenarios = set(scenario_ids[:n_test])
    val_scenarios = set(scenario_ids[n_test:n_test + n_val])
    train_scenarios = set(scenario_ids[n_test + n_val:])

    splits = {
        "train": [
            example for example in examples
            if example["scenario_id"] in train_scenarios
        ],
        "validation": [
            example for example in examples
            if example["scenario_id"] in val_scenarios
        ],
        "test": [
            example for example in examples
            if example["scenario_id"] in test_scenarios
        ],
    }

    # Verify scenario-level separation.
    scenario_sets = {
        name: {example["scenario_id"] for example in rows}
        for name, rows in splits.items()
    }

    names = list(scenario_sets)

    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = scenario_sets[left] & scenario_sets[right]
            if overlap:
                raise AssertionError(
                    f"Scenario leakage between {left} and {right}: "
                    f"{sorted(overlap)}"
                )

    # Verify all examples appear exactly once.
    split_example_ids = [
        example["example_id"]
        for rows in splits.values()
        for example in rows
    ]

    if (
        len(split_example_ids) != len(examples)
        or set(split_example_ids) != {
            example["example_id"] for example in examples
        }
    ):
        raise AssertionError("Examples were lost or duplicated")

    return splits


def summarize_partition(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "examples": len(rows),
        "scenarios": sorted({
            example["scenario_id"] for example in rows
        }),
        "scenario_count": len({
            example["scenario_id"] for example in rows
        }),
        "binary_labels": dict(
            Counter(example["binary_label"] for example in rows)
        ),
        "trinary_labels": dict(
            Counter(example["trinary_label"] for example in rows)
        ),
    }


def main() -> None:
    examples = read_jsonl(INPUT_FILE)
    validate_dataset(examples)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "split_method": "scenario-level",
        "fractions": {
            "train": TRAIN_FRACTION,
            "validation": VAL_FRACTION,
            "test": TEST_FRACTION,
        },
        "seeds": SEEDS,
        "splits": {},
    }

    for seed in SEEDS:
        splits = split_by_scenario(examples, seed)
        seed_dir = OUTPUT_DIR / f"seed_{seed}"

        for name, rows in splits.items():
            write_jsonl(seed_dir / f"{name}.jsonl", rows)

        manifest["splits"][str(seed)] = {
            name: summarize_partition(rows)
            for name, rows in splits.items()
        }

        print(f"\nSeed {seed}")
        for name, rows in splits.items():
            summary = summarize_partition(rows)
            print(
                f"  {name}: {summary['examples']} examples, "
                f"{summary['scenario_count']} scenarios"
            )
            print(f"    Binary: {summary['binary_labels']}")
            print(f"    Trinary: {summary['trinary_labels']}")

    with MANIFEST_FILE.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print("\nWallet dataset splitting complete.")
    print(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
