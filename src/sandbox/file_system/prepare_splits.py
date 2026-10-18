
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "filesystem" / "filesystem.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "filesystem" / "filesystem_split_manifest.json"

SEEDS = [42, 43, 44]
TEST_FRACTION = 0.20


def load_dataset():
    with DATA_PATH.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        raise ValueError("Dataset is empty.")

    ids = [row["scenario_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scenario_id values found.")

    return rows


def make_grouped_split(rows, seed):
    groups = defaultdict(list)

    for row in rows:
        groups[row["scenario_group_id"]].append(row)

    group_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_test = max(1, round(len(group_ids) * TEST_FRACTION))
    test_groups = set(group_ids[:n_test])
    train_groups = set(group_ids[n_test:])

    # Check that no group appears in both partitions.
    assert train_groups.isdisjoint(test_groups)
    assert train_groups | test_groups == set(group_ids)

    train_rows = [
        row for row in rows
        if row["scenario_group_id"] in train_groups
    ]
    test_rows = [
        row for row in rows
        if row["scenario_group_id"] in test_groups
    ]

    # Every variant of each group must stay in one partition.
    for group_id in group_ids:
        partitions = {
            "train" if row["scenario_group_id"] in train_groups else "test"
            for row in groups[group_id]
        }
        assert len(partitions) == 1

    return {
        "seed": seed,
        "train_scenario_groups": sorted(train_groups),
        "test_scenario_groups": sorted(test_groups),
        "train_scenario_ids": sorted(
            row["scenario_id"] for row in train_rows
        ),
        "test_scenario_ids": sorted(
            row["scenario_id"] for row in test_rows
        ),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_binary_counts": dict(
            Counter(row["binary_label"] for row in train_rows)
        ),
        "test_binary_counts": dict(
            Counter(row["binary_label"] for row in test_rows)
        ),
        "train_trinary_counts": dict(
            Counter(row["trinary_label"] for row in train_rows)
        ),
        "test_trinary_counts": dict(
            Counter(row["trinary_label"] for row in test_rows)
        ),
    }


def main():
    rows = load_dataset()
    all_groups = {row["scenario_group_id"] for row in rows}

    splits = [
        make_grouped_split(rows, seed)
        for seed in SEEDS
    ]

    manifest = {
        "dataset_path": str(DATA_PATH),
        "num_rows": len(rows),
        "num_scenario_groups": len(all_groups),
        "test_fraction": TEST_FRACTION,
        "split_method": "scenario_grouped",
        "seeds": SEEDS,
        "splits": splits,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Dataset rows: {len(rows)}")
    print(f"Scenario groups: {len(all_groups)}")
    print(f"Saved manifest: {OUTPUT_PATH}")

    for i, split in enumerate(splits, start=1):
        print(f"\nSplit {i} (seed={split['seed']})")
        print(
            f"Train: {split['train_rows']} rows, "
            f"{len(split['train_scenario_groups'])} groups"
        )
        print(
            f"Test:  {split['test_rows']} rows, "
            f"{len(split['test_scenario_groups'])} groups"
        )
        print("Train trinary labels:", split["train_trinary_counts"])
        print("Test trinary labels: ", split["test_trinary_counts"])


if __name__ == "__main__":
    main()
