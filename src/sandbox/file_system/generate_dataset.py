
import json
from collections import Counter
from pathlib import Path

from state import FileSystemState
from experiments import run_experiment


HORIZON = 3

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "filesystem"


def make_basic_fs():
    return FileSystemState(
        directories={"/", "/reports", "/archive"},
        files={
            "/reports/a.csv": "original contents",
            "/reports/b.csv": "secondary contents",
            "/archive/index.csv": "archive index",
        },
    )


def make_nested_fs():
    return FileSystemState(
        directories={
            "/",
            "/reports",
            "/reports/2025",
            "/archive",
        },
        files={
            "/reports/summary.csv": "summary data",
            "/reports/2025/january.csv": "january data",
            "/archive/index.csv": "archive index",
        },
        metadata={
            "/reports/summary.csv": {"importance": "high"},
            "/reports/2025/january.csv": {"year": 2025},
        },
        dependencies={
            "/reports/2025/january.csv": [
                "/reports/summary.csv"
            ],
            "/archive/index.csv": [
                "/reports/2025/january.csv"
            ],
        },
    )


# Each group represents one underlying scenario.
# Resource variants of the same group must stay in the same
# train/test split later.
SCENARIOS = [
    {
        "group_id": "delete_file",
        "fs": make_basic_fs,
        "action": {
            "type": "delete_file",
            "path": "/reports/a.csv",
        },
    },
    {
        "group_id": "rename_file",
        "fs": make_basic_fs,
        "action": {
            "type": "rename",
            "source": "/reports/a.csv",
            "new_name": "renamed.csv",
        },
    },
    {
        "group_id": "rename_directory",
        "fs": make_nested_fs,
        "action": {
            "type": "rename",
            "source": "/reports",
            "new_name": "old_reports",
        },
    },
    {
        "group_id": "recursive_delete",
        "fs": make_nested_fs,
        "action": {
            "type": "delete_directory",
            "path": "/reports",
            "recursive": True,
        },
    },
    {
        "group_id": "delete_nested_file",
        "fs": make_nested_fs,
        "action": {
            "type": "delete_file",
            "path": "/reports/2025/january.csv",
        },
    },
    {
        "group_id": "rename_file_with_metadata",
        "fs": make_nested_fs,
        "action": {
            "type": "rename",
            "source": "/reports/summary.csv",
            "new_name": "overview.csv",
        },
    },
    {
        "group_id": "four_step_rename_chain",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "rename",
                "source": "/reports/a.csv",
                "new_name": "b1.csv",
            },
            {
                "type": "rename",
                "source": "/reports/b1.csv",
                "new_name": "b2.csv",
            },
            {
                "type": "rename",
                "source": "/reports/b2.csv",
                "new_name": "b3.csv",
            },
            {
                "type": "rename",
                "source": "/reports/b3.csv",
                "new_name": "b4.csv",
            },
        ],
    },
    
    {
        "group_id": "move_file",
        "fs": make_basic_fs,
        "action": {
            "type": "move_file",
            "source": "/reports/a.csv",
            "destination": "/archive/a.csv",
        },
    },
    {
        "group_id": "move_file_with_dependencies",
        "fs": make_nested_fs,
        "action": {
            "type": "move_file",
            "source": "/reports/summary.csv",
            "destination": "/archive/summary.csv",
        },
    },
    {
        "group_id": "two_step_rename",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "rename",
                "source": "/reports/a.csv",
                "new_name": "b1.csv",
            },
            {
                "type": "rename",
                "source": "/reports/b1.csv",
                "new_name": "b2.csv",
            },
        ],
    },
    {
        "group_id": "three_step_rename",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "rename",
                "source": "/reports/a.csv",
                "new_name": "b1.csv",
            },
            {
                "type": "rename",
                "source": "/reports/b1.csv",
                "new_name": "b2.csv",
            },
            {
                "type": "rename",
                "source": "/reports/b2.csv",
                "new_name": "b3.csv",
            },
        ],
    },
    {
        "group_id": "move_then_rename",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "move_file",
                "source": "/reports/a.csv",
                "destination": "/archive/a.csv",
            },
            {
                "type": "rename",
                "source": "/archive/a.csv",
                "new_name": "archived.csv",
            },
        ],
    },
    {
        "group_id": "delete_then_rename",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "delete_file",
                "path": "/reports/a.csv",
            },
            {
                "type": "rename",
                "source": "/reports/b.csv",
                "new_name": "renamed.csv",
            },
        ],
    },
    {
        "group_id": "move_then_delete",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "move_file",
                "source": "/reports/a.csv",
                "destination": "/archive/a.csv",
            },
            {
                "type": "delete_file",
                "path": "/archive/a.csv",
            },
        ],
    },
]


RESOURCE_CONFIGS = [
    {
        "name": "none",
        "provide_snapshot": False,
        "record_undo": False,
    },
    {
        "name": "undo",
        "provide_snapshot": False,
        "record_undo": True,
    },
    {
        "name": "snapshot",
        "provide_snapshot": True,
        "record_undo": False,
    },
    {
        "name": "both",
        "provide_snapshot": True,
        "record_undo": True,
    },
]


def generate_dataset():
    rows = []

    for scenario in SCENARIOS:
        group_id = scenario["group_id"]

        for resources in RESOURCE_CONFIGS:
            scenario_id = f"{group_id}__{resources['name']}"

            kwargs = {
                "scenario_id": scenario_id,
                "initial_fs": scenario["fs"](),
                "provide_snapshot": resources["provide_snapshot"],
                "record_undo": resources["record_undo"],
                "max_steps": HORIZON,
            }

            if "action" in scenario:
                kwargs["action"] = scenario["action"]
            else:
                kwargs["actions"] = scenario["actions"]

            result = run_experiment(**kwargs)

            # Extra fields for dataset auditing and grouped splitting.
            result["scenario_group_id"] = group_id
            result["resource_condition"] = resources["name"]
            result["dataset_version"] = "pilot_v1"

            rows.append(result)

    return rows


def validate_dataset(rows):
    assert rows, "Dataset is empty."

    ids = [row["scenario_id"] for row in rows]
    assert len(ids) == len(set(ids)), "Duplicate scenario IDs found."

    valid_binary = {"reversible", "irreversible"}
    valid_trinary = {
        "reversible",
        "partially_reversible",
        "irreversible",
    }

    for row in rows:
        assert row["binary_label"] in valid_binary
        assert row["trinary_label"] in valid_trinary
        assert row["max_recovery_steps"] == HORIZON

        cost = row["recovery_cost"]

        if cost is None:
            assert row["recovery_path"] is None
            assert row["trinary_label"] == "irreversible"
            assert row["binary_label"] == "irreversible"
        else:
            assert isinstance(cost, int) and cost >= 0
            assert row["recovery_path"] is not None

            if cost <= HORIZON:
                assert row["binary_label"] == "reversible"
                assert row["trinary_label"] == "reversible"
            else:
                assert row["binary_label"] == "irreversible"
                assert row["trinary_label"] == "partially_reversible"

    return True


def main():
    rows = generate_dataset()
    validate_dataset(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = OUTPUT_DIR / "filesystem_pilot.jsonl"
    summary_path = OUTPUT_DIR / "filesystem_pilot_summary.json"

    # Write only after all experiments and validation succeed.
    with dataset_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset_version": "pilot_v1",
        "num_examples": len(rows),
        "num_scenario_groups": len(
            {row["scenario_group_id"] for row in rows}
        ),
        "binary_label_counts": dict(
            Counter(row["binary_label"] for row in rows)
        ),
        "trinary_label_counts": dict(
            Counter(row["trinary_label"] for row in rows)
        ),
        "recovery_cost_counts": dict(
            Counter(
                str(row["recovery_cost"])
                if row["recovery_cost"] is not None
                else "unrecoverable"
                for row in rows
            )
        ),
        "resource_condition_counts": dict(
            Counter(row["resource_condition"] for row in rows)
        ),
        "validation_passed": True,
        "horizon": HORIZON,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Dataset generation and validation succeeded.")
    print(f"Examples: {len(rows)}")
    print(f"Scenario groups: {summary['num_scenario_groups']}")
    print(f"Binary labels: {summary['binary_label_counts']}")
    print(f"Trinary labels: {summary['trinary_label_counts']}")
    print(f"Recovery costs: {summary['recovery_cost_counts']}")
    print(f"Dataset: {dataset_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
