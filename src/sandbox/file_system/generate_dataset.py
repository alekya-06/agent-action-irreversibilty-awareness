from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from state import FileSystemState
from experiments import run_experiment


HORIZON = 3
DATASET_VERSION = "v2"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "filesystem"


# ---------------------------------------------------------------------
# Initial filesystem states
# ---------------------------------------------------------------------

def make_basic_fs() -> FileSystemState:
    return FileSystemState(
        directories={"/", "/reports", "/archive"},
        files={
            "/reports/a.csv": "original contents",
            "/reports/b.csv": "secondary contents",
            "/archive/index.csv": "archive index",
        },
    )


def make_nested_fs() -> FileSystemState:
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
                "/reports/summary.csv",
            ],
            "/archive/index.csv": [
                "/reports/2025/january.csv",
            ],
        },
    )


# ---------------------------------------------------------------------
# Scenario definitions
# Each group has four resource variants.
# ---------------------------------------------------------------------

SCENARIOS: list[dict[str, Any]] = [
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
    {
        "group_id": "four_step_move_chain",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "move_file",
                "source": "/reports/a.csv",
                "destination": "/archive/a.csv",
            },
            {
                "type": "move_file",
                "source": "/archive/a.csv",
                "destination": "/reports/a.csv",
            },
            {
                "type": "move_file",
                "source": "/reports/a.csv",
                "destination": "/archive/a.csv",
            },
            {
                "type": "move_file",
                "source": "/archive/a.csv",
                "destination": "/reports/final.csv",
            },
        ],
    },
    {
        "group_id": "five_step_move_chain",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "move_file",
                "source": "/reports/a.csv",
                "destination": "/archive/a.csv",
            },
            {
                "type": "move_file",
                "source": "/archive/a.csv",
                "destination": "/reports/a.csv",
            },
            {
                "type": "move_file",
                "source": "/reports/a.csv",
                "destination": "/archive/a.csv",
            },
            {
                "type": "move_file",
                "source": "/archive/a.csv",
                "destination": "/reports/a.csv",
            },
            {
                "type": "move_file",
                "source": "/reports/a.csv",
                "destination": "/archive/final.csv",
            },
        ],
    },
    {
        "group_id": "four_step_mixed_chain",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "rename",
                "source": "/reports/a.csv",
                "new_name": "stage1.csv",
            },
            {
                "type": "move_file",
                "source": "/reports/stage1.csv",
                "destination": "/archive/stage1.csv",
            },
            {
                "type": "rename",
                "source": "/archive/stage1.csv",
                "new_name": "stage2.csv",
            },
            {
                "type": "move_file",
                "source": "/archive/stage2.csv",
                "destination": "/reports/final.csv",
            },
        ],
    },
    {
        "group_id": "five_step_rename_chain",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "rename",
                "source": "/reports/a.csv",
                "new_name": "stage1.csv",
            },
            {
                "type": "rename",
                "source": "/reports/stage1.csv",
                "new_name": "stage2.csv",
            },
            {
                "type": "rename",
                "source": "/reports/stage2.csv",
                "new_name": "stage3.csv",
            },
            {
                "type": "rename",
                "source": "/reports/stage3.csv",
                "new_name": "stage4.csv",
            },
            {
                "type": "rename",
                "source": "/reports/stage4.csv",
                "new_name": "stage5.csv",
            },
        ],
    },
    {
        "group_id": "six_step_rename_chain",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "rename",
                "source": "/reports/a.csv",
                "new_name": "stage1.csv",
            },
            {
                "type": "rename",
                "source": "/reports/stage1.csv",
                "new_name": "stage2.csv",
            },
            {
                "type": "rename",
                "source": "/reports/stage2.csv",
                "new_name": "stage3.csv",
            },
            {
                "type": "rename",
                "source": "/reports/stage3.csv",
                "new_name": "stage4.csv",
            },
            {
                "type": "rename",
                "source": "/reports/stage4.csv",
                "new_name": "stage5.csv",
            },
            {
                "type": "rename",
                "source": "/reports/stage5.csv",
                "new_name": "stage6.csv",
            },
        ],
    },
    {
        "group_id": "five_step_mixed_chain",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "rename",
                "source": "/reports/a.csv",
                "new_name": "stage1.csv",
            },
            {
                "type": "move_file",
                "source": "/reports/stage1.csv",
                "destination": "/archive/stage1.csv",
            },
            {
                "type": "rename",
                "source": "/archive/stage1.csv",
                "new_name": "stage2.csv",
            },
            {
                "type": "move_file",
                "source": "/archive/stage2.csv",
                "destination": "/reports/stage2.csv",
            },
            {
                "type": "rename",
                "source": "/reports/stage2.csv",
                "new_name": "final.csv",
            },
        ],
    },
    {
        "group_id": "six_step_mixed_chain",
        "fs": make_basic_fs,
        "actions": [
            {
                "type": "rename",
                "source": "/reports/a.csv",
                "new_name": "stage1.csv",
            },
            {
                "type": "move_file",
                "source": "/reports/stage1.csv",
                "destination": "/archive/stage1.csv",
            },
            {
                "type": "rename",
                "source": "/archive/stage1.csv",
                "new_name": "stage2.csv",
            },
            {
                "type": "move_file",
                "source": "/archive/stage2.csv",
                "destination": "/reports/stage2.csv",
            },
            {
                "type": "rename",
                "source": "/reports/stage2.csv",
                "new_name": "stage3.csv",
            },
            {
                "type": "move_file",
                "source": "/reports/stage3.csv",
                "destination": "/archive/stage3.csv",
            },
        ],
    },
]


# ---------------------------------------------------------------------
# Recovery resources
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

def generate_dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        group_id = scenario["group_id"]

        for resources in RESOURCE_CONFIGS:
            resource_name = resources["name"]

            # Shared identity for the underlying scenario.
            scenario_id = group_id

            # Unique identity for this resource-specific example.
            example_id = f"{group_id}__{resource_name}"

            kwargs: dict[str, Any] = {
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

            # Stable identifiers and metadata for auditing and splitting.
            result["scenario_id"] = scenario_id
            result["example_id"] = example_id
            result["scenario_group_id"] = group_id
            result["resource_condition"] = resource_name
            result["dataset_version"] = DATASET_VERSION

            rows.append(result)

    return rows


# ---------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------

def validate_dataset(rows: list[dict[str, Any]]) -> bool:
    assert rows, "Dataset is empty."

    required_fields = {
        "scenario_id",
        "example_id",
        "scenario_group_id",
        "resource_condition",
        "binary_label",
        "trinary_label",
        "recovery_cost",
        "recovery_path",
        "max_recovery_steps",
    }

    for index, row in enumerate(rows):
        missing = required_fields - row.keys()
        assert not missing, (
            f"Example at index {index} is missing fields: "
            f"{sorted(missing)}"
        )

    example_ids = [row["example_id"] for row in rows]
    assert len(example_ids) == len(set(example_ids)), (
        "Duplicate example IDs found."
    )

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
            assert row["binary_label"] == "irreversible"
            assert row["trinary_label"] == "irreversible"
        else:
            assert isinstance(cost, int) and cost >= 0
            assert row["recovery_path"] is not None

            if cost <= HORIZON:
                assert row["binary_label"] == "reversible"
                assert row["trinary_label"] == "reversible"
            else:
                assert row["binary_label"] == "irreversible"
                assert row["trinary_label"] == "partially_reversible"

    expected_conditions = {"none", "undo", "snapshot", "both"}
    groups: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        groups.setdefault(row["scenario_group_id"], []).append(row)

    for group_id, group_rows in groups.items():
        conditions = {
            row["resource_condition"] for row in group_rows
        }

        assert conditions == expected_conditions, (
            f"Unexpected resource conditions in group {group_id}: "
            f"{conditions}"
        )
        assert len(group_rows) == len(expected_conditions), (
            f"Expected four examples in group {group_id}, "
            f"found {len(group_rows)}."
        )

        assert all(
            row["scenario_id"] == group_id
            for row in group_rows
        ), f"Inconsistent scenario_id in group {group_id}."

    return True


# ---------------------------------------------------------------------
# Summary and output
# ---------------------------------------------------------------------

def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_version": DATASET_VERSION,
        "num_examples": len(rows),
        "num_unique_scenarios": len(
            {row["scenario_id"] for row in rows}
        ),
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


def main() -> None:
    rows = generate_dataset()
    validate_dataset(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = OUTPUT_DIR / "filesystem_expanded_v2.jsonl"
    summary_path = OUTPUT_DIR / "filesystem_expanded_v2_summary.json"

    # Write artifacts only after generation and validation succeed.
    with dataset_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(rows)

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Dataset generation and validation succeeded.")
    print(f"Examples: {summary['num_examples']}")
    print(f"Unique scenarios: {summary['num_unique_scenarios']}")
    print(f"Scenario groups: {summary['num_scenario_groups']}")
    print(f"Binary labels: {summary['binary_label_counts']}")
    print(f"Trinary labels: {summary['trinary_label_counts']}")
    print(f"Recovery costs: {summary['recovery_cost_counts']}")
    print(
        "Resource conditions:",
        summary["resource_condition_counts"],
    )
    print("Validation: PASSED")
    print(f"Dataset: {dataset_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
