"""
File system dataset in the shared schema (see ../common/schema.py).

Each scenario is run under four recovery conditions (none, undo, snapshot,
undo+snapshot). The four variants share a group_id and a scenario_id, so
they form a minimal-pair group and always land in the same split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from schema import check_dataset, make_record, summarise, write_jsonl  # noqa: E402

from labelling import CONDITIONS, label_actions  # noqa: E402
from render import render_actions, render_state  # noqa: E402
from state import FileSystemState  # noqa: E402


DATASET_VERSION = "v3"

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
# Dataset generation
# ---------------------------------------------------------------------

def generate_dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        group_id = f"filesystem:{scenario['group_id']}"
        actions = (
            [scenario["action"]] if "action" in scenario
            else list(scenario["actions"])
        )
        unit = "single" if len(actions) == 1 else "sequence"

        for condition in CONDITIONS:
            initial_fs = scenario["fs"]()
            result = label_actions(
                scenario_id=scenario["group_id"],
                initial_fs=initial_fs,
                actions=actions,
                condition=condition,
            )

            rows.append(make_record(
                id=f"{group_id}:{condition}",
                environment="filesystem",
                scenario_id=group_id,
                group_id=group_id,
                condition=condition,
                state_text=render_state(initial_fs, condition),
                action_text=render_actions(actions),
                action_unit=unit,
                labels=result["labels"],
                source=f"sandbox_{DATASET_VERSION}",
                details={
                    "actions": actions,
                    "recovery_path": result["recovery_path"],
                    "search_nodes": result["nodes"],
                },
            ))

    return rows


def main() -> None:
    rows = generate_dataset()
    check_dataset(rows)

    dataset_path = OUTPUT_DIR / f"filesystem_{DATASET_VERSION}.jsonl"
    summary_path = OUTPUT_DIR / f"filesystem_{DATASET_VERSION}_summary.json"
    write_jsonl(rows, dataset_path)

    summary = summarise(rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Dataset: {dataset_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
