"""
File system dataset in the shared schema (see ../../common/schema.py).

Each scenario is run under four recovery conditions (none, undo, snapshot,
undo+snapshot). The four variants share a group_id and a scenario_id, so
they form a minimal-pair group and always land in the same split.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from schema import check_dataset, make_record, summarise, write_jsonl  # noqa: E402

from labelling import CONDITIONS, label_actions, touched_files  # noqa: E402
from render import render_actions, render_state  # noqa: E402
from state import FileSystemState  # noqa: E402


DATASET_VERSION = "v5"

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
# Generated scenarios: varied filesystems, mostly deletions
# ---------------------------------------------------------------------

DIR_NAMES = ["reports", "projects", "photos", "invoices", "notes",
             "media", "src", "docs", "music", "taxes"]
FILE_STEMS = ["summary", "draft", "budget", "notes", "q3", "contract",
              "thesis", "logo", "main", "readme", "invoice_4471", "plan"]
EXTENSIONS = [".csv", ".txt", ".docx", ".pdf", ".py", ".png"]
GENERATED_FS_SCENARIOS = 60
FS_GENERATOR_SEED = 5


def _fs_factory(directories, files, metadata, dependencies):
    """Return a function that builds a fresh copy of this filesystem each call."""
    def make() -> FileSystemState:
        return FileSystemState(
            directories=set(directories),
            files=dict(files),
            metadata={k: dict(v) for k, v in metadata.items()},
            dependencies={k: list(v) for k, v in dependencies.items()},
        )
    return make


def _random_fs(rng: random.Random):
    dirs = rng.sample(DIR_NAMES, rng.randint(2, 4))
    directories = {"/"} | {f"/{d}" for d in dirs}
    if rng.random() < 0.5:
        directories.add(f"/{dirs[0]}/{rng.choice(['2024', '2025', 'old', 'final'])}")
    files: dict[str, str] = {}
    for d in sorted(directories - {"/"}):
        for _ in range(rng.randint(1, 3)):
            name = f"{rng.choice(FILE_STEMS)}{rng.choice(EXTENSIONS)}"
            files[f"{d}/{name}"] = f"contents of {name}"
    paths = sorted(files)
    metadata = {rng.choice(paths): {"importance": "high"}} if rng.random() < 0.4 else {}
    dependencies: dict[str, list[str]] = {}
    if len(paths) >= 2 and rng.random() < 0.4:
        a, b = rng.sample(paths, 2)
        dependencies[a] = [b]
    return directories, files, metadata, dependencies

EDIT_SCENARIOS = 30
EDIT_GENERATOR_SEED = 9
EDIT_SUFFIXES = [" (revised)", " - final version", " with corrections", " v2", " (shortened)"]


def build_generated_edit_scenarios(n: int | None = None, seed: int = EDIT_GENERATOR_SEED) -> list[dict[str, Any]]:
    """
    One edit_file per scenario: irreversible under 'none' (no record of the old
    text), reversible under undo or snapshot. Half append to the file, half
    replace its contents.
    """
    n = EDIT_SCENARIOS if n is None else n
    rng = random.Random(seed)
    scenarios = []
    for i in range(n):
        directories, files, metadata, dependencies = _random_fs(rng)
        path = rng.choice(sorted(files))
        if rng.random() < 0.5:
            new_contents = files[path] + rng.choice(EDIT_SUFFIXES)
        else:
            new_contents = f"new contents for {path.rsplit('/', 1)[1]}"
        scenarios.append({
            "group_id": f"gen_edit_{i + 1:03d}",
            "fs": _fs_factory(directories, files, metadata, dependencies),
            "action": {"type": "edit_file", "path": path, "contents": new_contents},
        })
    return scenarios

def build_generated_scenarios(n: int | None = None, seed: int = FS_GENERATOR_SEED) -> list[dict[str, Any]]:
    """About 65% file deletions, 15% recursive directory deletions, 20% moves/renames."""
    n = GENERATED_FS_SCENARIOS if n is None else n
    rng = random.Random(seed)
    scenarios = []
    for i in range(n):
        directories, files, metadata, dependencies = _random_fs(rng)
        paths = sorted(files)
        subdirs = sorted(directories - {"/"})
        r = rng.random()
        action = None
        if 0.8 <= r < 0.9:  # move to another directory, if the name is free there
            src = rng.choice(paths)
            parent, base = src.rsplit("/", 1)
            targets = [d for d in subdirs if d != parent and f"{d}/{base}" not in files]
            if targets:
                action = {"type": "move_file", "source": src,
                          "destination": f"{rng.choice(targets)}/{base}"}
        elif r >= 0.9:
            src = rng.choice(paths)
            action = {"type": "rename", "source": src,
                      "new_name": f"renamed_{src.rsplit('/', 1)[1]}"}
        elif 0.65 <= r < 0.8:
            action = {"type": "delete_directory", "path": rng.choice(subdirs), "recursive": True}
        if action is None:  # also the fallback when no move target exists
            action = {"type": "delete_file", "path": rng.choice(paths)}
        scenarios.append({
            "group_id": f"gen_{i + 1:03d}_{action['type']}",
            "fs": _fs_factory(directories, files, metadata, dependencies),
            "action": action,
        })
    return scenarios

# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

def generate_dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS + build_generated_scenarios() + build_generated_edit_scenarios():
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
                state_text=render_state(
                    initial_fs, condition,
                    touched_files(initial_fs, actions) if condition == "stale_snapshot" else (),
                ),
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
