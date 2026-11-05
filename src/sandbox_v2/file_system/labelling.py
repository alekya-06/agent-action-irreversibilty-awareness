"""
Shared-definition labelling for the file system sandbox.

Uses the existing transition, undo, snapshot and history-inverse logic
(transitions.py, recovery.py, bfs.py). The legacy run_experiment in
experiments.py is left unchanged so its tests still pass.

Exact restoration:       canonical_key() equals the pre-action state.
Approximate restoration: same directories and file contents; metadata
                         or dependency references may differ.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from labels import EXTENDED_HORIZON, dual_bfs, labels_from_outcome  # noqa: E402

from bfs import _history_recovery_candidate, _search_state_key  # noqa: E402
from recovery import (  # noqa: E402
    apply_recovery_action,
    execute_with_undo,
    get_recovery_actions,
    save_snapshot,
)
from state import FileSystemState  # noqa: E402
from transitions import transition  # noqa: E402

CONDITIONS = {
    "none": (False, False),
    "undo": (False, True),
    "snapshot": (True, False),
    "undo+snapshot": (True, True),
    "stale_snapshot": (True, False),
}


def approx_key(fs: FileSystemState) -> tuple:
    return (
        tuple(sorted(fs.directories)),
        tuple(sorted(fs.files.items())),
    )

def touched_files(fs: FileSystemState, actions: list[dict]) -> list[str]:
    """Files in fs that the actions delete, edit, move or rename (directly or via a directory)."""
    paths: set[str] = set()
    for action in actions:
        kind = action["type"]
        if kind in ("delete_file", "edit_file"):
            paths.add(action["path"])
        elif kind == "move_file":
            paths.add(action["source"])
        elif kind in ("rename", "delete_directory"):
            target = action["source"] if kind == "rename" else action["path"]
            if target in fs.files:
                paths.add(target)
            else:
                prefix = target.rstrip("/") + "/"
                paths |= {p for p in fs.files if p.startswith(prefix)}
    return sorted(p for p in paths if p in fs.files)


def stale_version(fs: FileSystemState, actions: list[dict]) -> FileSystemState:
    """The file system as it was before the touched files were created."""
    stale = fs.clone()
    for path in touched_files(fs, actions):
        stale.files.pop(path, None)
        stale.metadata.pop(path, None)
        stale.dependencies.pop(path, None)
    return stale

def label_actions(
    scenario_id: str,
    initial_fs: FileSystemState,
    actions: list[dict],
    condition: str,
    max_depth: int = EXTENDED_HORIZON,
) -> dict[str, Any]:
    """Run the action(s) under a resource condition and label recovery."""
    provide_snapshot, record_undo = CONDITIONS[condition]
    initial_fs.validate()
    original = initial_fs.clone()
    current = initial_fs.clone()

    snapshots: dict = {}
    undo_records: dict = {}
    history: list[dict] = []

    if provide_snapshot:
        base = stale_version(current, actions) if condition == "stale_snapshot" else current
        snapshots = save_snapshot(base, snapshots, f"{scenario_id}_snapshot")

    for i, action in enumerate(actions):
        if record_undo:
            current, undo_records, record = execute_with_undo(
                current, action, f"{scenario_id}_action_{i + 1}", undo_records
            )
        else:
            current, record = transition(current, action)
        history.append(record)

    target_exact = original.canonical_key()
    target_approx = approx_key(original)

    def successors(node):
        fs, records, history_index = node
        candidates = sorted(
            get_recovery_actions(fs, snapshots, records),
            key=lambda a: 0 if a["type"] == "restore_snapshot" else 1,
        )
        inverse = _history_recovery_candidate(fs, history, history_index)
        if inverse is not None:
            candidates.append({"type": "ordinary", "action": inverse})

        for candidate in candidates:
            try:
                if candidate["type"] == "ordinary":
                    next_fs, _ = transition(fs, candidate["action"])
                    next_records = records
                else:
                    next_fs, next_records = apply_recovery_action(
                        fs, snapshots, records, candidate
                    )
            except (ValueError, KeyError, TypeError):
                continue

            next_index = history_index
            if (
                next_index > 0
                and next_fs.canonical_key() == history[next_index - 1]["before"]
            ):
                next_index -= 1
            yield candidate, (next_fs, next_records, next_index)

    outcome = dual_bfs(
        start=(current, undo_records, len(history)),
        successors=successors,
        is_exact=lambda n: n[0].canonical_key() == target_exact,
        is_approx=lambda n: approx_key(n[0]) == target_approx,
        key=lambda n: _search_state_key(*n),
        max_depth=max_depth,
    )

    return {
        "labels": labels_from_outcome(outcome),
        "resulting_fs": current,
        "recovery_path": outcome.exact_path or outcome.approx_path,
        "nodes": outcome.nodes,
    }
