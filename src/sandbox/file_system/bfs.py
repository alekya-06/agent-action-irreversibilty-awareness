from collections import deque
import json

from recovery import get_recovery_actions, apply_recovery_action


def _search_state_key(fs, undo_records):
    """
    Identify a BFS node by its filesystem and available undo records.

    Snapshots remain fixed during one search, so they do not need
    to be repeated in every node key.
    """
    undo_key = json.dumps(
        undo_records,
        sort_keys=True,
        separators=(",", ":"),
    )

    return fs.canonical_key(), undo_key


def find_recovery_path(
    original_fs,
    current_fs,
    snapshots,
    undo_records,
    max_steps=3,
):
    """
    Find the shortest recovery sequence that restores original_fs.

    Args:
        original_fs: Filesystem state before the task action.
        current_fs: Filesystem state after the task action.
        snapshots: Agent-accessible snapshot dictionary.
        undo_records: Available action-specific undo records.
        max_steps: Maximum number of recovery actions.

    Returns:
        A list of recovery actions if restoration is possible
        within max_steps; otherwise None.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    original_fs.validate()
    current_fs.validate()

    if current_fs.exactly_equals(original_fs):
        return []

    # Each queue item contains the current filesystem,
    # remaining undo records, and the path used to reach it.
    queue = deque([
        (current_fs.clone(), undo_records.copy(), [])
    ])

    visited = {
        _search_state_key(current_fs, undo_records)
    }

    while queue:
        fs, records, path = queue.popleft()

        if len(path) >= max_steps:
            continue

        # Generate only valid recovery actions.
        actions = get_recovery_actions(
            fs,
            snapshots,
            records,
        )

        for action in actions:
            next_fs, next_records = apply_recovery_action(
                fs,
                snapshots,
                records,
                action,
            )

            next_path = path + [action]

            # Goal test: exact filesystem restoration.
            if next_fs.exactly_equals(original_fs):
                return next_path

            key = _search_state_key(next_fs, next_records)

            if key in visited:
                continue

            visited.add(key)

            queue.append((
                next_fs,
                next_records,
                next_path,
            ))

    return None


def assess_reversibility(
    original_fs,
    current_fs,
    snapshots,
    undo_records,
    max_steps=3,
):
    """
    Search for exact restoration and return a structured result.

    A failed bounded search does not establish that restoration
    is impossible in principle.
    """
    path = find_recovery_path(
        original_fs=original_fs,
        current_fs=current_fs,
        snapshots=snapshots,
        undo_records=undo_records,
        max_steps=max_steps,
    )

    if path is not None:
        return {
            "reversible_within_horizon": True,
            "recovery_path": path,
            "recovery_cost": len(path),
            "max_steps": max_steps,
        }

    return {
        "reversible_within_horizon": False,
        "recovery_path": None,
        "recovery_cost": None,
        "max_steps": max_steps,
    }
