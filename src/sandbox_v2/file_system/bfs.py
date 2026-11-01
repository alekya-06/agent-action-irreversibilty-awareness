from collections import deque
from copy import deepcopy
import json
import posixpath

from recovery import (
    get_recovery_actions,
    apply_recovery_action,
)
from transitions import transition


def _search_state_key(fs, undo_records, history_index):
    """Identify a node by filesystem state, recovery resources, and history."""
    undo_key = json.dumps(
        undo_records,
        sort_keys=True,
        separators=(",", ":"),
    )
    return fs.canonical_key(), undo_key, history_index


def _inverse_action(record):
    """Construct a direct inverse for supported move/rename actions."""
    action = record["action"]
    action_type = action["type"]

    if action_type == "move_file":
        return {
            "type": "move_file",
            "source": action["destination"],
            "destination": action["source"],
        }

    if action_type == "rename":
        source = action["source"]
        parent = posixpath.dirname(source) or "/"

        return {
            "type": "rename",
            "source": posixpath.join(
                parent, action["new_name"]
            ),
            "new_name": posixpath.basename(source),
        }

    return None


def _history_recovery_candidate(fs, history, history_index):
    """Return a valid inverse of the most recent unreversed action."""
    if history_index <= 0:
        return None

    record = history[history_index - 1]
    inverse = _inverse_action(record)

    if inverse is None:
        return None

    try:
        candidate_fs, _ = transition(fs, inverse)
    except (ValueError, KeyError, TypeError):
        return None

    if candidate_fs.canonical_key() != record["before"]:
        return None

    return inverse


def _ordinary_recovery_candidates(original_fs, current_fs):
    """
    Generate a finite set of non-destructive recovery candidates.

    Candidate names come from original paths and a fixed temporary-name
    vocabulary. No arbitrary rename suffixes are generated.
    """
    candidates = []

    current_paths = (
        current_fs.directories | set(current_fs.files)
    )
    original_paths = (
        original_fs.directories | set(original_fs.files)
    )

    # Fixed vocabulary: independent of the current path names.
    temp_names = {
        f".__recovery_tmp_{i}__"
        for i in range(
            len(original_paths) + len(current_paths) + 2
        )
    }

    target_names = {
        posixpath.basename(path)
        for path in original_paths
        if path != "/"
    }

    allowed_names = target_names | temp_names

    # Rename files/directories to an original basename or a temporary name.
    for source in sorted(current_paths - {"/"}):
        parent = posixpath.dirname(source) or "/"
        current_name = posixpath.basename(source)

        for new_name in sorted(allowed_names):
            if new_name == current_name:
                continue

            candidates.append({
                "type": "rename",
                "source": source,
                "new_name": new_name,
            })

    # Move existing files toward original file paths.
    for source in sorted(current_fs.files):
        for destination in sorted(original_fs.files):
            if source != destination:
                candidates.append({
                    "type": "move_file",
                    "source": source,
                    "destination": destination,
                })

    # Allow files to be moved temporarily when an original path is
    # occupied. These destinations use the same finite name vocabulary.
    for source in sorted(current_fs.files):
        for directory in sorted(current_fs.directories):
            for name in sorted(temp_names):
                candidates.append({
                    "type": "move_file",
                    "source": source,
                    "destination": posixpath.join(directory, name),
                })

    # Deduplicate deterministically.
    unique = []
    seen = set()

    for action in candidates:
        key = json.dumps(
            action,
            sort_keys=True,
            separators=(",", ":"),
        )

        if key not in seen:
            seen.add(key)
            unique.append(action)

    return unique


def find_recovery_path(
    original_fs,
    current_fs,
    snapshots,
    undo_records,
    max_steps=3,
    recovery_history=None,
):
    """
    Find a shortest valid recovery path to the exact original filesystem.

    Recovery can use:
    - Action-specific undo records.
    - Saved snapshots.
    - Reverse recorded transitions, when history is available.
    - Ordinary rename/move operations when no history is supplied.

    max_steps=None searches until a path is found or the reachable
    finite state space is exhausted.
    """
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps cannot be negative")

    original_fs.validate()
    current_fs.validate()

    history = (
        list(recovery_history)
        if recovery_history is not None
        else None
    )
    initial_history_index = (
        len(history) if history is not None else 0
    )

    if current_fs.exactly_equals(original_fs):
        return []

    initial_records = deepcopy(undo_records)

    queue = deque([
        (
            current_fs.clone(),
            initial_records,
            initial_history_index,
            [],
        )
    ])

    visited = {
        _search_state_key(
            current_fs,
            initial_records,
            initial_history_index,
        )
    }

    while queue:
        fs, records, history_index, path = queue.popleft()

        if max_steps is not None and len(path) >= max_steps:
            continue

        # Recovery-resource actions have the form:
        # {"type": "undo", "action_id": ...}
        # {"type": "restore_snapshot", "snapshot_id": ...}
        recovery_actions = get_recovery_actions(
                fs,
                snapshots,
                records,
            )
            
        # Prefer snapshot restoration over action-specific undo.
        candidates = sorted(
            recovery_actions,
            key=lambda action: (
                0 if action["type"] == "restore_snapshot" else 1
            ),
        )
        
        if history is not None:
            # With recorded history, reverse only the most recent
            # unreversed ordinary transition. Undo and snapshots
            # remain available as separate recovery mechanisms.
            inverse = _history_recovery_candidate(
                fs,
                history,
                history_index,
            )

            if inverse is not None:
                candidates.append({
                    "type": "ordinary",
                    "action": inverse,
                })
        else:
            # Without history, search ordinary recovery operations.
            # Keep rename candidates before move candidates.
            candidates.extend(
                {
                    "type": "ordinary",
                    "action": action,
                }
                for action in _ordinary_recovery_candidates(
                    original_fs,
                    fs,
                )
            )

        for candidate in candidates:
            next_records = deepcopy(records)

            try:
                if candidate["type"] == "ordinary":
                    next_fs, _ = transition(
                        fs,
                        candidate["action"],
                    )
                else:
                    # IMPORTANT: match recovery.py's actual signature:
                    # (fs, snapshots, undo_records, action)
                    next_fs, next_records = apply_recovery_action(
                        fs,
                        snapshots,
                        next_records,
                        candidate,
                    )

            except (ValueError, KeyError, TypeError):
                # This action is not valid in the current state.
                continue

            next_history_index = history_index

            if (
                history is not None
                and next_history_index > 0
                and next_fs.canonical_key()
                == history[next_history_index - 1]["before"]
            ):
                next_history_index -= 1

            next_path = path + [candidate]

            if next_fs.exactly_equals(original_fs):
                return next_path

            state_key = _search_state_key(
                next_fs,
                next_records,
                next_history_index,
            )

            if state_key in visited:
                continue

            visited.add(state_key)

            queue.append((
                next_fs,
                next_records,
                next_history_index,
                next_path,
            ))

    return None

def assess_reversibility(
    original_fs,
    current_fs,
    snapshots,
    undo_records,
    max_steps=3,
    recovery_history=None,
):
    """Assess exact restoration within the specified recovery horizon."""
    path = find_recovery_path(
        original_fs=original_fs,
        current_fs=current_fs,
        snapshots=snapshots,
        undo_records=undo_records,
        max_steps=max_steps,
        recovery_history=recovery_history,
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

