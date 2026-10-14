from copy import deepcopy

from state import FileSystemState
from transitions import transition


def save_snapshot(fs, snapshots, snapshot_id):
    """Save an independent filesystem snapshot."""
    if snapshot_id in snapshots:
        raise ValueError(f"Snapshot already exists: {snapshot_id}")

    new_snapshots = deepcopy(snapshots)
    new_snapshots[snapshot_id] = fs.clone()

    return new_snapshots


def _value_entry(mapping, key):
    """Capture a dictionary entry, including missing entries."""
    if key not in mapping:
        return {"exists": False}

    return {
        "exists": True,
        "value": deepcopy(mapping[key]),
    }


def _make_undo_record(before, after, action_id, action):
    """Record the filesystem changes caused by an action."""
    record = {
        "action_id": action_id,
        "action": deepcopy(action),
        "dict_changes": {},
        "directory_changes": {},
    }

    # Record changes to files, metadata and dependencies.
    for attr in ("files", "metadata", "dependencies"):
        old_map = getattr(before, attr)
        new_map = getattr(after, attr)

        changed_keys = {
            key
            for key in set(old_map) | set(new_map)
            if old_map.get(key, _MISSING)
            != new_map.get(key, _MISSING)
        }

        record["dict_changes"][attr] = {
            key: {
                "before": _value_entry(old_map, key),
                "after": _value_entry(new_map, key),
            }
            for key in sorted(changed_keys)
        }

    # Record added or removed directories.
    changed_directories = (
        before.directories ^ after.directories
    )

    record["directory_changes"] = {
        path: {
            "before": path in before.directories,
            "after": path in after.directories,
        }
        for path in sorted(changed_directories)
    }

    return record


_MISSING = object()


def execute_with_undo(fs, action, action_id, undo_records):
    """
    Execute an ordinary action and register its inverse delta.

    Returns:
        new_fs, new_undo_records, transition_record
    """
    if action_id in undo_records:
        raise ValueError(
            f"Duplicate action ID: {action_id}"
        )

    before = fs.clone()
    new_fs, transition_record = transition(fs, action)

    record = _make_undo_record(
        before=before,
        after=new_fs,
        action_id=action_id,
        action=action,
    )

    new_undo_records = deepcopy(undo_records)
    new_undo_records[action_id] = record

    return new_fs, new_undo_records, transition_record


def undo_action(fs, undo_records, action_id):
    """Undo an action if its recorded postconditions still hold."""
    if action_id not in undo_records:
        raise ValueError(f"No undo record: {action_id}")

    new_fs = fs.clone()
    record = undo_records[action_id]

    # Check all dictionary preconditions before making changes.
    for attr, changes in record["dict_changes"].items():
        mapping = getattr(new_fs, attr)

        for key, change in changes.items():
            if _value_entry(mapping, key) != change["after"]:
                raise ValueError(
                    f"Cannot undo {action_id}: "
                    f"{attr}[{key!r}] has changed"
                )

    # Check directory preconditions.
    for path, change in record["directory_changes"].items():
        if (path in new_fs.directories) != change["after"]:
            raise ValueError(
                f"Cannot undo {action_id}: "
                f"directory {path} has changed"
            )

    # Restore original dictionary entries.
    for attr, changes in record["dict_changes"].items():
        mapping = getattr(new_fs, attr)

        for key, change in changes.items():
            before_entry = change["before"]

            if before_entry["exists"]:
                mapping[key] = deepcopy(
                    before_entry["value"]
                )
            else:
                mapping.pop(key, None)

    # Restore original directory membership.
    for path, change in record["directory_changes"].items():
        if change["before"]:
            new_fs.directories.add(path)
        else:
            new_fs.directories.discard(path)

    new_fs.validate()

    # Consume the undo record only after successful restoration.
    new_undo_records = deepcopy(undo_records)
    del new_undo_records[action_id]

    return new_fs, new_undo_records


def restore_snapshot(fs, snapshots, snapshot_id):
    """Restore the filesystem from an available snapshot."""
    if snapshot_id not in snapshots:
        raise ValueError(
            f"Snapshot unavailable: {snapshot_id}"
        )

    restored_fs = snapshots[snapshot_id].clone()
    restored_fs.validate()

    return restored_fs


def get_recovery_actions(fs, snapshots, undo_records):
    """Generate currently valid undo and snapshot actions."""
    actions = []

    for action_id in sorted(undo_records):
        try:
            undo_action(fs, undo_records, action_id)
            actions.append({
                "type": "undo",
                "action_id": action_id,
            })
        except ValueError:
            # This undo is unavailable because its preconditions fail.
            continue

    for snapshot_id in sorted(snapshots):
        if not fs.exactly_equals(snapshots[snapshot_id]):
            actions.append({
                "type": "restore_snapshot",
                "snapshot_id": snapshot_id,
            })

    return actions


def apply_recovery_action(
    fs, snapshots, undo_records, action
):
    """Execute one recovery action."""
    action_type = action["type"]

    if action_type == "undo":
        new_fs, new_undo_records = undo_action(
            fs,
            undo_records,
            action["action_id"],
        )
        return new_fs, new_undo_records

    if action_type == "restore_snapshot":
        new_fs = restore_snapshot(
            fs,
            snapshots,
            action["snapshot_id"],
        )
        # Keep history; stale undo records are checked before use.
        return new_fs, deepcopy(undo_records)

    raise ValueError(f"Unknown recovery action: {action_type}")

def get_recovery_actions(fs, snapshots, undo_records):
    actions = []

    # 1. Try restoring snapshots first
    for snapshot_id, snapshot_fs in snapshots.items():
        if not fs.exactly_equals(snapshot_fs):
            actions.append({
                "type": "restore_snapshot",
                "snapshot_id": snapshot_id
            })

    # 2. Then try undo actions
    for action_id in undo_records:
        try:
            undo_action(fs, undo_records, action_id)
            actions.append({
                "type": "undo",
                "action_id": action_id
            })
        except (ValueError, KeyError, TypeError):
            continue

    return actions

