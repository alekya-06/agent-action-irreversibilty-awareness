import os
import sys

parent_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(parent_dir)

from state import FileSystemState
from recovery import (
    save_snapshot,
    execute_with_undo,
    undo_action,
    restore_snapshot,
    get_recovery_actions,
)


def make_filesystem():
    return FileSystemState(
        directories={"/", "/reports"},
        files={"/reports/a.csv": "original"},
    )


def test_save_snapshot():
    fs = make_filesystem()
    original = fs.clone()

    snapshots = save_snapshot(
        fs, {}, "before_delete"
    )

    assert "before_delete" in snapshots
    assert snapshots["before_delete"].exactly_equals(original)


def test_execute_with_undo_records_deletion():
    fs = make_filesystem()
    undo_records = {}

    fs, undo_records, _ = execute_with_undo(
        fs,
        {"type": "delete_file", "path": "/reports/a.csv"},
        "delete_001",
        undo_records,
    )

    assert "/reports/a.csv" not in fs.files
    assert "delete_001" in undo_records


def test_undo_deletion_restores_original():
    fs = make_filesystem()
    original = fs.clone()
    undo_records = {}

    fs, undo_records, _ = execute_with_undo(
        fs,
        {"type": "delete_file", "path": "/reports/a.csv"},
        "delete_001",
        undo_records,
    )

    fs, undo_records = undo_action(
        fs, undo_records, "delete_001"
    )

    assert fs.exactly_equals(original)
    assert "delete_001" not in undo_records


def test_get_recovery_actions_includes_undo():
    fs = make_filesystem()
    undo_records = {}

    fs, undo_records, _ = execute_with_undo(
        fs,
        {"type": "delete_file", "path": "/reports/a.csv"},
        "delete_001",
        undo_records,
    )

    actions = get_recovery_actions(
        fs, {}, undo_records
    )

    assert any(
        action["type"] == "undo"
        and action["action_id"] == "delete_001"
        for action in actions
    )


def test_restore_snapshot_restores_original():
    fs = make_filesystem()
    original = fs.clone()

    snapshots = save_snapshot(
        fs, {}, "before_delete"
    )

    fs, undo_records, _ = execute_with_undo(
        fs,
        {"type": "delete_file", "path": "/reports/a.csv"},
        "delete_001",
        {},
    )

    assert "/reports/a.csv" not in fs.files

    fs = restore_snapshot(
        fs, snapshots, "before_delete"
    )

    assert fs.exactly_equals(original)
    assert fs.validate()


def test_recovery_does_not_mutate_original_state():
    fs = make_filesystem()
    original = fs.clone()

    execute_with_undo(
        fs,
        {"type": "delete_file", "path": "/reports/a.csv"},
        "delete_001",
        {},
    )

    assert fs.exactly_equals(original)
