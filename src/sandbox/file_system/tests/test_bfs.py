import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from state import FileSystemState
from recovery import save_snapshot, execute_with_undo
from bfs import find_recovery_path, assess_reversibility
from transitions import transition

def make_filesystem():
    return FileSystemState(
        directories={"/", "/reports"},
        files={
            "/reports/a.csv": "original contents"
        },
    )


# ------------------------------------------
# Test A: recovery using action-specific undo
# ------------------------------------------

def test_recovery_with_undo():
    original = make_filesystem()
    snapshots = {}
    undo_records = {}
    
    current, undo_records, _ = execute_with_undo(
        original,
        {
            "type": "delete_file",
            "path": "/reports/a.csv",
        },
        "delete_001",
        undo_records
    )
    
    path = find_recovery_path(
        original_fs=original,
        current_fs=current,
        snapshots=snapshots,
        undo_records=undo_records,
        max_steps=3
    )
    
    assert path is not None
    assert len(path) == 1
    assert path[0]["type"] == "undo"
    
    print("Undo recovery path:", path)


# ------------------------------------------
# Test B: recovery using a snapshot
# ------------------------------------------

def test_recovery_with_snapshot():
    original = make_filesystem()
    snapshots = save_snapshot(
        original, {}, "snapshot_001"
    )
    
    current, undo_records, _ = execute_with_undo(
        original,
        {
            "type": "delete_file",
            "path": "/reports/a.csv",
        },
        "delete_002",
        {}
    )
    
    path = find_recovery_path(
        original_fs=original,
        current_fs=current,
        snapshots=snapshots,
        undo_records=undo_records,
        max_steps=3
    )
    
    assert path is not None
    assert len(path) == 1
    assert path[0]["type"] == "restore_snapshot"
    
    print("Snapshot recovery path:", path)


# ------------------------------------------
# Test C: no available recovery
# ------------------------------------------
def test_no_recovery_available():
    original = make_filesystem()
    
    current, undo_records, _ = execute_with_undo(
        original,
        {
            "type": "delete_file",
            "path": "/reports/a.csv",
        },
        "delete_003",
        {}
    )
    
    result = assess_reversibility(
        original_fs=original,
        current_fs=current,
        snapshots={},
        undo_records={},
        max_steps=3
    )
    
    assert result["reversible_within_horizon"] is False
    
    print("No-recovery test passed!")


def test_move_is_reversible_without_recovery_resources():
    original = FileSystemState(
        directories={"/", "/reports", "/archive"},
        files={"/reports/a.csv": "original contents"},
    )

    current, _ = transition(
        original,
        {
            "type": "move_file",
            "source": "/reports/a.csv",
            "destination": "/archive/a.csv",
        },
    )

    path = find_recovery_path(
        original_fs=original,
        current_fs=current,
        snapshots={},
        undo_records={},
        max_steps=3,
    )

    assert path is not None
    assert len(path) == 1
    assert path[0]["type"] == "ordinary"
    assert path[0]["action"]["type"] == "move_file"

    restored, _ = transition(current, path[0]["action"])
    assert restored.exactly_equals(original)


def test_rename_is_reversible_without_recovery_resources():
    original = make_filesystem()

    current, _ = transition(
        original,
        {
            "type": "rename",
            "source": "/reports/a.csv",
            "new_name": "renamed.csv",
        },
    )

    path = find_recovery_path(
        original_fs=original,
        current_fs=current,
        snapshots={},
        undo_records={},
        max_steps=3,
    )

    assert path is not None
    assert len(path) == 1
    assert path[0]["type"] == "ordinary"
    assert path[0]["action"]["type"] == "rename"

    restored, _ = transition(current, path[0]["action"])
    assert restored.exactly_equals(original)

