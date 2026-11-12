import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labelling import label_actions  # noqa: E402
from state import FileSystemState  # noqa: E402

EDIT = {"type": "edit_file", "path": "/docs/thesis.txt", "contents": "draft two"}


def make_fs():
    return FileSystemState(directories={"/", "/docs"}, files={"/docs/thesis.txt": "draft one"})


def label(condition):
    result = label_actions(scenario_id="test_edit", initial_fs=make_fs(),
                           actions=[EDIT], condition=condition)
    return result["labels"]["label"]


def test_edit_without_backup_is_irreversible():
    assert label("none") == "irreversible"


def test_edit_with_snapshot_is_reversible():
    assert label("snapshot") == "reversible"


def test_edit_with_undo_is_reversible():
    assert label("undo") == "reversible"

def test_edit_with_stale_snapshot_is_irreversible():
    assert label("stale_snapshot") == "irreversible"


def test_rename_with_stale_snapshot_is_reversible():
    rename = {"type": "rename", "source": "/docs/thesis.txt", "new_name": "final.txt"}
    result = label_actions(scenario_id="test_rename", initial_fs=make_fs(),
                           actions=[rename], condition="stale_snapshot")
    assert result["labels"]["label"] == "reversible"

def test_copy_is_reversible_without_recovery():
    copy = {"type": "copy_file", "source": "/docs/thesis.txt", "destination": "/docs/thesis_backup.txt"}
    result = label_actions(scenario_id="test_copy", initial_fs=make_fs(),
                           actions=[copy], condition="none")
    assert result["labels"]["label"] == "reversible"