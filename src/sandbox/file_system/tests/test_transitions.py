import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from state import FileSystemState
from transitions import transition


def make_filesystem():
    return FileSystemState(
        directories={"/", "/reports", "/archive"},
        files={
            "/reports/a.csv": "original",
            "/reports/b.csv": "second file",
        },
        metadata={
            "/reports/a.csv": {"importance": "high"}
        },
        dependencies={
            "/reports/b.csv": ["/reports/a.csv"]
        },
    )


def test_move_file_updates_dependencies_and_metadata():
    fs = make_filesystem()
    original = fs.clone()

    moved, _ = transition(fs, {
        "type": "move_file",
        "source": "/reports/a.csv",
        "destination": "/archive/a.csv",
    })

    assert "/archive/a.csv" in moved.files
    assert moved.files["/archive/a.csv"] == "original"
    assert "/archive/a.csv" in moved.metadata

    assert moved.dependencies["/reports/b.csv"] == [
        "/archive/a.csv"
    ]

    # The original state must remain unchanged.
    assert fs.exactly_equals(original)


def test_rename_directory_updates_dependencies():
    fs = make_filesystem()
    original = fs.clone()

    renamed, _ = transition(fs, {
        "type": "rename",
        "source": "/reports",
        "new_name": "old_reports",
    })

    assert "/old_reports/a.csv" in renamed.files
    assert "/old_reports/b.csv" in renamed.files

    assert renamed.dependencies["/old_reports/b.csv"] == [
        "/old_reports/a.csv"
    ]

    # Metadata must follow the renamed file.
    assert renamed.metadata["/old_reports/a.csv"] == {
        "importance": "high"
    }

    # The original state must remain unchanged.
    assert fs.exactly_equals(original)


def test_recursive_delete_directory():
    fs = make_filesystem()
    original = fs.clone()

    deleted, _ = transition(fs, {
        "type": "delete_directory",
        "path": "/reports",
        "recursive": True,
    })

    assert "/reports/a.csv" not in deleted.files
    assert "/reports/b.csv" not in deleted.files
    assert "/reports" not in deleted.directories

    # The original state must remain unchanged.
    assert fs.exactly_equals(original)


def test_resulting_states_are_valid():
    fs = make_filesystem()

    moved, _ = transition(fs, {
        "type": "move_file",
        "source": "/reports/a.csv",
        "destination": "/archive/a.csv",
    })

    renamed, _ = transition(fs, {
        "type": "rename",
        "source": "/reports",
        "new_name": "old_reports",
    })

    deleted, _ = transition(fs, {
        "type": "delete_directory",
        "path": "/reports",
        "recursive": True,
    })

    assert moved.validate()
    assert renamed.validate()
    assert deleted.validate()
