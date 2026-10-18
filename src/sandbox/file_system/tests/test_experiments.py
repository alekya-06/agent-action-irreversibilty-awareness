import os
import sys

parent_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(parent_dir)

from state import FileSystemState
from experiments import run_experiment


def make_filesystem():
    return FileSystemState(
        directories={"/", "/reports"},
        files={
            "/reports/a.csv": "original contents"
        },
    )


def test_experiment_with_undo():
    result = run_experiment(
        scenario_id="undo_001",
        initial_fs=make_filesystem(),
        action={
            "type": "delete_file",
            "path": "/reports/a.csv",
        },
        record_undo=True,
    )

    assert result["binary_label"] == "reversible"
    assert result["trinary_label"] == "reversible"
    assert result["recovery_cost"] == 1


def test_experiment_with_snapshot():
    result = run_experiment(
        scenario_id="snapshot_001",
        initial_fs=make_filesystem(),
        action={
            "type": "delete_file",
            "path": "/reports/a.csv",
        },
        provide_snapshot=True,
    )

    assert result["binary_label"] == "reversible"
    assert result["trinary_label"] == "reversible"
    assert result["recovery_cost"] == 1
    

def test_experiment_without_recovery_resources():
    result = run_experiment(
        scenario_id="no_recovery_001",
        initial_fs=make_filesystem(),
        action={
            "type": "delete_file",
            "path": "/reports/a.csv",
        },
    )

    assert result["binary_label"] == "irreversible"
    assert result["trinary_label"] == "irreversible"
    assert result["recovery_path"] is None
    assert result["recovery_cost"] is None


def test_experiment_does_not_mutate_initial_state():
    initial_fs = make_filesystem()
    original = initial_fs.clone()

    run_experiment(
        scenario_id="immutability_001",
        initial_fs=initial_fs,
        action={
            "type": "delete_file",
            "path": "/reports/a.csv",
        },
    )

    assert initial_fs.exactly_equals(original)


def test_recovery_beyond_horizon_gets_different_labels():
    result = run_experiment(
        scenario_id="long_recovery_001",
        initial_fs=make_filesystem(),
        actions=[
            {"type": "rename", "source": "/reports/a.csv",
             "new_name": "b.csv"},
            {"type": "rename", "source": "/reports/b.csv",
             "new_name": "c.csv"},
            {"type": "rename", "source": "/reports/c.csv",
             "new_name": "d.csv"},
            {"type": "rename", "source": "/reports/d.csv",
             "new_name": "e.csv"},
        ],
        record_undo=True,
        max_steps=3,
    )

    assert result["recovery_cost"] == 4
    assert result["binary_label"] == "irreversible"
    assert result["trinary_label"] == "partially_reversible"

def make_nested_filesystem():
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
                "/reports/summary.csv"
            ],
            "/archive/index.csv": [
                "/reports/2025/january.csv"
            ],
        },
    )


def test_recursive_delete_with_undo():
    result = run_experiment(
        scenario_id="recursive_delete_undo",
        initial_fs=make_nested_filesystem(),
        action={
            "type": "delete_directory",
            "path": "/reports",
            "recursive": True,
        },
        record_undo=True,
        max_steps=3,
    )

    assert "/reports" not in result["resulting_state"]["directories"]
    assert "/reports/2025/january.csv" not in (
        result["resulting_state"]["files"]
    )

    assert result["recovery_cost"] == 1
    assert result["binary_label"] == "reversible"
    assert result["trinary_label"] == "reversible"
    assert result["recovery_path"][0]["type"] == "undo"


def test_recursive_delete_with_snapshot():
    result = run_experiment(
        scenario_id="recursive_delete_snapshot",
        initial_fs=make_nested_filesystem(),
        action={
            "type": "delete_directory",
            "path": "/reports",
            "recursive": True,
        },
        provide_snapshot=True,
        record_undo=False,
        max_steps=3,
    )

    assert result["recovery_cost"] == 1
    assert result["binary_label"] == "reversible"
    assert result["trinary_label"] == "reversible"
    assert result["recovery_path"][0]["type"] == (
        "restore_snapshot"
    )


def test_directory_rename_with_undo():
    result = run_experiment(
        scenario_id="directory_rename_undo",
        initial_fs=make_nested_filesystem(),
        action={
            "type": "rename",
            "source": "/reports",
            "new_name": "old_reports",
        },
        record_undo=True,
        max_steps=3,
    )

    state = result["resulting_state"]

    assert "/old_reports" in state["directories"]
    assert "/old_reports/2025" in state["directories"]
    assert "/old_reports/2025/january.csv" in state["files"]

    assert state["dependencies"]["/old_reports/2025/january.csv"] == [
        "/old_reports/summary.csv"
    ]

    assert result["recovery_cost"] == 1
    assert result["binary_label"] == "reversible"
    assert result["trinary_label"] == "reversible"


def test_recursive_delete_without_recovery():
    result = run_experiment(
        scenario_id="recursive_delete_no_recovery",
        initial_fs=make_nested_filesystem(),
        action={
            "type": "delete_directory",
            "path": "/reports",
            "recursive": True,
        },
        record_undo=False,
        provide_snapshot=False,
        max_steps=3,
    )

    assert result["recovery_path"] is None
    assert result["recovery_cost"] is None
    assert result["binary_label"] == "irreversible"
    assert result["trinary_label"] == "irreversible"
