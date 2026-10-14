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
