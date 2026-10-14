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

    assert result["reversibility_label"] == (
        "recoverable_within_horizon"
    )
    assert result["recovery_cost"] == 1
    assert result["recovery_path"][0]["type"] == "undo"


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

    assert result["reversibility_label"] == (
        "recoverable_within_horizon"
    )
    assert result["recovery_cost"] == 1
    assert (
        result["recovery_path"][0]["type"]
        == "restore_snapshot"
    )


def test_experiment_without_recovery_resources():
    result = run_experiment(
        scenario_id="no_recovery_001",
        initial_fs=make_filesystem(),
        action={
            "type": "delete_file",
            "path": "/reports/a.csv",
        },
    )

    assert result["reversibility_label"] == (
        "not_recoverable_within_horizon"
    )
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
