from copy import deepcopy

from state import FileSystemState
from transitions import transition
from recovery import save_snapshot, execute_with_undo
from bfs import assess_reversibility


def serialize_state(fs: FileSystemState) -> dict:
    """Convert a filesystem state into a JSON-friendly dictionary."""
    return {
        "directories": sorted(fs.directories),
        "files": dict(sorted(fs.files.items())),
        "metadata": deepcopy(
            dict(sorted(fs.metadata.items()))
        ),
        "dependencies": deepcopy(
            dict(sorted(fs.dependencies.items()))
        ),
    }


def run_experiment(
    scenario_id: str,
    initial_fs: FileSystemState,
    action: dict,
    *,
    provide_snapshot: bool = False,
    record_undo: bool = False,
    max_steps: int = 3,
) -> dict:
    """
    Execute one controlled filesystem experiment.

    The initial filesystem is never mutated.

    A scenario may provide a snapshot, action-specific undo,
    both, or neither.
    """
    if not scenario_id:
        raise ValueError("scenario_id must not be empty")

    if max_steps < 0:
        raise ValueError("max_steps cannot be negative")

    initial_fs.validate()

    # Preserve the reference state for ground-truth comparison.
    original_fs = initial_fs.clone()
    current_fs = initial_fs.clone()

    snapshots = {}
    undo_records = {}

    # Make recovery resources available BEFORE the action.
    if provide_snapshot:
        snapshots = save_snapshot(
            current_fs,
            snapshots,
            f"{scenario_id}_snapshot",
        )

    # Execute the action, optionally recording its inverse.
    if record_undo:
        current_fs, undo_records, transition_record = (
            execute_with_undo(
                current_fs,
                action,
                f"{scenario_id}_action",
                undo_records,
            )
        )
    else:
        current_fs, transition_record = transition(
            current_fs, action
        )

    # Determine recoverability using only available resources.
    assessment = assess_reversibility(
        original_fs=original_fs,
        current_fs=current_fs,
        snapshots=snapshots,
        undo_records=undo_records,
        max_steps=max_steps,
    )

    if assessment["reversible_within_horizon"]:
        label = "recoverable_within_horizon"
    else:
        label = "not_recoverable_within_horizon"

    return {
        "scenario_id": scenario_id,
        "environment": "filesystem",
        "initial_state": serialize_state(original_fs),
        "action": deepcopy(action),
        "resulting_state": serialize_state(current_fs),
        "transition_record": transition_record,
        "recovery_resources": {
            "snapshot_available": provide_snapshot,
            "undo_available": record_undo,
        },
        "max_recovery_steps": max_steps,
        "reversibility_label": label,
        "recovery_path": assessment["recovery_path"],
        "recovery_cost": assessment["recovery_cost"],
    }
