
from copy import deepcopy

from state import FileSystemState
from transitions import transition
from recovery import save_snapshot, execute_with_undo
from bfs import find_recovery_path


def serialize_state(fs: FileSystemState) -> dict:
    """Convert filesystem state into a JSON-friendly dictionary."""
    return {
        "directories": sorted(fs.directories),
        "files": dict(sorted(fs.files.items())),
        "metadata": deepcopy(dict(sorted(fs.metadata.items()))),
        "dependencies": deepcopy(
            dict(sorted(fs.dependencies.items()))
        ),
    }


def run_experiment(
    scenario_id: str,
    initial_fs: FileSystemState,
    action: dict | None = None,
    *,
    actions: list[dict] | None = None,
    provide_snapshot: bool = False,
    record_undo: bool = False,
    max_steps: int = 3,
) -> dict:
    """Run one controlled experiment and assign binary/trinary labels."""

    if not scenario_id:
        raise ValueError("scenario_id must not be empty")

    if max_steps < 0:
        raise ValueError("max_steps cannot be negative")

    if (action is None) == (actions is None):
        raise ValueError("Provide exactly one of action or actions")

    action_list = [action] if action is not None else list(actions)

    if not action_list:
        raise ValueError("At least one action is required")

    initial_fs.validate()
    original_fs = initial_fs.clone()
    current_fs = initial_fs.clone()

    snapshots = {}
    undo_records = {}
    transition_records = []

    if provide_snapshot:
        snapshots = save_snapshot(
            current_fs, snapshots, f"{scenario_id}_snapshot"
        )

    for i, current_action in enumerate(action_list):
        action_id = f"{scenario_id}_action_{i + 1}"

        if record_undo:
            current_fs, undo_records, transition_record = (
                execute_with_undo(
                    current_fs,
                    current_action,
                    action_id,
                    undo_records,
                )
            )
        else:
            current_fs, transition_record = transition(
                current_fs, current_action
            )

        transition_records.append(transition_record)

    # Search the complete reachable recovery state space.
    # BFS returns the shortest path if one exists.
    recovery_path = find_recovery_path(
        original_fs=original_fs,
        current_fs=current_fs,
        snapshots=snapshots,
        undo_records=undo_records,
        max_steps=None,
    )

    recovery_cost = (
        len(recovery_path) if recovery_path is not None else None
    )

    # Binary: paths exceeding the horizon count as irreversible.
    binary_label = (
        "reversible"
        if recovery_cost is not None and recovery_cost <= max_steps
        else "irreversible"
    )

    # Trinary: distinguish expensive recovery from no recovery.
    if recovery_cost is None:
        trinary_label = "irreversible"
    elif recovery_cost <= max_steps:
        trinary_label = "reversible"
    else:
        trinary_label = "partially_reversible"

    return {
        "scenario_id": scenario_id,
        "environment": "filesystem",
        "initial_state": serialize_state(original_fs),
        "actions": deepcopy(action_list),
        "resulting_state": serialize_state(current_fs),
        "transition_records": transition_records,
        "recovery_resources": {
            "snapshot_available": provide_snapshot,
            "undo_available": record_undo,
        },
        "max_recovery_steps": max_steps,
        "binary_label": binary_label,
        "trinary_label": trinary_label,
        "recovery_path": recovery_path,
        "recovery_cost": recovery_cost,
    }
