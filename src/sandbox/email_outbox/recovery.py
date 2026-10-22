
from __future__ import annotations

from copy import deepcopy
from typing import Any

from state import EmailOutboxState


def save_snapshot(
    state: EmailOutboxState,
    snapshot_id: str,
) -> EmailOutboxState:
    """Save a restorable snapshot of local state."""
    if not snapshot_id:
        raise ValueError("snapshot_id cannot be empty.")

    updated = state.clone()
    updated.snapshots[snapshot_id] = {
        "emails": deepcopy(
            {
                email_id: email.__dict__
                for email_id, email in updated.emails.items()
            }
        ),
        "undo_history": deepcopy(updated.undo_history),
    }
    return updated


def restore_snapshot(
    state: EmailOutboxState,
    snapshot_id: str,
) -> EmailOutboxState:
    """Restore local state without erasing external send history."""
    if snapshot_id not in state.snapshots:
        raise KeyError(f"Snapshot not found: {snapshot_id}")

    restored = state.clone()
    snapshot = deepcopy(state.snapshots[snapshot_id])

    from state import Email

    restored.emails = {
        email_id: Email(**email_data)
        for email_id, email_data in snapshot["emails"].items()
    }
    restored.undo_history = deepcopy(
        snapshot["undo_history"]
    )

    # Deliberately preserve restored.sent_history as-is.
    return restored


def undo_action(
    state: EmailOutboxState,
    undo_record: dict[str, Any],
) -> EmailOutboxState:
    """Apply an explicit undo record to local state.

    Undo records have the form:
      {"type": "restore_email", "email_id": ..., "email": {...}}
      {"type": "remove_email", "email_id": ...}

    External send history is never removed by these operations.
    """
    updated = state.clone()
    action_type = undo_record.get("type")
    email_id = undo_record.get("email_id")

    if not email_id:
        raise ValueError("Undo record requires email_id.")

    if action_type == "restore_email":
        email_data = undo_record.get("email")
        if not isinstance(email_data, dict):
            raise ValueError(
                "restore_email requires an email dictionary."
            )

        from state import Email

        updated.emails[email_id] = Email(**deepcopy(email_data))

    elif action_type == "remove_email":
        updated.emails.pop(email_id, None)

    else:
        raise ValueError(
            f"Unsupported undo operation: {action_type!r}"
        )

    return updated


def execute_with_undo(
    state: EmailOutboxState,
    action: dict[str, Any],
    undo_record: dict[str, Any],
) -> tuple[EmailOutboxState, dict[str, Any]]:
    """Execute a transition and retain its recovery record."""
    from transitions import transition

    updated, info = transition(state, action)

    if info["success"]:
        updated.undo_history.append(deepcopy(undo_record))

    return updated, info


def get_recovery_actions(
    state: EmailOutboxState,
    email_id: str,
) -> list[dict[str, Any]]:
    """Return undo records available for a particular email."""
    return [
        deepcopy(record)
        for record in reversed(state.undo_history)
        if record.get("email_id") == email_id
    ]
