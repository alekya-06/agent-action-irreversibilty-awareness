
from __future__ import annotations

from typing import Any, Literal, TypedDict


ActionType = Literal[
    "create_draft",
    "edit_draft",
    "delete_draft",
    "queue_email",
    "send_email",
    "mark_delivered",
    "mark_failed",
    "recover_deleted_draft",
    "restore_recovered_draft",
]


class EmailAction(TypedDict, total=False):
    """A requested action in the email outbox sandbox."""

    type: ActionType
    email_id: str

    # Used when creating or editing a draft.
    sender: str
    recipients: list[str]
    cc: list[str]
    bcc: list[str]
    subject: str
    body: str

    # Optional simulated delivery outcome.
    delivery_status: Literal["delivered", "failed", "unknown"]


VALID_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "create_draft",
        "edit_draft",
        "delete_draft",
        "queue_email",
        "send_email",
        "recall_email",
        "mark_delivered",
        "mark_failed",
        "recover_deleted_draft",
        "restore_recovered_draft",
    }
)


def validate_action(action: dict[str, Any]) -> None:
    """Validate an action's structure before applying it."""
    if not isinstance(action, dict):
        raise TypeError("Action must be a dictionary.")

    action_type = action.get("type")
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(
            f"Unsupported action type: {action_type!r}"
        )

    if action_type != "create_draft" and not action.get("email_id"):
        raise ValueError(
            f"Action {action_type!r} requires an email_id."
        )

    if action_type == "create_draft":
        required = ("email_id", "sender", "recipients", "subject", "body")
        missing = [
            key for key in required
            if key not in action
        ]
        if missing:
            raise ValueError(
                f"create_draft missing fields: {missing}"
            )

    if action_type == "edit_draft":
        editable = {"sender", "recipients", "cc", "bcc", "subject", "body"}
        if not editable.intersection(action):
            raise ValueError(
                "edit_draft requires at least one field to update."
            )

    for key in ("recipients", "cc", "bcc"):
        if key in action:
            if not isinstance(action[key], list):
                raise TypeError(f"{key} must be a list.")
            if not all(isinstance(value, str) for value in action[key]):
                raise TypeError(f"Every {key} entry must be a string.")


def create_draft_action(
    email_id: str,
    sender: str,
    recipients: list[str],
    subject: str,
    body: str,
    *,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
) -> EmailAction:
    """Build an action that creates a draft."""
    return {
        "type": "create_draft",
        "email_id": email_id,
        "sender": sender,
        "recipients": list(recipients),
        "cc": list(cc or []),
        "bcc": list(bcc or []),
        "subject": subject,
        "body": body,
    }


def edit_draft_action(
    email_id: str,
    **updates: Any,
) -> EmailAction:
    """Build an action that edits selected draft fields."""
    action: EmailAction = {
        "type": "edit_draft",
        "email_id": email_id,
    }
    action.update(updates)
    validate_action(action)
    return action


def simple_action(
    action_type: ActionType,
    email_id: str,
) -> EmailAction:
    """Build a simple action that targets an existing email."""
    action: EmailAction = {
        "type": action_type,
        "email_id": email_id,
    }
    validate_action(action)
    return action
