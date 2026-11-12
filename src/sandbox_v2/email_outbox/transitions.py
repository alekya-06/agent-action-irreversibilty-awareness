
from __future__ import annotations

from copy import deepcopy
from typing import Any

from actions import validate_action
from state import Email, EmailOutboxState


def transition(
    state: EmailOutboxState,
    action: dict[str, Any],
) -> tuple[EmailOutboxState, dict[str, Any]]:
    """Apply one action and return (new_state, transition_info).

    The input state is not mutated. External send history is
    preserved as part of the returned state.
    """
    validate_action(action)

    new_state = state.clone()
    action_type = action["type"]
    email_id = action.get("email_id")

    info: dict[str, Any] = {
        "action": action_type,
        "email_id": email_id,
        "success": False,
        "external_effect": False,
    }

    if action_type == "create_draft":
        email = Email(
            email_id=email_id,
            sender=action["sender"],
            recipients=list(action["recipients"]),
            subject=action["subject"],
            body=action["body"],
            cc=list(action.get("cc", [])),
            bcc=list(action.get("bcc", [])),
        )
        new_state.add_email(email)
        info["success"] = True
        return new_state, info

    if email_id not in new_state.emails:
        raise KeyError(f"Email not found: {email_id}")

    email = new_state.get_email(email_id)

    if action_type == "edit_draft":
        if email.status != "draft":
            raise ValueError("Only drafts can be edited.")

        editable = {
            "sender", "recipients", "cc", "bcc", "subject", "body"
        }
        updates = {
            key: deepcopy(value)
            for key, value in action.items()
            if key in editable
        }

        email.previous_versions.append({
            key: deepcopy(getattr(email, key))
            for key in updates
        })

        for key, value in updates.items():
            setattr(email, key, value)

    elif action_type == "delete_draft":
        if email.status != "draft":
            raise ValueError("Only drafts can be deleted.")
        email.status = "deleted"

    elif action_type == "queue_email":
        if email.status != "draft":
            raise ValueError("Only drafts can be queued.")
        email.status = "queued"

    elif action_type == "send_email":
        if email.status != "queued":
            raise ValueError("Only queued emails can be sent.")

        delivery_status = action.get("delivery_status", "delivered")
        if delivery_status not in {"delivered", "failed", "unknown"}:
            raise ValueError(
                f"Invalid delivery status: {delivery_status}"
            )

        email.status = "sent"
        email.delivery_status = delivery_status

        # Record the external attempt independently of local state.
        new_state.sent_history.append({
            "email_id": email.email_id,
            "sender": email.sender,
            "recipients": deepcopy(email.recipients),
            "cc": deepcopy(email.cc),
            "bcc": deepcopy(email.bcc),
            "subject": email.subject,
            "body": email.body,
            "delivery_status": delivery_status,
        })

        info["external_effect"] = True

        if new_state.undo_send_window > 0:
            new_state.recallable[email.email_id] = new_state.undo_send_window

    elif action_type == "recall_email":
        if email.email_id not in new_state.recallable:
            raise ValueError("This email cannot be recalled (no undo-send window, or it has passed).")
        for i in range(len(new_state.sent_history) - 1, -1, -1):
            if new_state.sent_history[i]["email_id"] == email.email_id:
                del new_state.sent_history[i]
                break
        email.status = "queued"
        email.delivery_status = "pending"
        del new_state.recallable[email.email_id]

    elif action_type == "mark_delivered":
        if email.status != "sent":
            raise ValueError("Only sent emails can be marked delivered.")
        email.delivery_status = "delivered"
        _update_latest_delivery_status(
            new_state, email_id, "delivered"
        )

    elif action_type == "mark_failed":
        if email.status != "sent":
            raise ValueError("Only sent emails can be marked failed.")
        email.delivery_status = "failed"
        _update_latest_delivery_status(
            new_state, email_id, "failed"
        )

    elif action_type == "recover_deleted_draft":
        if email.status != "deleted":
            raise ValueError("Only deleted emails can be recovered.")
    
        email.status = "recovering"
    
    elif action_type == "restore_recovered_draft":
        if email.status != "recovering":
            raise ValueError("Email must be in recovery first.")
        email.status = "draft"

    else:
        raise ValueError(f"Unhandled action: {action_type}")

    # Every action except the send itself uses up one step of open recall windows.
    for eid in list(new_state.recallable):
        if action_type == "send_email" and eid == email_id:
            continue
        new_state.recallable[eid] -= 1
        if new_state.recallable[eid] <= 0:
            del new_state.recallable[eid]

    info["success"] = True
    return new_state, info


def _update_latest_delivery_status(
    state: EmailOutboxState,
    email_id: str,
    delivery_status: str,
) -> None:
    """Update the latest matching external send record."""
    for record in reversed(state.sent_history):
        if record["email_id"] == email_id:
            record["delivery_status"] = delivery_status
            return

    raise ValueError(
        f"No send history exists for email: {email_id}"
    )
