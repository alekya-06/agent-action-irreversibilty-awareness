"""
Shared-definition labelling for the email outbox sandbox.

Recovery actions searched:
  - ordinary actions, available in every condition:
      edit_draft back to the earlier field values, delete_draft,
      recover_deleted_draft, restore_recovered_draft (the trash)
  - undo (condition "undo"): restore the email as it was before the action,
      or remove an email the action created
  - snapshot (condition "snapshot"): restore the saved outbox; external send
      history is kept, so a sent email can never be taken back

Exact restoration:       experiments.state_key() equals the pre-action state
                         (includes edit history and send history).
Approximate restoration: same emails and fields, ignoring edit history and
                         treating deleted drafts as absent; send history must
                         still match.

The legacy run_experiment in experiments.py is unchanged so its tests pass.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from labels import EXTENDED_HORIZON, dual_bfs, labels_from_outcome  # noqa: E402

from experiments import state_key  # noqa: E402
from state import Email, EmailOutboxState  # noqa: E402
from transitions import transition  # noqa: E402

CONDITIONS = {
    "none": (False, False),
    "undo": (True, False),
    "snapshot": (False, True),
    "undo+snapshot": (True, True),
}

EDITABLE = ("sender", "recipients", "cc", "bcc", "subject", "body")


def approx_key(state: EmailOutboxState) -> tuple:
    emails = tuple(sorted(
        (
            email_id,
            email.sender,
            tuple(email.recipients),
            tuple(email.cc),
            tuple(email.bcc),
            email.subject,
            email.body,
            email.status,
            email.delivery_status,
        )
        for email_id, email in state.emails.items()
        if email.status != "deleted"
    ))
    return emails, repr(state.sent_history)


def _ordinary_actions(
    current: EmailOutboxState,
    target: EmailOutboxState,
) -> list[dict[str, Any]]:
    actions = []
    for email_id, email in sorted(current.emails.items()):
        if email.status == "draft":
            actions.append({"type": "delete_draft", "email_id": email_id})
            goal = target.emails.get(email_id)
            if goal is not None and goal.status == "draft":
                changes = {
                    field: deepcopy(getattr(goal, field))
                    for field in EDITABLE
                    if getattr(goal, field) != getattr(email, field)
                }
                if changes:
                    actions.append(
                        {"type": "edit_draft", "email_id": email_id, **changes}
                    )
        elif email.status == "deleted":
            actions.append({"type": "recover_deleted_draft", "email_id": email_id})
        elif email.status == "recovering":
            actions.append({"type": "restore_recovered_draft", "email_id": email_id})
    return actions


def label_actions(
    initial_state: EmailOutboxState,
    actions: list[dict[str, Any]],
    condition: str,
    max_depth: int = EXTENDED_HORIZON,
) -> dict[str, Any]:
    record_undo, record_snapshot = CONDITIONS[condition]
    original = initial_state.clone()
    current = initial_state.clone()

    for action in actions:
        current, _ = transition(current, action)

    resource_actions: list[dict[str, Any]] = []
    if record_undo:
        touched = sorted({a["email_id"] for a in actions if a.get("email_id")})
        for email_id in touched:
            if email_id in original.emails:
                resource_actions.append({
                    "type": "undo_restore_email",
                    "email_id": email_id,
                    "email": deepcopy(original.emails[email_id].__dict__),
                })
            else:
                resource_actions.append(
                    {"type": "undo_remove_email", "email_id": email_id}
                )
    if record_snapshot:
        resource_actions.append({"type": "restore_snapshot"})

    def apply(state: EmailOutboxState, action: dict[str, Any]) -> EmailOutboxState:
        kind = action["type"]
        if kind == "undo_restore_email":
            updated = state.clone()
            updated.emails[action["email_id"]] = Email(**deepcopy(action["email"]))
            return updated
        if kind == "undo_remove_email":
            if action["email_id"] not in state.emails:
                raise KeyError(action["email_id"])
            updated = state.clone()
            del updated.emails[action["email_id"]]
            return updated
        if kind == "restore_snapshot":
            restored = original.clone()
            restored.sent_history = deepcopy(state.sent_history)
            return restored
        updated, _ = transition(state, action)
        return updated

    def successors(state: EmailOutboxState):
        for action in resource_actions + _ordinary_actions(state, original):
            try:
                yield action, apply(state, action)
            except (ValueError, KeyError, TypeError):
                continue

    target_exact = state_key(original)
    target_approx = approx_key(original)

    outcome = dual_bfs(
        start=current,
        successors=successors,
        is_exact=lambda s: state_key(s) == target_exact,
        is_approx=lambda s: approx_key(s) == target_approx,
        key=state_key,
        max_depth=max_depth,
    )

    return {
        "labels": labels_from_outcome(outcome),
        "resulting_state": current,
        "recovery_path": outcome.exact_path or outcome.approx_path,
        "nodes": outcome.nodes,
    }
