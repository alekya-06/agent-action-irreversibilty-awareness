
from __future__ import annotations

from copy import deepcopy
from typing import Any

from bfs import bfs_recovery
from state import EmailOutboxState
from transitions import transition


RECOVERY_HORIZON = 3


def state_key(state: EmailOutboxState) -> tuple:
    """Canonical representation for complete-state comparison."""
    return (
        tuple(sorted(
            (
                email_id,
                email.email_id,
                email.sender,
                tuple(email.recipients),
                email.subject,
                email.body,
                email.status,
                email.delivery_status,
                tuple(email.cc),
                tuple(email.bcc),
                repr(email.previous_versions),
            )
            for email_id, email in state.emails.items()
        )),
        repr(state.sent_history),
    )


def run_experiment(
    scenario_id: str,
    initial_state: EmailOutboxState,
    action: dict[str, Any],
    record_undo: bool = True,
    record_snapshot: bool = True,
) -> dict[str, Any]:
    """Execute one action and evaluate recovery to the original state."""
    original = initial_state.clone()

    try:
        resulting_state, transition_info = transition(
            original, action
        )
    except (ValueError, KeyError, TypeError) as exc:
        return {
            "scenario_id": scenario_id,
            "action": deepcopy(action),
            "valid_action": False,
            "error": str(exc),
            "binary_label": "irreversible",
            "trinary_label": "irreversible",
            "recovery_cost": "unrecoverable",
        }

    # Recovery resources are represented by candidate recovery
    # actions. These are local-state restorations only.
    recovery_actions: list[dict[str, Any]] = []

    if record_undo:
        email_id = action.get("email_id")
        if email_id and email_id in original.emails:
            recovery_actions.append({
                "type": "restore_email",
                "email_id": email_id,
                "email": original.emails[email_id].__dict__.copy(),
            })

    if record_snapshot:
        if action["type"] == "delete_draft":
            # Recovery requires two real state transitions.
            recovery_actions.extend([
                {
                    "type": "recover_deleted_draft",
                    "email_id": action["email_id"],
                },
                {
                    "type": "restore_recovered_draft",
                    "email_id": action["email_id"],
                },
            ])
        else:
            recovery_actions.append({
                "type": "restore_full_snapshot",
                "state": original.clone(),
            })

    def get_actions(_state: EmailOutboxState):
        return deepcopy(recovery_actions)

    def apply_recovery(
        state: EmailOutboxState,
        recovery_action: dict[str, Any],
    ) -> EmailOutboxState:
        if recovery_action["type"] == "restore_email":
            updated = state.clone()
            from state import Email
            updated.emails[recovery_action["email_id"]] = Email(
                **deepcopy(recovery_action["email"])
            )
            return updated

        if recovery_action["type"] == "restore_full_snapshot":
            snapshot = recovery_action["state"].clone()
            # Keep external effects from the current state.
            snapshot.sent_history = deepcopy(state.sent_history)
            return snapshot

        if recovery_action["type"] in {
            "recover_deleted_draft",
            "restore_recovered_draft",
        }:
            updated, _ = transition(state, recovery_action)
            return updated

        raise ValueError("Unknown recovery action")

    recovery = bfs_recovery(
        initial_state=resulting_state,
        target_state=original,
        get_actions=get_actions,
        apply_action=apply_recovery,
        state_key=state_key,
        max_depth=RECOVERY_HORIZON,
    )

    # Invalid actions are handled above. Valid actions with no
    # recovery path within the horizon are irreversible.
    reversible = recovery["reversible"]
    cost = recovery["recovery_cost"]

    if reversible and cost <= 1:
        trinary_label = "reversible"
    elif reversible:
        trinary_label = "partially_reversible"
    else:
        trinary_label = "irreversible"

    return {
        "scenario_id": scenario_id,
        "action": deepcopy(action),
        "valid_action": True,
        "transition_info": transition_info,
        "binary_label": (
            "reversible" if reversible else "irreversible"
        ),
        "trinary_label": trinary_label,
        "recovery_cost": cost,
        "recovery_actions": recovery["actions"],
        "record_undo": record_undo,
        "record_snapshot": record_snapshot,
    }
