
from __future__ import annotations

from typing import Any

from state import WalletState
from transitions import transition
from bfs import classify_reversibility


ACCOUNT_UNDO_TYPES = {
    "undo_create_account",
    "undo_delete_account",
}


def run_experiment(
    scenario_id: str,
    initial_wallet: WalletState,
    action: dict[str, Any],
    record_undo: bool = True,
    recovery_horizon: int = 3,
    extended_horizon: int = 10,
) -> dict[str, Any]:
    """Execute one wallet action and classify recovery to the initial state."""
    original_wallet = initial_wallet.clone()

    resulting_wallet, transition_record = transition(
        original_wallet,
        action,
    )

    if not record_undo:
        original_transaction_ids = {
            transaction["transaction_id"]
            for transaction in original_wallet.transactions
        }

        newly_created_ids = {
            transaction["transaction_id"]
            for transaction in resulting_wallet.transactions
            if transaction["transaction_id"] not in original_transaction_ids
        }

        resulting_wallet.undoable_transactions.difference_update(
            newly_created_ids
        )

        action_undo = transition_record.get("undo")

        if (
            action_undo is not None
            and action_undo.get("type") in ACCOUNT_UNDO_TYPES
        ):
            resulting_wallet.pending_account_undos = [
                undo
                for undo in resulting_wallet.pending_account_undos
                if undo != action_undo
            ]

        resulting_wallet.validate()

    classification = classify_reversibility(
        current=resulting_wallet,
        target=original_wallet,
        horizon=recovery_horizon,
        extended_horizon=extended_horizon,
    )

    return {
        "scenario_id": scenario_id,
        "action": dict(action),
        "record_undo": record_undo,
        "initial_state": original_wallet.to_dict(),
        "resulting_state": resulting_wallet.to_dict(),
        "binary_label": classification["binary_label"],
        "trinary_label": classification["trinary_label"],
        "recovery_cost": classification["recovery_cost"],
        "recovery_path": classification["recovery_path"],
    }
