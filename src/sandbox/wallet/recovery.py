
from __future__ import annotations

from typing import Any

from state import WalletState
from transitions import apply_undo


def get_recovery_actions(
    wallet: WalletState,
) -> list[dict[str, Any]]:
    """
    Return action-specific undo operations currently available.

    This function only proposes candidates. Each candidate is validated
    by apply_undo() against the current wallet state.
    """
    candidates: list[dict[str, Any]] = []

    # Transactions can be undone only while their undo permission remains.
    for transaction in wallet.transactions:
        transaction_id = transaction["transaction_id"]

        if transaction_id not in wallet.undoable_transactions:
            continue

        transaction_type = transaction.get("type")

        if transaction_type == "deposit":
            candidates.append({
                "type": "undo_deposit",
                "transaction_id": transaction_id,
                "account_id": transaction["account_id"],
                "amount": transaction["amount"],
            })

        elif transaction_type == "withdraw":
            candidates.append({
                "type": "undo_withdraw",
                "transaction_id": transaction_id,
                "account_id": transaction["account_id"],
                "amount": transaction["amount"],
            })

        elif transaction_type == "transfer":
            candidates.append({
                "type": "undo_transfer",
                "transaction_id": transaction_id,
                "source": transaction["source"],
                "destination": transaction["destination"],
                "amount": transaction["amount"],
            })

    # Validate against the current state, not the original state.
    available: list[dict[str, Any]] = []

    for undo in candidates:
        try:
            apply_undo(wallet, undo)
        except (ValueError, KeyError, TypeError):
            continue

        available.append(undo)

    return available


def execute_recovery_action(
    wallet: WalletState,
    undo: dict[str, Any],
) -> tuple[WalletState, dict[str, Any]]:
    """Execute one validated action-specific undo."""
    return apply_undo(wallet, undo)
