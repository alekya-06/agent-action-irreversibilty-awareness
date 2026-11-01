
from __future__ import annotations

from typing import Any

from state import WalletState
from transitions import transition


def get_permissible_actions(
    wallet: WalletState,
) -> list[dict[str, Any]]:
    """
    Generate valid ordinary actions for the current wallet state.

    Recovery actions are deliberately excluded. They are handled by
    recovery.py and searched by bfs.py.
    """
    candidates: list[dict[str, Any]] = []

    # Read-only actions
    for account_id in sorted(wallet.accounts):
        candidates.append({
            "type": "check_balance",
            "account_id": account_id,
        })

    candidates.append({"type": "view_transactions"})

    # Account creation and deletion
    candidates.append({
        "type": "create_account",
        "account_id": _next_account_id(wallet),
        "initial_balance": 0,
    })

    for account_id in sorted(wallet.accounts):
        candidates.append({
            "type": "delete_account",
            "account_id": account_id,
        })

    # Deposits and withdrawals
    for account_id in sorted(wallet.accounts):
        candidates.append({
            "type": "deposit",
            "account_id": account_id,
            "amount": 100,
        })

        if wallet.accounts[account_id] >= 100:
            candidates.append({
                "type": "withdraw",
                "account_id": account_id,
                "amount": 100,
            })

    # Transfers between distinct accounts
    account_ids = sorted(wallet.accounts)

    for source in account_ids:
        for destination in account_ids:
            if source == destination:
                continue

            if wallet.accounts[source] >= 100:
                candidates.append({
                    "type": "transfer",
                    "source": source,
                    "destination": destination,
                    "amount": 100,
                })

    # Explicit reversal actions for eligible transfers
    for transaction in wallet.transactions:
        transaction_id = transaction["transaction_id"]

        if (
            transaction.get("type") == "transfer"
            and transaction_id in wallet.undoable_transactions
            and transaction_id not in wallet.reversed_transactions
        ):
            candidates.append({
                "type": "reverse_transfer",
                "transaction_id": transaction_id,
            })

    # Keep only candidates accepted by the transition layer.
    permissible: list[dict[str, Any]] = []

    for action in candidates:
        try:
            transition(wallet, action)
        except (ValueError, KeyError, TypeError):
            continue

        permissible.append(action)

    return permissible


def _next_account_id(wallet: WalletState) -> str:
    """Return a deterministic unused account ID."""
    index = 1

    while f"account_{index}" in wallet.accounts:
        index += 1

    return f"account_{index}"
