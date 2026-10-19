
from __future__ import annotations

from collections import deque
from typing import Any

from state import WalletState
from recovery import get_recovery_actions, execute_recovery_action


RECOVERY_HORIZON = 3
EXTENDED_RECOVERY_HORIZON = 10



def recovery_goal_reached(
    current: WalletState,
    target: WalletState,
) -> bool:
    """
    Account existence and balances must match the target.
    The target's ledger must remain an unchanged prefix.
    New reversal records must reference transfers in the added history.
    """
    if current.accounts != target.accounts:
        return False

    target_history_length = len(target.transactions)

    if len(current.transactions) < target_history_length:
        return False

    if (
        current.transactions[:target_history_length]
        != target.transactions
    ):
        return False

    new_transactions = current.transactions[target_history_length:]

    # Collect transfers from the CURRENT added history, not the target.
    new_transfer_ids = {
        transaction["transaction_id"]
        for transaction in new_transactions
        if transaction.get("type") == "transfer"
    }

    for transaction in new_transactions:
        transaction_type = transaction.get("type")

        if transaction_type not in {
            "deposit",
            "withdraw",
            "transfer",
            "transfer_reversal",
        }:
            return False

        if transaction_type == "transfer_reversal":
            original_id = transaction.get("original_transaction_id")

            if original_id not in new_transfer_ids:
                return False

    return True

def _state_key(wallet: WalletState) -> tuple:
    """Canonical representation for BFS visited-state tracking."""
    return (
        tuple(sorted(wallet.accounts.items())),
        tuple(
            tuple(sorted(transaction.items()))
            for transaction in wallet.transactions
        ),
        tuple(sorted(wallet.reversed_transactions)),
        tuple(sorted(wallet.undoable_transactions)),
        wallet.next_transaction_id,
    )


def find_recovery_path(
    current: WalletState,
    target: WalletState,
    max_depth: int = RECOVERY_HORIZON,
) -> list[dict[str, Any]] | None:
    """Return a shortest valid recovery path within max_depth."""
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative.")

    start = current.clone()
    queue = deque([(start, [])])
    visited = {_state_key(start)}

    while queue:
        state, path = queue.popleft()

        if recovery_goal_reached(state, target):
            return path

        if len(path) >= max_depth:
            continue

        for action in get_recovery_actions(state):
            try:
                next_state, _ = execute_recovery_action(state, action)
            except (ValueError, KeyError, TypeError):
                continue

            key = _state_key(next_state)

            if key in visited:
                continue

            visited.add(key)
            queue.append((next_state, path + [action]))

    return None


def classify_reversibility(
    current: WalletState,
    target: WalletState,
    horizon: int = RECOVERY_HORIZON,
) -> dict[str, Any]:
    """
    Binary labels:
      reversible     - recovery found within the horizon
      irreversible   - no recovery found within the horizon

    Trinary labels:
      reversible          - recovery found within the horizon
      partially_reversible - recovery found beyond the horizon
      irreversible        - no recovery found within the extended bound

    IMPORTANT: 'unrecoverable' means no path was found within the
    configured extended bound; it is not a mathematical proof that
    no path exists at any depth.
    """
    if horizon < 0:
        raise ValueError("horizon must be nonnegative.")

    short_path = find_recovery_path(
        current,
        target,
        max_depth=horizon,
    )

    if short_path is not None:
        return {
            "binary_label": "reversible",
            "trinary_label": "reversible",
            "recovery_cost": len(short_path),
            "recovery_path": short_path,
        }

    extended_horizon = max(
        EXTENDED_RECOVERY_HORIZON,
        horizon,
    )

    extended_path = find_recovery_path(
        current,
        target,
        max_depth=extended_horizon,
    )

    if extended_path is not None:
        return {
            "binary_label": "irreversible",
            "trinary_label": "partially_reversible",
            "recovery_cost": len(extended_path),
            "recovery_path": extended_path,
        }

    return {
        "binary_label": "irreversible",
        "trinary_label": "irreversible",
        "recovery_cost": "unrecoverable",
        "recovery_path": None,
    }
