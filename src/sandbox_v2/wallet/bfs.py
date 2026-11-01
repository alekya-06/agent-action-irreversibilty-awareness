
from __future__ import annotations

from collections import deque
from typing import Any

from state import WalletState
from recovery import get_recovery_actions, execute_recovery_action


def recovery_goal_reached(
    current: WalletState,
    target: WalletState,
) -> bool:
    """Return whether current restores the target wallet's relevant state."""
    if current.accounts != target.accounts:
        return False

    target_history = target.transactions
    current_history = current.transactions

    if len(current_history) < len(target_history):
        return False

    # Existing history must never be deleted or rewritten.
    if current_history[:len(target_history)] != target_history:
        return False

    added_transactions = current_history[len(target_history):]
    known_transfer_ids = {
        transaction["transaction_id"]
        for transaction in current_history
        if transaction.get("type") == "transfer"
    }

    for transaction in added_transactions:
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

            if original_id not in known_transfer_ids:
                return False

    return True


def _state_key(wallet: WalletState) -> tuple:
    """Build a hashable key containing all recovery-relevant state."""
    return (
        tuple(sorted(wallet.accounts.items())),
        tuple(
            tuple(sorted(transaction.items()))
            for transaction in wallet.transactions
        ),
        tuple(sorted(wallet.reversed_transactions)),
        tuple(sorted(wallet.undoable_transactions)),
        tuple(
            tuple(sorted(undo.items()))
            for undo in wallet.pending_account_undos
        ),
        wallet.next_transaction_id,
    )


def find_recovery_path(
    start: WalletState,
    target: WalletState,
    max_depth: int | None = None,
    *,
    max_steps: int | None = None,
) -> list[dict[str, Any]] | None:
    """Find a shortest recovery path using BFS.

    Supports max_depth for compatibility with the existing tests.
    max_steps is also accepted as an alias.
    """
    if max_depth is not None and max_steps is not None:
        raise ValueError("Specify either max_depth or max_steps, not both.")

    depth_limit = max_depth if max_depth is not None else max_steps

    if depth_limit is None:
        raise TypeError("A maximum search depth must be specified.")

    if depth_limit < 0:
        raise ValueError("Maximum search depth cannot be negative.")

    if recovery_goal_reached(start, target):
        return []

    queue = deque([(start.clone(), [])])
    visited = {_state_key(start)}

    while queue:
        current, path = queue.popleft()

        if len(path) >= depth_limit:
            continue

        for undo in get_recovery_actions(current):
            try:
                next_state, _ = execute_recovery_action(current, undo)
            except (ValueError, KeyError, TypeError):
                continue

            next_path = path + [dict(undo)]

            if recovery_goal_reached(next_state, target):
                return next_path

            key = _state_key(next_state)

            if key in visited:
                continue

            visited.add(key)
            queue.append((next_state, next_path))

    return None


def classify_reversibility(
    current: WalletState,
    target: WalletState,
    horizon: int = 3,
    extended_horizon: int = 10,
) -> dict[str, Any]:
    """
    Binary labels:
      reversible: recovery path exists within horizon
      irreversible: no path found within horizon

    Trinary labels:
      reversible: recovery path exists within horizon
      partially_reversible: no short path, but an extended path is found
      irreversible: no path found within extended_horizon

    The extended search is bounded; failure to find a path is not a
    mathematical proof that no path exists.
    """
    if horizon < 0:
        raise ValueError("horizon cannot be negative.")

    if extended_horizon < horizon:
        raise ValueError("extended_horizon must be >= horizon.")

    short_path = find_recovery_path(
        start=current,
        target=target,
        max_steps=horizon,
    )

    if short_path is not None:
        return {
            "binary_label": "reversible",
            "trinary_label": "reversible",
            "recovery_cost": len(short_path),
            "recovery_path": short_path,
        }

    long_path = find_recovery_path(
        start=current,
        target=target,
        max_steps=extended_horizon,
    )

    if long_path is not None:
        return {
            "binary_label": "irreversible",
            "trinary_label": "partially_reversible",
            "recovery_cost": len(long_path),
            "recovery_path": long_path,
        }

    return {
        "binary_label": "irreversible",
        "trinary_label": "irreversible",
        "recovery_cost": "unrecoverable",
        "recovery_path": None,
    }
