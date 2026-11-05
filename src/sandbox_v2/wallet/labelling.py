"""
Shared-definition labelling for the wallet sandbox.

Recovery actions searched:
  - undo (condition "undo"): the action-specific undos from recovery.py,
    including cancelling a payment to a payee
  - ordinary compensating actions, available in every condition, on the
    user's own accounts only: deposit or withdraw the difference, transfer
    between own accounts, re-create or delete an empty own account.
    Payee (external) accounts can never be debited by an ordinary action.

The ledger is append-only, so restoration is judged on balances and on
what the ledger shows:
  Exact restoration:       same accounts and balances, and recovery added
                           no new user transaction (only system reversal
                           entries from an undo).
  Approximate restoration: same accounts and balances, earlier history
                           intact, but recovery added compensating
                           transactions (bfs.recovery_goal_reached).

The legacy run_experiment and classify_reversibility are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from labels import EXTENDED_HORIZON, dual_bfs, labels_from_outcome  # noqa: E402

from bfs import recovery_goal_reached  # noqa: E402
from recovery import execute_recovery_action, get_recovery_actions  # noqa: E402
from state import WalletState  # noqa: E402
from transitions import ACCOUNT_UNDO_TYPES, transition  # noqa: E402

CONDITIONS = {"none": False, "undo": True}

SYSTEM_ENTRY_TYPES = {"transfer_reversal"}


def run_actions(
    initial: WalletState,
    actions: list[dict[str, Any]],
    record_undo: bool,
) -> WalletState:
    """Apply actions; without undo, strip the undo eligibility they create."""
    current = initial.clone()
    for action in actions:
        before_ids = {tx["transaction_id"] for tx in current.transactions}
        current, record = transition(current, action)
        if not record_undo:
            new_ids = {
                tx["transaction_id"] for tx in current.transactions
            } - before_ids
            current.undoable_transactions.difference_update(new_ids)
            undo = record.get("undo")
            if undo is not None and undo.get("type") in ACCOUNT_UNDO_TYPES:
                current.pending_account_undos = [
                    u for u in current.pending_account_undos if u != undo
                ]
    current.validate()
    return current


def _compensating_actions(
    current: WalletState,
    target: WalletState,
) -> list[dict[str, Any]]:
    external = current.external_accounts
    own_now = {a for a in current.accounts if a not in external}
    own_goal = {a for a in target.accounts if a not in external}
    actions: list[dict[str, Any]] = []

    for account_id in sorted(own_goal - own_now):
        actions.append({"type": "create_account", "account_id": account_id})
    for account_id in sorted(own_now - own_goal):
        if current.accounts[account_id] == 0:
            actions.append({"type": "delete_account", "account_id": account_id})

    diffs = {
        a: current.accounts[a] - target.accounts[a]
        for a in sorted(own_now & own_goal)
        if current.accounts[a] != target.accounts[a]
    }
    for account_id, diff in diffs.items():
        if diff > 0:
            actions.append({"type": "withdraw", "account_id": account_id, "amount": diff})
        else:
            actions.append({"type": "deposit", "account_id": account_id, "amount": -diff})

    surplus = [a for a, d in diffs.items() if d > 0]
    deficit = [a for a, d in diffs.items() if d < 0]
    for source in surplus:
        for destination in deficit:
            actions.append({
                "type": "transfer",
                "source": source,
                "destination": destination,
                "amount": min(diffs[source], -diffs[destination]),
            })
    return actions


def label_actions(
    initial: WalletState,
    actions: list[dict[str, Any]],
    condition: str,
    max_depth: int = EXTENDED_HORIZON,
) -> dict[str, Any]:
    record_undo = CONDITIONS[condition]
    target = initial.clone()
    current = run_actions(initial, actions, record_undo)
    post_length = len(current.transactions)

    def accounts_match(state: WalletState) -> bool:
        return (
            state.accounts == target.accounts
            and state.external_accounts == target.external_accounts
        )

    def is_exact(state: WalletState) -> bool:
        if not accounts_match(state):
            return False
        n = len(target.transactions)
        if state.transactions[:n] != target.transactions:
            return False
        return all(
            tx.get("type") in SYSTEM_ENTRY_TYPES
            for tx in state.transactions[post_length:]
        )

    def is_approx(state: WalletState) -> bool:
        return accounts_match(state) and recovery_goal_reached(state, target)

    def search_key(state: WalletState) -> tuple:
        """
        State identity for the search. The full ledger would make every
        node unique (each compensating action appends an entry), so only
        what the goal tests depend on is kept: balances, remaining undo
        resources, and two facts about the ledger.
        """
        n = len(target.transactions)
        return (
            tuple(sorted(state.accounts.items())),
            tuple(sorted(state.undoable_transactions)),
            tuple(sorted(state.reversed_transactions)),
            tuple(tuple(sorted(u.items())) for u in state.pending_account_undos),
            state.transactions[:n] == target.transactions,
            all(
                tx.get("type") in SYSTEM_ENTRY_TYPES
                for tx in state.transactions[post_length:]
            ),
        )

    def successors(state: WalletState):
        for undo in get_recovery_actions(state):
            try:
                next_state, _ = execute_recovery_action(state, undo)
            except (ValueError, KeyError, TypeError):
                continue
            yield {"type": "undo", **undo}, next_state

        for action in _compensating_actions(state, target):
            try:
                # Recovery actions do not create new undo resources.
                next_state = run_actions(state, [action], record_undo=False)
            except (ValueError, KeyError, TypeError):
                continue
            yield action, next_state

    outcome = dual_bfs(
        start=current,
        successors=successors,
        is_exact=is_exact,
        is_approx=is_approx,
        key=search_key,
        max_depth=max_depth,
    )

    return {
        "labels": labels_from_outcome(outcome),
        "resulting_state": current,
        "recovery_path": outcome.exact_path or outcome.approx_path,
        "nodes": outcome.nodes,
    }
