import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from state import WalletState
from transitions import transition
from bfs import find_recovery_path, classify_reversibility


def test_transfer_is_reversible_within_horizon():
    initial = WalletState(accounts={"A": 1000, "B": 0})

    current, _ = transition(
        initial,
        {
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 300,
        },
    )

    result = classify_reversibility(current, initial)

    assert result["binary_label"] == "reversible"
    assert result["trinary_label"] == "reversible"
    assert result["recovery_cost"] == 1


def test_read_only_action_requires_zero_recovery_actions():
    initial = WalletState(accounts={"A": 1000})

    current, _ = transition(
        initial,
        {"type": "check_balance", "account_id": "A"},
    )

    path = find_recovery_path(current, initial, max_depth=0)

    assert path == []


def test_recovery_path_is_shortest():
    initial = WalletState(accounts={"A": 1000, "B": 0})

    current, _ = transition(
        initial,
        {
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 300,
        },
    )

    path = find_recovery_path(current, initial, max_depth=3)

    assert path is not None
    assert len(path) == 1


def test_no_recovery_path_returns_none():
    initial = WalletState(accounts={"A": 1000, "B": 0})

    current, _ = transition(
        initial,
        {
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 300,
        },
    )

    # The transfer's undo permission has been removed.
    current.undoable_transactions.clear()

    path = find_recovery_path(current, initial, max_depth=3)

    assert path is None
