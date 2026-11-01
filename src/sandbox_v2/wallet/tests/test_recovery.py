import pytest
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from state import WalletState
from transitions import transition
from recovery import get_recovery_actions, execute_recovery_action

def test_transfer_undo_is_available():
    state = WalletState(accounts={"A": 1000, "B": 0})

    transferred, _ = transition(
        state,
        {
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 300,
        },
    )

    actions = get_recovery_actions(transferred)

    assert len(actions) == 1
    assert actions[0]["type"] == "undo_transfer"


def test_transfer_undo_is_unavailable_after_funds_are_spent():
    state = WalletState(accounts={"A": 1000, "B": 0})

    transferred, _ = transition(
        state,
        {
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 300,
        },
    )

    spent, _ = transition(
        transferred,
        {
            "type": "transfer",
            "source": "B",
            "destination": "A",
            "amount": 300,
        },
    )

    actions = get_recovery_actions(spent)

    assert not any(
        action["type"] == "undo_transfer"
        and action["transaction_id"] == "T1"
        for action in actions
    )


def test_undo_transfer_preserves_original_transaction():
    state = WalletState(accounts={"A": 1000, "B": 0})

    transferred, _ = transition(
        state,
        {
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 300,
        },
    )

    undo = get_recovery_actions(transferred)[0]
    recovered, record = execute_recovery_action(transferred, undo)

    assert recovered.accounts == {"A": 1000, "B": 0}
    assert len(recovered.transactions) == 2
    assert recovered.transactions[0]["type"] == "transfer"
    assert recovered.transactions[1]["type"] == "transfer_reversal"
    assert record["changed"] is True

    # The input state remains unchanged.
    assert transferred.accounts == {"A": 700, "B": 300}
    assert len(transferred.transactions) == 1


def test_recovery_actions_are_not_ordinary_actions():
    state = WalletState(accounts={"A": 1000, "B": 0})

    transferred, _ = transition(
        state,
        {
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 300,
        },
    )

    recovery_action = get_recovery_actions(transferred)[0]

    with pytest.raises(ValueError):
        transition(transferred, recovery_action)


def test_deposit_undo_requires_sufficient_remaining_balance():
    state = WalletState(accounts={"A": 0})

    deposited, _ = transition(
        state,
        {"type": "deposit", "account_id": "A", "amount": 500},
    )

    # Simulate spending the deposited amount after the deposit.
    spent, _ = transition(
        deposited,
        {"type": "withdraw", "account_id": "A", "amount": 500},
    )

    actions = get_recovery_actions(spent)

    assert not any(
        action["type"] == "undo_deposit"
        and action["transaction_id"] == "T1"
        for action in actions
    )
