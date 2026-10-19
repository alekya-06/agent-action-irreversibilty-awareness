import pytest
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from state import WalletState
from transitions import transition, apply_undo


def test_check_balance_is_read_only():
    state = WalletState(accounts={"A": 10000})

    new_state, record = transition(
        state,
        {"type": "check_balance", "account_id": "A"},
    )

    assert new_state.accounts == state.accounts
    assert record["result"]["balance"] == 10000
    assert record["changed"] is False


def test_create_and_delete_empty_account():
    state = WalletState(accounts={"A": 10000})

    created, _ = transition(
        state,
        {"type": "create_account", "account_id": "B"},
    )
    assert created.accounts["B"] == 0

    deleted, _ = transition(
        created,
        {"type": "delete_account", "account_id": "B"},
    )
    assert "B" not in deleted.accounts


def test_cannot_delete_nonempty_account():
    state = WalletState(accounts={"A": 10000})

    with pytest.raises(ValueError):
        transition(
            state,
            {"type": "delete_account", "account_id": "A"},
        )


def test_deposit_and_withdraw():
    state = WalletState(accounts={"A": 10000})

    deposited, _ = transition(
        state,
        {"type": "deposit", "account_id": "A", "amount": 500},
    )
    assert deposited.accounts["A"] == 10500

    withdrawn, _ = transition(
        deposited,
        {"type": "withdraw", "account_id": "A", "amount": 200},
    )
    assert withdrawn.accounts["A"] == 10300


def test_insufficient_funds_rejected():
    state = WalletState(accounts={"A": 100})

    with pytest.raises(ValueError):
        transition(
            state,
            {"type": "withdraw", "account_id": "A", "amount": 101},
        )


def test_transfer_and_reverse_preserves_history():
    state = WalletState(accounts={"A": 1000, "B": 0})

    transferred, transfer_record = transition(
        state,
        {
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 300,
        },
    )

    transaction_id = transfer_record["undo"]["transaction_id"]

    assert transferred.accounts == {"A": 700, "B": 300}

    reversed_state, _ = transition(
        transferred,
        {"type": "reverse_transfer", "transaction_id": transaction_id},
    )

    assert reversed_state.accounts == {"A": 1000, "B": 0}
    assert len(reversed_state.transactions) == 2
    assert reversed_state.transactions[0]["type"] == "transfer"
    assert reversed_state.transactions[1]["type"] == "transfer_reversal"


def test_transfer_undo_requires_available_funds():
    state = WalletState(accounts={"A": 1000, "B": 0})

    transferred, record = transition(
        state,
        {
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 300,
        },
    )

    transaction_id = record["undo"]["transaction_id"]

    spent, _ = transition(
        transferred,
        {
            "type": "transfer",
            "source": "B",
            "destination": "A",
            "amount": 300,
        },
    )

    with pytest.raises(ValueError):
        apply_undo(spent, record["undo"])


def test_original_state_is_not_mutated():
    state = WalletState(accounts={"A": 1000, "B": 0})

    transition(
        state,
        {
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 100,
        },
    )

    assert state.accounts == {"A": 1000, "B": 0}
    assert state.transactions == []


def test_transfer_cannot_be_reversed_twice():
    state = WalletState(accounts={"A": 1000, "B": 0})

    transferred, record = transition(
        state,
        {
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 300,
        },
    )

    transaction_id = record["undo"]["transaction_id"]

    reversed_state, _ = transition(
        transferred,
        {"type": "reverse_transfer", "transaction_id": transaction_id},
    )

    with pytest.raises(ValueError):
        transition(
            reversed_state,
            {"type": "reverse_transfer", "transaction_id": transaction_id},
        )
