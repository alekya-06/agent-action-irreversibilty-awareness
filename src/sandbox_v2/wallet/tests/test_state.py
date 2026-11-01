import pytest
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from state import WalletState

def test_initial_wallet_state():
    state = WalletState(accounts={"A": 10000, "B": 2500})

    assert state.get_balance("A") == 10000
    assert state.get_balance("B") == 2500
    assert state.total_balance() == 12500


def test_missing_account_raises_error():
    state = WalletState(accounts={"A": 10000})

    with pytest.raises(KeyError):
        state.get_balance("B")


def test_negative_balance_is_rejected():
    with pytest.raises(ValueError):
        WalletState(accounts={"A": -100})


def test_clone_is_independent():
    original = WalletState(accounts={"A": 10000})
    copied = original.clone()

    copied.accounts["A"] = 5000

    assert original.accounts["A"] == 10000
    assert copied.accounts["A"] == 5000


def test_serialization_round_trip():
    original = WalletState(accounts={"A": 10000, "B": 0})
    restored = WalletState.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()


def test_transaction_ids_must_be_unique():
    transaction = {
        "transaction_id": "T1",
        "type": "deposit",
        "account_id": "A",
        "amount": 100,
    }

    with pytest.raises(ValueError):
        WalletState(
            accounts={"A": 100},
            transactions=[transaction, dict(transaction)],
        )
