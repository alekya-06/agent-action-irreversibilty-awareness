import os
import sys
import pytest

parent_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(parent_dir)

from state import WalletState
from experiments import run_experiment

def test_transfer_is_reversible():
    wallet = WalletState(
        accounts={"A": 1000, "B": 0}
    )

    result = run_experiment(
        scenario_id="transfer_001",
        initial_wallet=wallet,
        action={
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 100,
        },
    )

    assert result["binary_label"] == "reversible"
    assert result["trinary_label"] == "reversible"
    assert result["recovery_cost"] == 1
    assert wallet.accounts == {"A": 1000, "B": 0}


def test_read_only_action_needs_no_recovery():
    wallet = WalletState(accounts={"A": 1000})

    result = run_experiment(
        scenario_id="balance_001",
        initial_wallet=wallet,
        action={
            "type": "check_balance",
            "account_id": "A",
        },
    )

    assert result["binary_label"] == "reversible"
    assert result["trinary_label"] == "reversible"
    assert result["recovery_cost"] == 0


def test_transfer_without_undo_record_is_not_reversible():
    wallet = WalletState(
        accounts={"A": 1000, "B": 0}
    )

    result = run_experiment(
        scenario_id="transfer_002",
        initial_wallet=wallet,
        action={
            "type": "transfer",
            "source": "A",
            "destination": "B",
            "amount": 100,
        },
        record_undo=False,
    )

    assert result["binary_label"] == "irreversible"
    assert result["trinary_label"] == "irreversible"
