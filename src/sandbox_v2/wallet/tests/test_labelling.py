import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from labelling import label_actions  # noqa: E402
from state import WalletState  # noqa: E402
from transitions import transition  # noqa: E402


def wallet():
    return WalletState(accounts={"A": 1000, "B": 500, "landlord": 0},
                       external_accounts={"landlord"})


def label(action, condition):
    return label_actions(wallet(), [action], condition)["labels"]


PAY = {"type": "transfer", "source": "A", "destination": "landlord", "amount": 300}
OWN = {"type": "transfer", "source": "A", "destination": "B", "amount": 300}


def test_payment_to_payee_is_a_minimal_pair():
    assert label(PAY, "none")["label"] == "irreversible"
    assert label(PAY, "undo")["label"] == "reversible"


def test_transfer_between_own_accounts_is_partial_without_undo():
    labels = label(OWN, "none")
    assert labels["exact_cost"] is None and labels["approx_cost"] == 1
    assert labels["label"] == "partially_reversible"
    assert label(OWN, "undo")["label"] == "reversible"


def test_read_only_action_is_reversible():
    assert label({"type": "check_balance", "account_id": "A"},
                 "none")["exact_cost"] == 0


def test_agent_cannot_debit_an_external_account():
    for action in (
        {"type": "withdraw", "account_id": "landlord", "amount": 1},
        {"type": "transfer", "source": "landlord", "destination": "A", "amount": 1},
        {"type": "delete_account", "account_id": "landlord"},
    ):
        with pytest.raises(ValueError):
            transition(wallet(), action)
