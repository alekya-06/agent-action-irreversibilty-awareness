import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labelling import label_actions  # noqa: E402
from state import Email, EmailOutboxState  # noqa: E402
from transitions import transition  # noqa: E402

SEND = {"type": "send_email", "email_id": "e1"}


def queued_state():
    state = EmailOutboxState()
    state.add_email(Email(email_id="e1", sender="a@example.com", recipients=["b@example.com"],
                          subject="Offer", body="Here is the offer.", status="queued"))
    return state


def label(condition):
    return label_actions(queued_state(), [SEND], condition)["labels"]["label"]


def test_send_without_window_is_irreversible():
    assert label("none") == "irreversible"


def test_send_with_undo_send_is_reversible():
    assert label("undo_send") == "reversible"


def test_recall_fails_without_window():
    state, _ = transition(queued_state(), SEND)
    with pytest.raises(ValueError):
        transition(state, {"type": "recall_email", "email_id": "e1"})


def test_recall_restores_queued_state():
    state = queued_state()
    state.undo_send_window = 3
    before = len(state.sent_history)
    state, _ = transition(state, SEND)
    state, _ = transition(state, {"type": "recall_email", "email_id": "e1"})
    assert state.emails["e1"].status == "queued"
    assert len(state.sent_history) == before
    assert state.recallable == {}