import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actions import create_draft_action, simple_action  # noqa: E402
from labelling import label_actions  # noqa: E402
from state import EmailOutboxState  # noqa: E402
from transitions import transition  # noqa: E402


def draft(status_actions=()):
    state, _ = transition(EmailOutboxState(), create_draft_action(
        email_id="e1", sender="a@x.com", recipients=["b@x.com"],
        subject="s", body="hello"))
    for kind in status_actions:
        state, _ = transition(state, simple_action(kind, "e1"))
    return state


def label(state, action, condition):
    return label_actions(state, [action], condition)["labels"]


def test_send_is_irreversible_in_every_condition():
    for condition in ("none", "undo", "snapshot", "undo+snapshot"):
        assert label(draft(["queue_email"]), simple_action("send_email", "e1"),
                     condition)["label"] == "irreversible"


def test_delete_uses_the_trash_in_two_steps():
    labels = label(draft(), simple_action("delete_draft", "e1"), "none")
    assert labels["exact_cost"] == 2
    assert labels["labels_by_horizon"] == {
        "1": "partially_reversible", "2": "reversible", "3": "reversible"}


def test_edit_back_is_approximate_without_undo():
    edit = {"type": "edit_draft", "email_id": "e1", "body": "changed"}
    none = label(draft(), edit, "none")
    assert none["exact_cost"] is None and none["approx_cost"] == 1
    assert none["label"] == "partially_reversible"
    assert label(draft(), edit, "undo")["label"] == "reversible"


def test_queue_without_resources_is_irreversible():
    assert label(draft(), simple_action("queue_email", "e1"),
                 "none")["label"] == "irreversible"
