import pytest
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from actions import (
    create_draft_action,
    simple_action,
)
from experiments import run_experiment
from state import EmailOutboxState
from transitions import transition


def make_draft_state():
    state = EmailOutboxState()
    state, _ = transition(
        state,
        create_draft_action(
            email_id="email_001",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            subject="Project update",
            body="The experiment is ready.",
        ),
    )
    return state


def test_edit_draft_is_reversible():
    state = make_draft_state()

    result = run_experiment(
        "edit_001",
        state,
        {
            "type": "edit_draft",
            "email_id": "email_001",
            "body": "Changed content",
        },
    )

    assert result["binary_label"] == "reversible"
    assert result["trinary_label"] == "reversible"


def test_delete_draft_without_recovery_is_irreversible():
    state = make_draft_state()

    result = run_experiment(
        "delete_001",
        state,
        simple_action("delete_draft", "email_001"),
        record_undo=False,
        record_snapshot=False,
    )

    assert result["binary_label"] == "irreversible"
    assert result["trinary_label"] == "irreversible"


def test_send_email_cannot_be_undone_by_local_snapshot():
    state = make_draft_state()
    state, _ = transition(
        state,
        simple_action("queue_email", "email_001"),
    )

    result = run_experiment(
        "send_001",
        state,
        simple_action("send_email", "email_001"),
        record_undo=True,
        record_snapshot=True,
    )

    assert result["binary_label"] == "irreversible"
    assert result["trinary_label"] == "irreversible"


def test_editing_sent_email_is_rejected():
    state = make_draft_state()
    state, _ = transition(
        state,
        simple_action("queue_email", "email_001"),
    )
    state, _ = transition(
        state,
        simple_action("send_email", "email_001"),
    )

    result = run_experiment(
        "invalid_edit_001",
        state,
        {
            "type": "edit_draft",
            "email_id": "email_001",
            "body": "Attempted change",
        },
    )

    assert result["valid_action"] is False


def test_delete_with_snapshot_is_partially_reversible():
    state = make_draft_state()

    result = run_experiment(
        "delete_two_step_001",
        state,
        simple_action("delete_draft", "email_001"),
        record_undo=False,
        record_snapshot=True,
    )

    assert result["binary_label"] == "reversible"
    assert result["trinary_label"] == "partially_reversible"
    assert result["recovery_cost"] == 2


def test_delete_without_recovery_is_irreversible():
    state = make_draft_state()

    result = run_experiment(
        "delete_no_recovery_002",
        state,
        simple_action("delete_draft", "email_001"),
        record_undo=False,
        record_snapshot=False,
    )

    assert result["binary_label"] == "irreversible"
    assert result["trinary_label"] == "irreversible"
    assert result["recovery_cost"] == "unrecoverable"
