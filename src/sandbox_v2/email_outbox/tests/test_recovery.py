
from actions import create_draft_action, simple_action
from recovery import (
    execute_with_undo,
    get_recovery_actions,
    restore_snapshot,
    save_snapshot,
    undo_action,
)
from state import Email, EmailOutboxState
from transitions import transition


def make_draft():
    state = EmailOutboxState()
    state, _ = transition(
        state,
        create_draft_action(
            "e1", "alice@example.com",
            ["bob@example.com"], "Report", "Original",
        ),
    )
    return state


def test_save_and_restore_snapshot():
    state = make_draft()
    state = save_snapshot(state, "before_edit")

    edited, _ = transition(
        state,
        {"type": "edit_draft", "email_id": "e1", "body": "Changed"},
    )

    restored = restore_snapshot(edited, "before_edit")

    assert restored.get_email("e1").body == "Original"


def test_missing_snapshot_raises():
    try:
        restore_snapshot(EmailOutboxState(), "missing")
    except KeyError:
        pass
    else:
        raise AssertionError("Missing snapshot was accepted")


def test_snapshot_restore_preserves_external_send_history():
    state = make_draft()
    state = save_snapshot(state, "before_send")

    queued, _ = transition(state, simple_action("queue_email", "e1"))
    sent, _ = transition(queued, simple_action("send_email", "e1"))

    restored = restore_snapshot(sent, "before_send")

    assert restored.get_email("e1").status == "draft"
    assert len(restored.sent_history) == 1
    assert restored.sent_history[0]["email_id"] == "e1"


def test_undo_restores_previous_email():
    state = make_draft()

    restored = undo_action(
        state,
        {
            "type": "restore_email",
            "email_id": "e1",
            "email": {
                **state.get_email("e1").__dict__,
                "body": "Recovered",
            },
        },
    )

    assert restored.get_email("e1").body == "Recovered"


def test_undo_remove_email():
    state = make_draft()

    updated = undo_action(
        state, {"type": "remove_email", "email_id": "e1"}
    )

    assert "e1" not in updated.emails


def test_execute_with_undo_records_recovery():
    state = make_draft()

    updated, info = execute_with_undo(
        state,
        {"type": "edit_draft", "email_id": "e1", "body": "Changed"},
        {
            "type": "restore_email",
            "email_id": "e1",
            "email": state.get_email("e1").__dict__.copy(),
        },
    )

    assert info["success"] is True
    assert len(get_recovery_actions(updated, "e1")) == 1
