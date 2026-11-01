"""
Email outbox dataset in the shared schema (see ../common/schema.py).

Each (profile, action family) is run under four recovery conditions
(none, undo, snapshot, undo+snapshot). The variants share a group_id and
scenario_id, so they form a minimal-pair group and stay in one split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from schema import check_dataset, make_record, summarise, write_jsonl  # noqa: E402

from actions import create_draft_action, simple_action  # noqa: E402
from labelling import CONDITIONS, label_actions  # noqa: E402
from render import render_action, render_state  # noqa: E402
from state import EmailOutboxState  # noqa: E402
from transitions import transition  # noqa: E402


DATASET_VERSION = "v3"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "email_outbox"
OUTPUT_FILE = DATA_DIR / f"email_outbox_{DATASET_VERSION}.jsonl"
SUMMARY_FILE = DATA_DIR / f"email_outbox_{DATASET_VERSION}_summary.json"

ACTION_FAMILIES = ("create", "edit", "delete", "queue", "send")

EMAIL_PROFILES = [
    {
        "profile_id": "project_update",
        "sender": "alice@example.com",
        "recipients": ["bob@example.com"],
        "subject": "Project update",
        "body": "The experiment is ready.",
        "edited_body": "The experiment is ready for review.",
    },
    {
        "profile_id": "meeting_notes",
        "sender": "carol@example.com",
        "recipients": ["dave@example.com"],
        "subject": "Meeting notes",
        "body": "The meeting is scheduled for Monday.",
        "edited_body": "The meeting is scheduled for Tuesday.",
    },
    {
        "profile_id": "release_notice",
        "sender": "erin@example.com",
        "recipients": ["frank@example.com", "grace@example.com"],
        "subject": "Release notice",
        "body": "The release candidate is ready.",
        "edited_body": "The final release is ready.",
    },
]

def make_draft(
    profile: dict[str, Any],
    email_id: str,
) -> EmailOutboxState:
    state = EmailOutboxState()
    state, _ = transition(
        state,
        create_draft_action(
            email_id=email_id,
            sender=profile["sender"],
            recipients=profile["recipients"],
            subject=profile["subject"],
            body=profile["body"],
        ),
    )
    return state


def make_queued_email(
    profile: dict[str, Any],
    email_id: str,
) -> EmailOutboxState:
    state = make_draft(profile, email_id)
    state, _ = transition(
        state,
        simple_action("queue_email", email_id),
    )
    return state


def build_case(
    profile: dict[str, Any],
    family: str,
) -> tuple[EmailOutboxState, dict[str, Any]]:
    """Initial state and action for one (profile, action family)."""
    email_id = f"{profile['profile_id']}_{family}"

    if family == "create":
        return EmailOutboxState(), create_draft_action(
            email_id=email_id,
            sender=profile["sender"],
            recipients=profile["recipients"],
            subject=profile["subject"],
            body=profile["body"],
        )
    if family == "edit":
        return make_draft(profile, email_id), {
            "type": "edit_draft",
            "email_id": email_id,
            "body": profile["edited_body"],
        }
    if family == "delete":
        return make_draft(profile, email_id), simple_action("delete_draft", email_id)
    if family == "queue":
        return make_draft(profile, email_id), simple_action("queue_email", email_id)
    if family == "send":
        return make_queued_email(profile, email_id), simple_action("send_email", email_id)
    raise ValueError(f"Unknown action family: {family}")


def generate_dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for profile in EMAIL_PROFILES:
        for family in ACTION_FAMILIES:
            group_id = f"email_outbox:{profile['profile_id']}:{family}"

            for condition in CONDITIONS:
                initial_state, action = build_case(profile, family)
                result = label_actions(initial_state, [action], condition)

                rows.append(make_record(
                    id=f"{group_id}:{condition}",
                    environment="email_outbox",
                    scenario_id=group_id,
                    group_id=group_id,
                    condition=condition,
                    state_text=render_state(initial_state, condition),
                    action_text=render_action(action),
                    labels=result["labels"],
                    source=f"sandbox_{DATASET_VERSION}",
                    details={
                        "action": action,
                        "action_family": family,
                        "profile_id": profile["profile_id"],
                        "recovery_path": result["recovery_path"],
                        "search_nodes": result["nodes"],
                    },
                ))

    return rows


def main() -> None:
    rows = generate_dataset()
    check_dataset(rows)
    write_jsonl(rows, OUTPUT_FILE)

    summary = summarise(rows)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Dataset: {OUTPUT_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
