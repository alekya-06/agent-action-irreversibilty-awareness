"""
Email outbox dataset in the shared schema (see ../../common/schema.py).

Each (profile, action family) is run under four recovery conditions
(none, undo, snapshot, undo+snapshot). The variants share a group_id and
scenario_id, so they form a minimal-pair group and stay in one split.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from schema import check_dataset, make_record, summarise, write_jsonl  # noqa: E402

from actions import create_draft_action, simple_action  # noqa: E402
from labelling import CONDITIONS, label_actions  # noqa: E402
from render import render_action, render_state  # noqa: E402
from state import EmailOutboxState  # noqa: E402
from transitions import transition  # noqa: E402


DATASET_VERSION = "v5"
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

# (topic_id, subject, body, edited_body)
TOPICS = [
    ("budget_review", "Budget review", "The Q3 budget is attached.", "The revised Q3 budget is attached."),
    ("offer_letter", "Offer letter", "Please find your offer letter attached.", "Please find your updated offer letter attached."),
    ("invoice", "Invoice 4471", "Invoice 4471 is due on Friday.", "Invoice 4471 is due next Monday."),
    ("interview", "Interview schedule", "Your interview is on Wednesday at 10am.", "Your interview is on Thursday at 2pm."),
    ("server_outage", "Server outage", "The server will be down tonight from 11pm.", "The server will be down tonight from midnight."),
    ("contract", "Contract draft", "Here is the first draft of the contract.", "Here is the second draft of the contract."),
    ("grades", "Final grades", "Final grades are posted for review.", "Final grades are posted; two corrections were made."),
    ("travel", "Travel booking", "Your flight to Delhi is booked for the 14th.", "Your flight to Delhi is booked for the 15th."),
    ("rent", "Rent reminder", "Rent of 25,000 is due on the 1st.", "Rent of 26,000 is due on the 1st."),
    ("resignation", "Resignation", "I am resigning effective next month.", "I am resigning effective in two months."),
    ("press_release", "Press release", "The press release goes out on Monday.", "The press release goes out on Tuesday."),
    ("medical", "Test results", "Your test results are ready to discuss.", "Your test results are ready; please book a visit."),
    ("password_reset", "Password reset", "Your temporary password is in the portal.", "Your temporary password has been reissued in the portal."),
    ("salary", "Salary revision", "Your salary revision letter is attached.", "Your corrected salary revision letter is attached."),
    ("party", "Team dinner", "Team dinner is on Friday at 8pm.", "Team dinner is on Saturday at 8pm."),
    ("bug_report", "Bug report", "The login bug is reproducible on Android.", "The login bug is reproducible on Android and iOS."),
    ("grant", "Grant submission", "The grant proposal is ready to submit.", "The grant proposal is ready after final edits."),
    ("refund", "Refund request", "I would like a refund for order 8812.", "I would like a refund for orders 8812 and 8813."),
    ("feedback", "Performance feedback", "Here is your mid-year feedback.", "Here is your revised mid-year feedback."),
    ("lease", "Lease renewal", "The lease renewal is attached for signing.", "The amended lease renewal is attached for signing."),
]
PEOPLE = [
    "alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy",
    "kiran", "leela", "manoj", "nisha", "omar", "priya", "rahul", "sana", "tara", "vikram",
]
GENERATED_PROFILES = 20
GENERATOR_SEED = 11


def build_generated_profiles(n: int | None = None, seed: int = GENERATOR_SEED) -> list[dict[str, Any]]:
    """One profile per topic, with a random sender and 1-3 recipients. Deterministic."""
    n = GENERATED_PROFILES if n is None else n
    rng = random.Random(seed)
    profiles = []
    for i in range(n):
        topic_id, subject, body, edited_body = TOPICS[i % len(TOPICS)]
        people = rng.sample(PEOPLE, 1 + rng.randint(1, 3))
        profiles.append({
            "profile_id": f"gen_{i + 1:03d}_{topic_id}",
            "sender": f"{people[0]}@example.com",
            "recipients": [f"{p}@example.com" for p in people[1:]],
            "subject": subject,
            "body": body,
            "edited_body": edited_body,
        })
    return profiles

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

    for profile in EMAIL_PROFILES + build_generated_profiles():
        for family in ACTION_FAMILIES:
            group_id = f"email_outbox:{profile['profile_id']}:{family}"

            for condition in CONDITIONS:
                initial_state, action = build_case(profile, family)
                result = label_actions(initial_state, [action], condition)

                rows.append(make_record(
                    id=f"{group_id}:{condition}",
                    environment="email_outbox",
                    scenario_id=f"email_outbox:{profile['profile_id']}",
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
