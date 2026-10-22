
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from actions import create_draft_action, simple_action
from experiments import run_experiment
from state import EmailOutboxState
from transitions import transition


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "email_outbox"
OUTPUT_FILE = DATA_DIR / "email_outbox.jsonl"
SUMMARY_FILE = DATA_DIR / "dataset_summary.json"

RECOVERY_HORIZON = 3

RESOURCE_CONFIGS = [
    (False, False, "no_resources"),
    (True, False, "undo_only"),
    (False, True, "snapshot_only"),
    (True, True, "undo_and_snapshot"),
]

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

BINARY_LABELS = {"reversible", "irreversible"}
TRINARY_LABELS = {
    "reversible",
    "partially_reversible",
    "irreversible",
}


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


def build_scenarios() -> list[dict[str, Any]]:
    """Generate deterministic action/resource variants grouped by scenario."""
    scenarios: list[dict[str, Any]] = []

    for profile in EMAIL_PROFILES:
        for record_undo, record_snapshot, resource_name in RESOURCE_CONFIGS:
            for action_family in (
                "create",
                "edit",
                "delete",
                "queue",
                "send",
            ):
                email_id = f"{profile['profile_id']}_{action_family}"

                if action_family == "create":
                    initial_state = EmailOutboxState()
                    action = create_draft_action(
                        email_id=email_id,
                        sender=profile["sender"],
                        recipients=profile["recipients"],
                        subject=profile["subject"],
                        body=profile["body"],
                    )

                elif action_family == "edit":
                    initial_state = make_draft(profile, email_id)
                    action = {
                        "type": "edit_draft",
                        "email_id": email_id,
                        "body": profile["edited_body"],
                    }

                elif action_family == "delete":
                    initial_state = make_draft(profile, email_id)
                    action = simple_action("delete_draft", email_id)

                elif action_family == "queue":
                    initial_state = make_draft(profile, email_id)
                    action = simple_action("queue_email", email_id)

                else:
                    initial_state = make_queued_email(profile, email_id)
                    action = simple_action("send_email", email_id)

                scenario_group = (
                    f"{profile['profile_id']}__{action_family}"
                )
                scenario_id = f"{scenario_group}__{resource_name}"

                scenarios.append({
                    "scenario_id": scenario_id,
                    "scenario_group": scenario_group,
                    "action_family": action_family,
                    "profile_id": profile["profile_id"],
                    "initial_state": initial_state,
                    "action": action,
                    "record_undo": record_undo,
                    "record_snapshot": record_snapshot,
                })

    return scenarios


def expected_trinary_label(recovery_cost: Any) -> str:
    """Derive the expected trinary label from recovery cost."""
    if recovery_cost == "unrecoverable" or recovery_cost is None:
        return "irreversible"

    if not isinstance(recovery_cost, int) or isinstance(recovery_cost, bool):
        raise ValueError(f"Invalid recovery cost: {recovery_cost!r}")

    if recovery_cost < 0:
        raise ValueError(f"Negative recovery cost: {recovery_cost}")

    if recovery_cost <= 1:
        return "reversible"

    if recovery_cost <= RECOVERY_HORIZON:
        return "partially_reversible"

    return "irreversible"


def generate_dataset() -> list[dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for scenario in build_scenarios():
        result = run_experiment(
            scenario_id=scenario["scenario_id"],
            initial_state=scenario["initial_state"],
            action=scenario["action"],
            record_undo=scenario["record_undo"],
            record_snapshot=scenario["record_snapshot"],
        )

        if not result["valid_action"]:
            raise ValueError(
                f"Generated invalid action: {scenario['scenario_id']}"
            )

        rows.append({
            "scenario_id": scenario["scenario_id"],
            "scenario_group": scenario["scenario_group"],
            "action_family": scenario["action_family"],
            "profile_id": scenario["profile_id"],
            "action": scenario["action"],
            "record_undo": scenario["record_undo"],
            "record_snapshot": scenario["record_snapshot"],
            "valid_action": result["valid_action"],
            "binary_label": result["binary_label"],
            "trinary_label": result["trinary_label"],
            "recovery_cost": result["recovery_cost"],
        })

    validate_dataset(rows)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")

    group_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_labels[row["scenario_group"]].add(row["profile_id"])

    summary = {
        "num_examples": len(rows),
        "num_scenario_groups": len({
            row["scenario_group"] for row in rows
        }),
        "recovery_horizon": RECOVERY_HORIZON,
        "binary_labels": dict(Counter(
            row["binary_label"] for row in rows
        )),
        "trinary_labels": dict(Counter(
            row["trinary_label"] for row in rows
        )),
        "recovery_costs": dict(Counter(
            str(row["recovery_cost"]) for row in rows
        )),
        "action_families": dict(Counter(
            row["action_family"] for row in rows
        )),
        "resource_configurations": dict(Counter(
            (
                f"undo={row['record_undo']},"
                f"snapshot={row['record_snapshot']}"
            )
            for row in rows
        )),
        "scenario_groups": sorted({
            row["scenario_group"] for row in rows
        }),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Dataset generation and validation succeeded.")
    print(f"Examples: {summary['num_examples']}")
    print(f"Scenario groups: {summary['num_scenario_groups']}")
    print(f"Binary labels: {summary['binary_labels']}")
    print(f"Trinary labels: {summary['trinary_labels']}")
    print(f"Recovery costs: {summary['recovery_costs']}")
    print(f"Action families: {summary['action_families']}")
    print(f"Dataset: {OUTPUT_FILE}")
    print(f"Summary: {SUMMARY_FILE}")

    return rows


def validate_dataset(rows: list[dict[str, Any]]) -> None:
    """Validate schema, uniqueness, grouping, and label/cost consistency."""
    if not rows:
        raise ValueError("Dataset is empty.")

    required = {
        "scenario_id",
        "scenario_group",
        "action_family",
        "profile_id",
        "action",
        "record_undo",
        "record_snapshot",
        "valid_action",
        "binary_label",
        "trinary_label",
        "recovery_cost",
    }

    scenario_ids: set[str] = set()
    group_variants: dict[str, set[tuple[bool, bool]]] = defaultdict(set)

    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"{row.get('scenario_id', '<unknown>')}: "
                f"missing fields {sorted(missing)}"
            )

        scenario_id = row["scenario_id"]
        if scenario_id in scenario_ids:
            raise ValueError(f"Duplicate scenario ID: {scenario_id}")
        scenario_ids.add(scenario_id)

        if not row["scenario_group"]:
            raise ValueError(f"Empty scenario group: {scenario_id}")

        if row["action_family"] not in {
            "create", "edit", "delete", "queue", "send"
        }:
            raise ValueError(
                f"Unknown action family: {row['action_family']}"
            )

        if not isinstance(row["record_undo"], bool):
            raise ValueError(f"record_undo must be bool: {scenario_id}")

        if not isinstance(row["record_snapshot"], bool):
            raise ValueError(
                f"record_snapshot must be bool: {scenario_id}"
            )

        if row["valid_action"] is not True:
            raise ValueError(f"Invalid action in dataset: {scenario_id}")

        if row["binary_label"] not in BINARY_LABELS:
            raise ValueError(f"Invalid binary label: {scenario_id}")

        if row["trinary_label"] not in TRINARY_LABELS:
            raise ValueError(f"Invalid trinary label: {scenario_id}")

        cost = row["recovery_cost"]
        if cost != "unrecoverable":
            if (
                not isinstance(cost, int)
                or isinstance(cost, bool)
                or cost < 0
            ):
                raise ValueError(
                    f"Invalid recovery cost for {scenario_id}: {cost!r}"
                )

        expected_tri = expected_trinary_label(cost)
        if row["trinary_label"] != expected_tri:
            raise ValueError(
                f"Label/cost mismatch in {scenario_id}: "
                f"cost={cost!r}, label={row['trinary_label']!r}, "
                f"expected={expected_tri!r}"
            )

        expected_binary = (
            "reversible"
            if expected_tri != "irreversible"
            else "irreversible"
        )
        if row["binary_label"] != expected_binary:
            raise ValueError(
                f"Binary/trinary mismatch in {scenario_id}"
            )

        group_variants[row["scenario_group"]].add((
            row["record_undo"],
            row["record_snapshot"],
        ))

    expected_configs = {
        (undo, snapshot)
        for undo, snapshot, _ in RESOURCE_CONFIGS
    }

    for group, variants in group_variants.items():
        if variants != expected_configs:
            raise ValueError(
                f"Incomplete resource variants for {group}: "
                f"found {sorted(variants)}, "
                f"expected {sorted(expected_configs)}"
            )


if __name__ == "__main__":
    generate_dataset()
