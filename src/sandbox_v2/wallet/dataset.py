"""
Wallet dataset in the shared schema (see ../common/schema.py).

Every action is run under two recovery conditions (none, undo); the two
variants share a group_id. All examples built from the same initial wallet
share a scenario_id, so they stay in one split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from schema import check_dataset, make_record, summarise, write_jsonl  # noqa: E402

from actions import get_permissible_actions  # noqa: E402
from labelling import CONDITIONS, label_actions  # noqa: E402
from render import render_actions, render_state  # noqa: E402
from state import WalletState  # noqa: E402


DATASET_VERSION = "v3"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "wallet"
OUTPUT_FILE = OUTPUT_DIR / f"wallet_{DATASET_VERSION}.jsonl"
SUMMARY_FILE = OUTPUT_DIR / f"wallet_{DATASET_VERSION}_summary.json"


def build_scenarios() -> list[dict[str, Any]]:
    """Original single-action scenarios."""
    return [
        {"scenario_id": "wallet_001", "accounts": {"A": 1000, "B": 500}},
        {"scenario_id": "wallet_002", "accounts": {"A": 100, "B": 0}},
        {"scenario_id": "wallet_003", "accounts": {"A": 5000, "B": 2000, "C": 1000}},
        {"scenario_id": "wallet_004", "accounts": {"A": 0, "B": 0}},
        {"scenario_id": "wallet_005", "accounts": {"A": 100, "B": 100}},
        {"scenario_id": "wallet_006", "accounts": {"A": 10000, "B": 0}},
        # Scenarios with payees: accounts owned by other people.
        {"scenario_id": "wallet_007", "accounts": {"A": 5000, "landlord": 0},
         "external": ["landlord"]},
        {"scenario_id": "wallet_008",
         "accounts": {"A": 2000, "B": 1000, "electricity_board": 0},
         "external": ["electricity_board"]},
        {"scenario_id": "wallet_009", "accounts": {"A": 1500, "B": 300, "friend_ravi": 0},
         "external": ["friend_ravi"]},
    ]


def build_compound_scenarios() -> list[dict[str, Any]]:
    """
    Compound scenarios use varied operation types, accounts, amounts,
    and sequence lengths. Each scenario is kept together when splitting.
    """
    return [
        {
            "scenario_id": "wallet_seq_001",
            "accounts": {"A": 1000, "B": 600, "C": 400},
            "actions": [
                {"type": "deposit", "account_id": "A", "amount": 25},
                {"type": "withdraw", "account_id": "A", "amount": 10},
            ],
        },
        {
            "scenario_id": "wallet_seq_002",
            "accounts": {"A": 800, "B": 500, "C": 300},
            "actions": [
                {"type": "transfer", "source": "A", "destination": "B", "amount": 40},
                {"type": "deposit", "account_id": "B", "amount": 75},
                {"type": "withdraw", "account_id": "C", "amount": 20},
            ],
        },
        {
            "scenario_id": "wallet_seq_003",
            "accounts": {"A": 900, "B": 700, "C": 500},
            "actions": [
                {"type": "deposit", "account_id": "C", "amount": 50},
                {"type": "transfer", "source": "B", "destination": "C", "amount": 30},
                {"type": "withdraw", "account_id": "A", "amount": 20},
                {"type": "deposit", "account_id": "A", "amount": 60},
            ],
        },
        {
            "scenario_id": "wallet_seq_004",
            "accounts": {"A": 1000, "B": 800, "C": 600},
            "actions": [
                {"type": "transfer", "source": "A", "destination": "B", "amount": 100},
                {"type": "withdraw", "account_id": "B", "amount": 25},
                {"type": "deposit", "account_id": "C", "amount": 90},
                {"type": "transfer", "source": "C", "destination": "A", "amount": 40},
                {"type": "withdraw", "account_id": "A", "amount": 15},
            ],
        },
        {
            "scenario_id": "wallet_seq_005",
            "accounts": {"A": 700, "B": 600, "C": 500},
            "actions": [
                {"type": "withdraw", "account_id": "A", "amount": 30},
                {"type": "deposit", "account_id": "B", "amount": 45},
            ],
        },
        {
            "scenario_id": "wallet_seq_006",
            "accounts": {"A": 1200, "B": 900, "C": 700},
            "actions": [
                {"type": "transfer", "source": "A", "destination": "C", "amount": 80},
                {"type": "withdraw", "account_id": "C", "amount": 35},
                {"type": "deposit", "account_id": "B", "amount": 120},
            ],
        },
        {
            "scenario_id": "wallet_seq_007",
            "accounts": {"A": 1500, "B": 1000, "C": 800},
            "actions": [
                {"type": "deposit", "account_id": "A", "amount": 60},
                {"type": "transfer", "source": "A", "destination": "B", "amount": 45},
                {"type": "withdraw", "account_id": "B", "amount": 20},
                {"type": "deposit", "account_id": "C", "amount": 35},
            ],
        },
        {
            "scenario_id": "wallet_seq_008",
            "accounts": {"A": 1000, "B": 900, "C": 800},
            "actions": [
                {"type": "transfer", "source": "B", "destination": "A", "amount": 55},
                {"type": "deposit", "account_id": "C", "amount": 25},
                {"type": "withdraw", "account_id": "A", "amount": 40},
                {"type": "transfer", "source": "A", "destination": "C", "amount": 30},
                {"type": "deposit", "account_id": "B", "amount": 70},
            ],
        },
        {
            "scenario_id": "wallet_seq_009",
            "accounts": {"A": 600, "B": 500, "C": 400},
            "actions": [
                {"type": "deposit", "account_id": "B", "amount": 100},
                {"type": "withdraw", "account_id": "A", "amount": 50},
            ],
        },
        {
            "scenario_id": "wallet_seq_010",
            "accounts": {"A": 900, "B": 750, "C": 650},
            "actions": [
                {"type": "transfer", "source": "A", "destination": "B", "amount": 60},
                {"type": "deposit", "account_id": "C", "amount": 80},
                {"type": "withdraw", "account_id": "B", "amount": 25},
            ],
        },
        {
            "scenario_id": "wallet_seq_011",
            "accounts": {"A": 1100, "B": 900, "C": 700},
            "actions": [
                {"type": "withdraw", "account_id": "A", "amount": 40},
                {"type": "transfer", "source": "B", "destination": "C", "amount": 75},
                {"type": "deposit", "account_id": "A", "amount": 20},
                {"type": "withdraw", "account_id": "C", "amount": 30},
            ],
        },
        {
            "scenario_id": "wallet_seq_012",
            "accounts": {"A": 1300, "B": 1000, "C": 900},
            "actions": [
                {"type": "deposit", "account_id": "A", "amount": 35},
                {"type": "transfer", "source": "A", "destination": "B", "amount": 65},
                {"type": "withdraw", "account_id": "B", "amount": 25},
                {"type": "transfer", "source": "C", "destination": "A", "amount": 90},
                {"type": "deposit", "account_id": "C", "amount": 55},
            ],
        },
    ]


def make_wallet(scenario: dict[str, Any]) -> WalletState:
    return WalletState(
        accounts=dict(scenario["accounts"]),
        external_accounts=set(scenario.get("external", [])),
    )


def _rows_for(
    scenario: dict[str, Any],
    group_id: str,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    unit = "single" if len(actions) == 1 else "sequence"
    for condition in CONDITIONS:
        initial = make_wallet(scenario)
        result = label_actions(initial, actions, condition)
        rows.append(make_record(
            id=f"{group_id}:{condition}",
            environment="wallet",
            scenario_id=f"wallet:{scenario['scenario_id']}",
            group_id=group_id,
            condition=condition,
            state_text=render_state(initial, condition),
            action_text=render_actions(actions),
            action_unit=unit,
            labels=result["labels"],
            source=f"sandbox_{DATASET_VERSION}",
            details={
                "actions": actions,
                "action_type": actions[0]["type"] if unit == "single" else "sequence",
                "recovery_path": result["recovery_path"],
                "search_nodes": result["nodes"],
            },
        ))
    return rows


def generate_dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for scenario in build_scenarios():
        actions = get_permissible_actions(make_wallet(scenario))
        for index, action in enumerate(actions):
            group_id = f"wallet:{scenario['scenario_id']}:a{index:03d}"
            rows.extend(_rows_for(scenario, group_id, [action]))

    for scenario in build_compound_scenarios():
        group_id = f"wallet:{scenario['scenario_id']}"
        rows.extend(_rows_for(scenario, group_id, scenario["actions"]))

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
