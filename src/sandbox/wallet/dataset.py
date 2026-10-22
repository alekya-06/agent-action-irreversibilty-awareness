
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from state import WalletState
from actions import get_permissible_actions
from transitions import transition
from experiments import run_experiment
from bfs import classify_reversibility


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "wallet"
OUTPUT_FILE = OUTPUT_DIR / "wallet.jsonl"

STATE_CHANGING_ACTIONS = {
    "create_account",
    "delete_account",
    "deposit",
    "withdraw",
    "transfer",
}


def build_scenarios() -> list[dict[str, Any]]:
    """Original single-action scenarios."""
    return [
        {"scenario_id": "wallet_001", "accounts": {"A": 1000, "B": 500}},
        {"scenario_id": "wallet_002", "accounts": {"A": 100, "B": 0}},
        {"scenario_id": "wallet_003", "accounts": {"A": 5000, "B": 2000, "C": 1000}},
        {"scenario_id": "wallet_004", "accounts": {"A": 0, "B": 0}},
        {"scenario_id": "wallet_005", "accounts": {"A": 100, "B": 100}},
        {"scenario_id": "wallet_006", "accounts": {"A": 10000, "B": 0}},
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


def generate_sequence_example(
    scenario: dict[str, Any],
    record_undo: bool,
    recovery_horizon: int = 3,
) -> dict[str, Any]:
    """Execute a sequence, then classify recovery using the real BFS."""
    scenario_id = scenario["scenario_id"]
    actions = scenario["actions"]

    initial_wallet = WalletState(accounts=scenario["accounts"].copy())
    current_wallet = initial_wallet.clone()

    for action in actions:
        previous_transaction_ids = {
            tx["transaction_id"] for tx in current_wallet.transactions
        }

        current_wallet, _ = transition(current_wallet, action)

        if not record_undo:
            # Remove undo eligibility for transactions created by this step.
            new_transaction_ids = {
                tx["transaction_id"] for tx in current_wallet.transactions
            } - previous_transaction_ids

            current_wallet.undoable_transactions.difference_update(
                new_transaction_ids
            )

    classification = classify_reversibility(
        current_wallet,
        initial_wallet,
        horizon=recovery_horizon,
        extended_horizon=max(10, len(actions)),
    )

    return {
        "scenario_id": scenario_id,
        "action": {
            "type": "sequence",
            "steps": len(actions),
        },
        "action_sequence": actions,
        "record_undo": record_undo,
        "initial_state": initial_wallet.to_dict(),
        "resulting_state": current_wallet.to_dict(),
        "binary_label": classification["binary_label"],
        "trinary_label": classification["trinary_label"],
        "recovery_cost": classification["recovery_cost"],
        "recovery_path": classification["recovery_path"],
        "example_id": (
            f"{scenario_id}_{'with_undo' if record_undo else 'without_undo'}"
        ),
    }


def generate_dataset(
    output_path: str | Path = OUTPUT_FILE,
    recovery_horizon: int = 3,
) -> list[dict[str, Any]]:
    if recovery_horizon < 0:
        raise ValueError("recovery_horizon must be non-negative")

    examples: list[dict[str, Any]] = []

    # Original one-action scenarios.
    for scenario in build_scenarios():
        scenario_id = scenario["scenario_id"]
        initial_wallet = WalletState(accounts=scenario["accounts"].copy())
        actions = get_permissible_actions(initial_wallet)

        for action_index, action in enumerate(actions):
            action_type = action.get("type")
            undo_options = (
                (True, False)
                if action_type in STATE_CHANGING_ACTIONS
                else (True,)
            )

            for record_undo in undo_options:
                result = run_experiment(
                    scenario_id=scenario_id,
                    initial_wallet=initial_wallet,
                    action=action,
                    record_undo=record_undo,
                    recovery_horizon=recovery_horizon,
                )

                result["example_id"] = (
                    f"{scenario_id}_action_{action_index:03d}_"
                    f"{'with_undo' if record_undo else 'without_undo'}"
                )
                examples.append(result)

    # Compound scenarios, paired by scenario ID to prevent split leakage.
    for scenario in build_compound_scenarios():
        for record_undo in (True, False):
            examples.append(
                generate_sequence_example(
                    scenario=scenario,
                    record_undo=record_undo,
                    recovery_horizon=recovery_horizon,
                )
            )

    ids = [example["example_id"] for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example_id values detected")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    return examples


def print_dataset_summary(examples: list[dict[str, Any]]) -> None:
    print("\nWallet dataset generation complete.")
    print(f"Total examples: {len(examples)}")
    print(f"Unique scenarios: {len({e['scenario_id'] for e in examples})}")

    for field, labels in (
        ("binary_label", ("reversible", "irreversible")),
        (
            "trinary_label",
            ("reversible", "partially_reversible", "irreversible"),
        ),
    ):
        counts = Counter(e[field] for e in examples)
        print(f"\n{field.replace('_', ' ').title()}:")
        for label in labels:
            print(f"  {label}: {counts[label]}")

    print("\nRecovery costs:")
    for cost, count in sorted(
        Counter(str(e["recovery_cost"]) for e in examples).items()
    ):
        print(f"  {cost}: {count}")

    print("\nLabels by undo setting:")
    for setting in (True, False):
        subset = [e for e in examples if e.get("record_undo") == setting]
        print(
            f"  record_undo={setting}: {len(subset)} examples — "
            f"{dict(Counter(e['binary_label'] for e in subset))}"
        )

    print("\nAction types:")
    action_types = Counter(
        e.get("action", {}).get("type", "unknown") for e in examples
    )
    print(dict(action_types))


if __name__ == "__main__":
    dataset = generate_dataset()
    print_dataset_summary(dataset)
