
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any


@dataclass
class WalletState:
    """
    State of the simulated wallet environment.

    Balances are stored as integer paise to avoid floating-point errors.
    Transaction history is append-only.
    """

    accounts: dict[str, int] = field(default_factory=dict)
    transactions: list[dict[str, Any]] = field(default_factory=list)

    # IDs of transactions that have already been reversed.
    reversed_transactions: set[str] = field(default_factory=set)

    # IDs of transactions whose action-specific undo is available.
    undoable_transactions: set[str] = field(default_factory=set)

    next_transaction_id: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check wallet invariants."""
        if not all(
            isinstance(account_id, str) and account_id.strip()
            for account_id in self.accounts
        ):
            raise ValueError("Account IDs must be non-empty strings.")

        for account_id, balance in self.accounts.items():
            if not isinstance(balance, int) or isinstance(balance, bool):
                raise ValueError(
                    f"Balance for {account_id!r} must be integer paise."
                )
            if balance < 0:
                raise ValueError("Account balances cannot be negative.")

        transaction_ids = [
            transaction["transaction_id"]
            for transaction in self.transactions
        ]

        if len(transaction_ids) != len(set(transaction_ids)):
            raise ValueError("Transaction IDs must be unique.")

        known_ids = set(transaction_ids)

        if not self.reversed_transactions.issubset(known_ids):
            raise ValueError("Reversed transaction IDs must exist.")

        if not self.undoable_transactions.issubset(known_ids):
            raise ValueError("Undoable transaction IDs must exist.")

        if self.next_transaction_id < 1:
            raise ValueError("Next transaction ID must be positive.")

    def account_exists(self, account_id: str) -> bool:
        return account_id in self.accounts

    def get_balance(self, account_id: str) -> int:
        if account_id not in self.accounts:
            raise KeyError(f"Account {account_id!r} does not exist.")
        return self.accounts[account_id]

    def total_balance(self) -> int:
        """Total funds held across all existing accounts."""
        return sum(self.accounts.values())

    def clone(self) -> WalletState:
        """Return an independent copy for transitions and experiments."""
        return deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "accounts": dict(self.accounts),
            "transactions": deepcopy(self.transactions),
            "reversed_transactions": sorted(self.reversed_transactions),
            "undoable_transactions": sorted(self.undoable_transactions),
            "next_transaction_id": self.next_transaction_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WalletState:
        return cls(
            accounts=dict(data["accounts"]),
            transactions=deepcopy(data["transactions"]),
            reversed_transactions=set(
                data.get("reversed_transactions", [])
            ),
            undoable_transactions=set(
                data.get("undoable_transactions", [])
            ),
            next_transaction_id=data.get("next_transaction_id", 1),
        )
