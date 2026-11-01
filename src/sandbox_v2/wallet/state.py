
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
    pending_account_undos: list[dict[str, Any]] = field(default_factory=list)

    next_transaction_id: int = 1

    # Accounts that belong to other people (payees). Money can be sent to
    # them, but the agent cannot withdraw, transfer or delete from them.
    # Only an explicit undo (e.g. cancelling a payment) can take it back.
    external_accounts: set[str] = field(default_factory=set)

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
    
        for undo in self.pending_account_undos:
            if not isinstance(undo, dict):
                raise ValueError("Pending account undos must be dictionaries.")
        
            if undo.get("type") not in {
                "undo_create_account",
                "undo_delete_account",
            }:
                raise ValueError("Invalid pending account undo type.")
        
            if not isinstance(undo.get("account_id"), str) or not undo["account_id"].strip():
                raise ValueError("Pending account undo requires an account ID.")

        if not self.external_accounts.issubset(self.accounts):
            raise ValueError("External accounts must exist in accounts.")

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
            "pending_account_undos": deepcopy(self.pending_account_undos),
            "next_transaction_id": self.next_transaction_id,
            "external_accounts": sorted(self.external_accounts),
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
            pending_account_undos=deepcopy(
                    data.get("pending_account_undos", [])
            ),
            next_transaction_id=data.get("next_transaction_id", 1),
            external_accounts=set(data.get("external_accounts", [])),
        )
