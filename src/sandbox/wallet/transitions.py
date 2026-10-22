
from __future__ import annotations

from typing import Any

from state import WalletState


ACCOUNT_UNDO_TYPES = {
    "undo_create_account",
    "undo_delete_account",
}


def transition(
    wallet: WalletState,
    action: dict[str, Any],
) -> tuple[WalletState, dict[str, Any]]:
    """Apply one ordinary wallet action without mutating the input state."""
    if not isinstance(action, dict):
        raise TypeError("Action must be a dictionary.")

    action_type = action.get("type")
    if not isinstance(action_type, str):
        raise ValueError("Action must have a string 'type'.")

    new_wallet = wallet.clone()
    before = wallet.to_dict()

    if action_type == "check_balance":
        account_id = _require_account_id(action, "account_id")
        return new_wallet, {
            "action": dict(action),
            "changed": False,
            "result": {
                "account_id": account_id,
                "balance": wallet.get_balance(account_id),
            },
        }

    if action_type == "view_transactions":
        return new_wallet, {
            "action": dict(action),
            "changed": False,
            "result": {
                "transactions": [
                    dict(transaction)
                    for transaction in wallet.transactions
                ]
            },
        }

    if action_type == "create_account":
        account_id = _require_account_id(action, "account_id")
        initial_balance = _require_amount(
            action.get("initial_balance", 0),
            allow_zero=True,
        )

        if account_id in new_wallet.accounts:
            raise ValueError(f"Account {account_id!r} already exists.")

        if initial_balance != 0:
            raise ValueError(
                "Account creation must start with zero balance. "
                "Use deposit to introduce funds."
            )

        new_wallet.accounts[account_id] = 0
        undo = {
            "type": "undo_create_account",
            "account_id": account_id,
        }

    elif action_type == "delete_account":
        account_id = _require_account_id(action, "account_id")
        balance = new_wallet.get_balance(account_id)

        if balance != 0:
            raise ValueError(
                "Cannot delete an account with a nonzero balance."
            )

        del new_wallet.accounts[account_id]
        undo = {
            "type": "undo_delete_account",
            "account_id": account_id,
            "balance": balance,
        }

    elif action_type == "deposit":
        account_id = _require_account_id(action, "account_id")
        amount = _require_amount(action.get("amount"))

        new_wallet.get_balance(account_id)
        new_wallet.accounts[account_id] += amount

        transaction = _append_transaction(
            new_wallet,
            {
                "type": "deposit",
                "account_id": account_id,
                "amount": amount,
            },
        )

        transaction_id = transaction["transaction_id"]
        new_wallet.undoable_transactions.add(transaction_id)

        undo = {
            "type": "undo_deposit",
            "transaction_id": transaction_id,
            "account_id": account_id,
            "amount": amount,
        }

    elif action_type == "withdraw":
        account_id = _require_account_id(action, "account_id")
        amount = _require_amount(action.get("amount"))
        balance = new_wallet.get_balance(account_id)

        if balance < amount:
            raise ValueError("Insufficient funds for withdrawal.")

        new_wallet.accounts[account_id] -= amount

        transaction = _append_transaction(
            new_wallet,
            {
                "type": "withdraw",
                "account_id": account_id,
                "amount": amount,
            },
        )

        transaction_id = transaction["transaction_id"]
        new_wallet.undoable_transactions.add(transaction_id)

        undo = {
            "type": "undo_withdraw",
            "transaction_id": transaction_id,
            "account_id": account_id,
            "amount": amount,
        }

    elif action_type == "transfer":
        source = _require_account_id(action, "source")
        destination = _require_account_id(action, "destination")
        amount = _require_amount(action.get("amount"))

        if source == destination:
            raise ValueError("Source and destination must differ.")

        source_balance = new_wallet.get_balance(source)
        new_wallet.get_balance(destination)

        if source_balance < amount:
            raise ValueError("Insufficient funds for transfer.")

        new_wallet.accounts[source] -= amount
        new_wallet.accounts[destination] += amount

        transaction = _append_transaction(
            new_wallet,
            {
                "type": "transfer",
                "source": source,
                "destination": destination,
                "amount": amount,
            },
        )

        transaction_id = transaction["transaction_id"]
        new_wallet.undoable_transactions.add(transaction_id)

        undo = {
            "type": "undo_transfer",
            "transaction_id": transaction_id,
            "source": source,
            "destination": destination,
            "amount": amount,
        }

    elif action_type == "reverse_transfer":
        transaction_id = action.get("transaction_id")

        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError("A transaction ID is required.")

        original = _find_transaction(new_wallet, transaction_id)

        if original.get("type") != "transfer":
            raise ValueError("Only transfers can be reversed.")

        if transaction_id in new_wallet.reversed_transactions:
            raise ValueError("Transfer has already been reversed.")

        if transaction_id not in new_wallet.undoable_transactions:
            raise ValueError("This transfer is not eligible for reversal.")

        source = original["source"]
        destination = original["destination"]
        amount = original["amount"]

        new_wallet.get_balance(source)
        destination_balance = new_wallet.get_balance(destination)

        if destination_balance < amount:
            raise ValueError(
                "Insufficient funds in destination to reverse transfer."
            )

        new_wallet.accounts[destination] -= amount
        new_wallet.accounts[source] += amount

        reversal = _append_transaction(
            new_wallet,
            {
                "type": "transfer_reversal",
                "original_transaction_id": transaction_id,
                "source": destination,
                "destination": source,
                "amount": amount,
            },
        )

        new_wallet.reversed_transactions.add(transaction_id)
        new_wallet.undoable_transactions.discard(transaction_id)

        undo = {
            "type": "undo_transfer_reversal",
            "transaction_id": reversal["transaction_id"],
            "original_transaction_id": transaction_id,
            "source": source,
            "destination": destination,
            "amount": amount,
        }

    else:
        raise ValueError(f"Unsupported wallet action: {action_type!r}")

    # Account undos must be registered when the ordinary action occurs.
    # They are consumed later, only when apply_undo successfully uses them.
    if undo["type"] in ACCOUNT_UNDO_TYPES:
        new_wallet.pending_account_undos.append(dict(undo))

    new_wallet.validate()

    return new_wallet, {
        "action": dict(action),
        "changed": True,
        "undo": dict(undo),
        "before": before,
        "after": new_wallet.to_dict(),
    }


def apply_undo(
    wallet: WalletState,
    undo: dict[str, Any],
) -> tuple[WalletState, dict[str, Any]]:
    """Apply one action-specific undo without mutating the input state."""
    if not isinstance(undo, dict):
        raise TypeError("Undo action must be a dictionary.")

    new_wallet = wallet.clone()
    undo_type = undo.get("type")

    # An account undo must correspond to an undo token produced by a
    # real create_account or delete_account action.
    if undo_type in ACCOUNT_UNDO_TYPES:
        if undo not in new_wallet.pending_account_undos:
            raise ValueError("Account undo is unavailable.")

    if undo_type == "undo_create_account":
        account_id = _require_account_id(undo, "account_id")

        if account_id not in new_wallet.accounts:
            raise ValueError("Created account no longer exists.")

        if new_wallet.accounts[account_id] != 0:
            raise ValueError(
                "Cannot undo account creation while its balance is nonzero."
            )

        if any(
            account_id in (
                transaction.get("account_id"),
                transaction.get("source"),
                transaction.get("destination"),
            )
            for transaction in new_wallet.transactions
        ):
            raise ValueError(
                "Cannot remove an account referenced by transaction history."
            )

        del new_wallet.accounts[account_id]

    elif undo_type == "undo_delete_account":
        account_id = _require_account_id(undo, "account_id")

        if account_id in new_wallet.accounts:
            raise ValueError("Account already exists.")

        if undo.get("balance") != 0:
            raise ValueError("Only empty-account deletion is supported.")

        new_wallet.accounts[account_id] = 0

    elif undo_type == "undo_deposit":
        account_id = _require_account_id(undo, "account_id")
        amount = _require_amount(undo.get("amount"))
        transaction_id = undo.get("transaction_id")

        original = _find_transaction(new_wallet, transaction_id)
        if (
            original.get("type") != "deposit"
            or original.get("account_id") != account_id
            or original.get("amount") != amount
        ):
            raise ValueError("Undo does not match the original deposit.")

        if transaction_id not in new_wallet.undoable_transactions:
            raise ValueError("Deposit undo is unavailable.")

        if new_wallet.get_balance(account_id) < amount:
            raise ValueError("Insufficient funds to undo deposit.")

        new_wallet.accounts[account_id] -= amount
        new_wallet.undoable_transactions.remove(transaction_id)

    elif undo_type == "undo_withdraw":
        account_id = _require_account_id(undo, "account_id")
        amount = _require_amount(undo.get("amount"))
        transaction_id = undo.get("transaction_id")

        original = _find_transaction(new_wallet, transaction_id)
        if (
            original.get("type") != "withdraw"
            or original.get("account_id") != account_id
            or original.get("amount") != amount
        ):
            raise ValueError("Undo does not match the original withdrawal.")

        if transaction_id not in new_wallet.undoable_transactions:
            raise ValueError("Withdrawal undo is unavailable.")

        new_wallet.get_balance(account_id)
        new_wallet.accounts[account_id] += amount
        new_wallet.undoable_transactions.remove(transaction_id)

    elif undo_type == "undo_transfer":
        transaction_id = undo.get("transaction_id")
        source = _require_account_id(undo, "source")
        destination = _require_account_id(undo, "destination")
        amount = _require_amount(undo.get("amount"))

        original = _find_transaction(new_wallet, transaction_id)

        if original.get("type") != "transfer":
            raise ValueError("Undo target is not a transfer.")

        if (
            original.get("source") != source
            or original.get("destination") != destination
            or original.get("amount") != amount
        ):
            raise ValueError("Undo does not match the original transfer.")

        if transaction_id not in new_wallet.undoable_transactions:
            raise ValueError("Transfer undo is unavailable.")

        if transaction_id in new_wallet.reversed_transactions:
            raise ValueError("Transfer has already been reversed.")

        if new_wallet.get_balance(destination) < amount:
            raise ValueError(
                "Insufficient destination funds to undo transfer."
            )

        new_wallet.get_balance(source)
        new_wallet.accounts[source] += amount
        new_wallet.accounts[destination] -= amount
        new_wallet.undoable_transactions.remove(transaction_id)
        new_wallet.reversed_transactions.add(transaction_id)

        _append_transaction(
            new_wallet,
            {
                "type": "transfer_reversal",
                "original_transaction_id": transaction_id,
                "source": destination,
                "destination": source,
                "amount": amount,
            },
        )

    elif undo_type == "undo_transfer_reversal":
        raise ValueError(
            "A recorded transfer reversal cannot be erased by undo."
        )

    else:
        raise ValueError(f"Unsupported undo operation: {undo_type!r}")

    # Consume account undo metadata only after the operation succeeds.
    if undo_type in ACCOUNT_UNDO_TYPES:
        new_wallet.pending_account_undos.remove(undo)

    new_wallet.validate()

    return new_wallet, {
        "undo": dict(undo),
        "changed": True,
        "after": new_wallet.to_dict(),
    }


def _require_account_id(
    action: dict[str, Any],
    field: str,
) -> str:
    account_id = action.get(field)

    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError(f"{field!r} must be a non-empty string.")

    return account_id


def _require_amount(
    value: Any,
    *,
    allow_zero: bool = False,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Amount must be integer paise.")

    minimum = 0 if allow_zero else 1

    if value < minimum:
        raise ValueError(
            f"Amount must be {'nonnegative' if allow_zero else 'positive'}."
        )

    return value


def _append_transaction(
    wallet: WalletState,
    fields: dict[str, Any],
) -> dict[str, Any]:
    transaction_id = f"T{wallet.next_transaction_id}"
    wallet.next_transaction_id += 1

    transaction = {
        "transaction_id": transaction_id,
        **fields,
    }

    wallet.transactions.append(transaction)
    return transaction


def _find_transaction(
    wallet: WalletState,
    transaction_id: str,
) -> dict[str, Any]:
    if not isinstance(transaction_id, str):
        raise ValueError("Transaction ID must be a string.")

    for transaction in wallet.transactions:
        if transaction["transaction_id"] == transaction_id:
            return transaction

    raise KeyError(f"Unknown transaction ID: {transaction_id}")
