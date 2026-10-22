
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


EmailStatus = Literal["draft", "queued", "sent", "deleted", "recovering"]
DeliveryStatus = Literal["pending", "delivered", "failed", "unknown"]


@dataclass
class Email:
    """Represents one email and its lifecycle."""

    email_id: str
    sender: str
    recipients: list[str]
    subject: str
    body: str

    status: EmailStatus = "draft"
    delivery_status: DeliveryStatus = "pending"

    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)

    # Retained locally for audit and recovery experiments.
    previous_versions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EmailOutboxState:
    """Complete local state of the email outbox sandbox."""

    emails: dict[str, Email] = field(default_factory=dict)

    # Emails that have been externally sent are recorded separately
    # from the editable local outbox.
    sent_history: list[dict[str, Any]] = field(default_factory=list)

    # Recovery resources are explicit experimental conditions.
    undo_history: list[dict[str, Any]] = field(default_factory=list)
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_email(self, email: Email) -> None:
        """Add an email, rejecting duplicate identifiers."""
        if email.email_id in self.emails:
            raise ValueError(
                f"Email already exists: {email.email_id}"
            )

        self.emails[email.email_id] = deepcopy(email)

    def get_email(self, email_id: str) -> Email:
        """Return an email or raise a descriptive error."""
        try:
            return self.emails[email_id]
        except KeyError as exc:
            raise KeyError(
                f"Email not found: {email_id}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete state into JSON-compatible data."""
        return deepcopy(asdict(self))

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> EmailOutboxState:
        """Reconstruct state from serialized data."""
        emails = {
            email_id: Email(**email_data)
            for email_id, email_data in data.get(
                "emails", {}
            ).items()
        }

        return cls(
            emails=emails,
            sent_history=deepcopy(
                data.get("sent_history", [])
            ),
            undo_history=deepcopy(
                data.get("undo_history", [])
            ),
            snapshots=deepcopy(
                data.get("snapshots", {})
            ),
        )

    def clone(self) -> EmailOutboxState:
        """Return an independent copy of the complete state."""
        return deepcopy(self)

    def local_state_dict(self) -> dict[str, Any]:
        """Return state that can be restored locally.

        External send history is deliberately excluded from
        this representation so restoring a local snapshot
        cannot erase evidence of an already-sent email.
        """
        return {
            "emails": deepcopy(asdict(self)["emails"]),
            "undo_history": deepcopy(self.undo_history),
            "snapshots": deepcopy(self.snapshots),
        }
