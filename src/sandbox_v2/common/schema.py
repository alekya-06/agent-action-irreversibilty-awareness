"""
Shared record schema for every testbed (sandbox, ScienceWorld, ToolEmu).

Required fields (Section 4.3 of the brief asks for id, environment,
state_text, action_text, label, horizon, source; the rest make the
records comparable and splittable):

    id                unique example id
    environment       "filesystem" | "email_outbox" | "wallet" | "scienceworld" | "toolemu"
    scenario_id       split key: every row with the same scenario_id goes to
                      the same split, so minimal-pair variants never leak
    group_id          minimal-pair group (variants that differ only in condition)
    condition         recovery-resource condition, e.g. "none", "undo",
                      "snapshot", "undo+snapshot"
    state_text        what the model sees about the current state, including
                      which recovery resources exist
    action_text       the candidate action as a tool call
    action_unit       "single" or "sequence"
    label             label at horizon 3
    binary_label      "reversible" | "irreversible" | None (partial excluded)
    labels_by_horizon {"1": ..., "2": ..., "3": ...}
    horizon           3
    extended_horizon  depth searched to decide partial vs irreversible
    exact_cost        cheapest exact restoration found, or None
    approx_cost       cheapest approximate restoration found, or None
    exhausted_depth   every path up to this length was examined
    label_method      "search" | "rule" | "design"
    source            how the example was produced

Environment-specific details go under "details".
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from labels import ALL_LABELS, HORIZONS, derive_labels

ENVIRONMENTS = {
    "filesystem", "email_outbox", "wallet", "scienceworld", "toolemu",
}
LABEL_METHODS = {"search", "rule", "design"}
ACTION_UNITS = {"single", "sequence"}

REQUIRED_FIELDS = (
    "id", "environment", "scenario_id", "group_id", "condition",
    "state_text", "action_text", "action_unit",
    "label", "binary_label", "labels_by_horizon", "horizon",
    "extended_horizon", "exact_cost", "approx_cost", "exhausted_depth",
    "label_method", "source",
)


def make_record(
    *,
    id: str,
    environment: str,
    scenario_id: str,
    group_id: str,
    condition: str,
    state_text: str,
    action_text: str,
    labels: dict[str, Any],
    source: str,
    label_method: str = "search",
    action_unit: str = "single",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "id": id,
        "environment": environment,
        "scenario_id": scenario_id,
        "group_id": group_id,
        "condition": condition,
        "state_text": state_text,
        "action_text": action_text,
        "action_unit": action_unit,
        **labels,
        "label_method": label_method,
        "source": source,
        "details": details or {},
    }
    validate_record(record)
    return record


def validate_record(record: dict[str, Any]) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"{record.get('id')}: missing fields {missing}")

    rid = record["id"]
    if record["environment"] not in ENVIRONMENTS:
        raise ValueError(f"{rid}: unknown environment")
    if record["label_method"] not in LABEL_METHODS:
        raise ValueError(f"{rid}: unknown label_method")
    if record["action_unit"] not in ACTION_UNITS:
        raise ValueError(f"{rid}: unknown action_unit")
    for field in ("state_text", "action_text", "scenario_id", "group_id"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{rid}: {field} must be a non-empty string")
    if record["label"] not in ALL_LABELS:
        raise ValueError(f"{rid}: invalid label {record['label']!r}")
    if set(record["labels_by_horizon"]) != {str(k) for k in HORIZONS}:
        raise ValueError(f"{rid}: labels_by_horizon must cover {HORIZONS}")

    if record["label_method"] == "search":
        # Labels must follow from the stored search results.
        expected = derive_labels(
            record["exact_cost"],
            record["approx_cost"],
            record["exhausted_depth"],
            record["extended_horizon"],
        )
        for field in ("label", "binary_label", "labels_by_horizon"):
            if record[field] != expected[field]:
                raise ValueError(
                    f"{rid}: {field}={record[field]!r} does not follow "
                    f"from the search results (expected {expected[field]!r})"
                )


def check_dataset(records: list[dict[str, Any]]) -> None:
    ids = [r["id"] for r in records]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise ValueError(f"Duplicate ids: {dupes[:5]}")
    for record in records:
        validate_record(record)


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    def count(fn) -> dict[str, int]:
        return dict(sorted(Counter(str(fn(r)) for r in records).items()))

    summary: dict[str, Any] = {
        "examples": len(records),
        "scenarios": len({r["scenario_id"] for r in records}),
        "groups": len({r["group_id"] for r in records}),
        "environment": count(lambda r: r["environment"]),
        "condition": count(lambda r: r["condition"]),
        "action_unit": count(lambda r: r["action_unit"]),
        "label_method": count(lambda r: r["label_method"]),
        "binary_label": count(lambda r: r["binary_label"]),
    }
    for k in HORIZONS:
        summary[f"label_k{k}"] = count(
            lambda r, k=k: r["labels_by_horizon"][str(k)]
        )

    # Groups whose variants do not all share one label are the minimal
    # pairs that matter: same action, reversibility changed by state.
    by_group: dict[str, set[str]] = {}
    for r in records:
        by_group.setdefault(r["group_id"], set()).add(r["label"])
    contrastive = {g for g, labels in by_group.items() if len(labels) > 1}
    summary["contrastive_groups"] = len(contrastive)
    summary["rows_in_contrastive_groups"] = sum(
        r["group_id"] in contrastive for r in records
    )
    return summary
