import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))

from dataset import generate_dataset  # noqa: E402
from schema import check_dataset  # noqa: E402


def test_dataset_follows_the_shared_schema():
    rows = generate_dataset()
    check_dataset(rows)
    assert {r["condition"] for r in rows} == {"none", "undo"}


def test_every_group_has_both_conditions():
    groups = {}
    for row in generate_dataset():
        groups.setdefault(row["group_id"], set()).add(row["condition"])
    assert all(conditions == {"none", "undo"} for conditions in groups.values())
