import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labelling import label_actions  # noqa: E402
from state import FileSystemState  # noqa: E402


def make_fs():
    return FileSystemState(
        directories={"/", "/reports", "/archive"},
        files={"/reports/a.csv": "x", "/archive/i.csv": "y"},
    )


def label(actions, condition):
    return label_actions("t", make_fs(), actions, condition)["labels"]


DELETE = [{"type": "delete_file", "path": "/reports/a.csv"}]
RENAME = [{"type": "rename", "source": "/reports/a.csv", "new_name": "b.csv"}]


def test_delete_is_a_minimal_pair_over_conditions():
    assert label(DELETE, "none")["label"] == "irreversible"
    for condition in ("undo", "snapshot", "undo+snapshot"):
        assert label(DELETE, condition)["label"] == "reversible"


def test_rename_is_reversible_without_resources():
    labels = label(RENAME, "none")
    assert labels["label"] == "reversible" and labels["exact_cost"] == 1


def test_long_chain_without_resources_is_partial_at_k3():
    chain = [
        {"type": "rename", "source": f"/reports/{a}", "new_name": b}
        for a, b in [("a.csv", "s1"), ("s1", "s2"), ("s2", "s3"), ("s3", "s4")]
    ]
    labels = label(chain, "none")
    assert labels["exact_cost"] == 4
    assert labels["label"] == "partially_reversible"
    assert label(chain, "snapshot")["label"] == "reversible"
