import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labels import derive_labels, dual_bfs, label_at_horizon  # noqa: E402

EXT = 10


def test_exact_within_horizon_is_reversible():
    assert label_at_horizon(1, 1, 1, 0, EXT) == "reversible"


def test_exact_beyond_horizon_is_partial():
    assert label_at_horizon(1, 2, 2, 1, EXT) == "partially_reversible"
    assert label_at_horizon(3, 4, 4, 3, EXT) == "partially_reversible"


def test_approximate_only_is_partial_at_every_horizon():
    labels = derive_labels(None, 1, EXT, EXT)["labels_by_horizon"]
    assert set(labels.values()) == {"partially_reversible"}


def test_nothing_found_after_exhaustive_search_is_irreversible():
    labels = derive_labels(None, None, EXT, EXT)
    assert labels["label"] == "irreversible"
    assert labels["binary_label"] == "irreversible"


def test_unfinished_search_is_inconclusive():
    assert label_at_horizon(3, None, None, 2, EXT) == "inconclusive"
    assert label_at_horizon(3, None, 1, 2, EXT) == "inconclusive"


def test_partial_is_excluded_from_binary():
    assert derive_labels(4, 4, 3, EXT)["binary_label"] is None


def test_dual_bfs_on_a_line_graph():
    # States 0..5, target 0, start 3, one step left or right.
    def successors(s):
        for nxt in (s - 1, s + 1):
            if 0 <= nxt <= 5:
                yield nxt, nxt
    out = dual_bfs(3, successors, lambda s: s == 0, lambda s: s <= 1,
                   key=lambda s: s, max_depth=EXT)
    assert out.exact_cost == 3
    assert out.approx_cost == 2
    assert out.exhausted_depth == 2


def test_dual_bfs_exhausts_a_finite_graph():
    out = dual_bfs(1, lambda s: iter(()), lambda s: s == 0, lambda s: False,
                   key=lambda s: s, max_depth=EXT)
    assert out.exact_cost is None and out.exhausted_depth == EXT
