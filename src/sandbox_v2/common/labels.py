"""
Shared reversibility labelling rule for every testbed.

Definition (agreed for the project, following Section 4.2 of the brief,
"restored only approximately, or at a cost"). For a horizon k:

    reversible            the exact previous state can be restored in <= k actions
    partially_reversible  not reversible at k, but a restoration exists within the
                          extended search horizon: either an exact restoration that
                          needs more than k actions, or an approximate restoration
                          (e.g. balances restored but the ledger keeps a trace)
    irreversible          no exact or approximate restoration within the extended
                          horizon (the search was exhaustive)
    inconclusive          the search stopped before it could decide

One breadth-first search records the cheapest exact and the cheapest approximate
restoration, so labels for every horizon come from the same run. Consequence of
the definition: irreversible at k=1 is irreversible at every k. The horizon only
moves examples between reversible and partially_reversible.

Each environment supplies:
    successors(state) -> iterable of (action, next_state)
    is_exact(state)   -> bool
    is_approx(state)  -> bool   (must be True whenever is_exact is True)
    key(state)        -> hashable state identity used to skip repeated states
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable

HORIZONS = (1, 2, 3)
MAIN_HORIZON = 3
EXTENDED_HORIZON = 10

LABELS = ("reversible", "partially_reversible", "irreversible")
ALL_LABELS = LABELS + ("inconclusive",)


@dataclass
class SearchOutcome:
    exact_cost: int | None = None
    exact_path: list[Any] | None = None
    approx_cost: int | None = None
    approx_path: list[Any] | None = None
    exhausted_depth: int = 0      # every path of this length was examined
    max_depth: int = EXTENDED_HORIZON
    nodes: int = 0


def dual_bfs(
    start: Any,
    successors: Callable[[Any], Iterable[tuple[Any, Any]]],
    is_exact: Callable[[Any], bool],
    is_approx: Callable[[Any], bool],
    key: Callable[[Any], Hashable],
    max_depth: int = EXTENDED_HORIZON,
) -> SearchOutcome:
    """
    Level-by-level BFS from the post-action state toward the pre-action state.
    Stops at the first exact restoration; records the first approximate one.
    """
    out = SearchOutcome(max_depth=max_depth)

    if is_exact(start):
        out.exact_cost = out.approx_cost = 0
        out.exact_path, out.approx_path = [], []
        out.exhausted_depth = max_depth
        return out

    if is_approx(start):
        out.approx_cost, out.approx_path = 0, []

    frontier = [(start, [])]
    seen = {key(start)}

    for depth in range(1, max_depth + 1):
        next_frontier = []

        for state, path in frontier:
            for action, child in successors(state):
                out.nodes += 1
                child_path = path + [action]

                if is_exact(child):
                    out.exact_cost, out.exact_path = depth, child_path
                    if out.approx_cost is None:
                        out.approx_cost, out.approx_path = depth, child_path
                    out.exhausted_depth = depth - 1
                    return out

                if out.approx_cost is None and is_approx(child):
                    out.approx_cost, out.approx_path = depth, child_path

                child_key = key(child)
                if child_key in seen:
                    continue
                seen.add(child_key)
                next_frontier.append((child, child_path))

        out.exhausted_depth = depth
        if not next_frontier:
            # Nothing left to expand: deeper levels are empty.
            out.exhausted_depth = max_depth
            break
        frontier = next_frontier

    return out


def label_at_horizon(
    k: int,
    exact_cost: int | None,
    approx_cost: int | None,
    exhausted_depth: int,
    max_depth: int,
) -> str:
    """Apply the shared definition at horizon k."""
    if exact_cost is not None and exact_cost <= k:
        return "reversible"

    if exact_cost is not None:
        # BFS found the exact path at depth > k after finishing every
        # shallower level, so no exact path within k was missed.
        return "partially_reversible"

    if approx_cost is not None:
        # An exact path within k could only have been missed if depth k
        # was not fully searched.
        return (
            "partially_reversible" if exhausted_depth >= k
            else "inconclusive"
        )

    return "irreversible" if exhausted_depth >= max_depth else "inconclusive"


def binary_from_label(label: str) -> str | None:
    """Binary task (reversible vs irreversible); partial cases are excluded."""
    return label if label in ("reversible", "irreversible") else None


def derive_labels(
    exact_cost: int | None,
    approx_cost: int | None,
    exhausted_depth: int,
    max_depth: int,
) -> dict[str, Any]:
    """All label fields of the shared schema, computed from search results."""
    by_k = {
        str(k): label_at_horizon(
            k, exact_cost, approx_cost, exhausted_depth, max_depth
        )
        for k in HORIZONS
    }
    main = by_k[str(MAIN_HORIZON)]
    return {
        "label": main,
        "binary_label": binary_from_label(main),
        "labels_by_horizon": by_k,
        "horizon": MAIN_HORIZON,
        "extended_horizon": max_depth,
        "exact_cost": exact_cost,
        "approx_cost": approx_cost,
        "exhausted_depth": exhausted_depth,
    }


def labels_from_outcome(outcome: SearchOutcome) -> dict[str, Any]:
    return derive_labels(
        outcome.exact_cost,
        outcome.approx_cost,
        outcome.exhausted_depth,
        outcome.max_depth,
    )
