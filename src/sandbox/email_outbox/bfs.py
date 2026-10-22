
from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Callable


StateKey = Callable[[Any], Any]


def bfs_recovery(
    initial_state: Any,
    target_state: Any,
    get_actions: Callable[[Any], list[dict[str, Any]]],
    apply_action: Callable[
        [Any, dict[str, Any]], Any
    ],
    state_key: StateKey | None = None,
    max_depth: int = 3,
) -> dict[str, Any]:
    """Find the shortest action sequence restoring target state.

    get_actions(state) returns candidate recovery actions.
    apply_action(state, action) returns the resulting state.

    state_key(state) must represent the state properties that
    need to be restored. By default, states are compared directly.
    """
    if max_depth < 0:
        raise ValueError("max_depth cannot be negative.")

    key = state_key or (lambda state: state)

    target_key = key(target_state)
    start = deepcopy(initial_state)

    if key(start) == target_key:
        return {
            "reversible": True,
            "recovery_cost": 0,
            "actions": [],
            "final_state": start,
        }

    queue = deque([(start, [])])
    visited = {key(start)}

    while queue:
        current, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        for action in get_actions(current):
            try:
                next_state = apply_action(
                    deepcopy(current), deepcopy(action)
                )
            except (ValueError, KeyError, TypeError):
                continue

            next_key = key(next_state)

            if next_key in visited:
                continue

            next_path = path + [deepcopy(action)]

            if next_key == target_key:
                return {
                    "reversible": True,
                    "recovery_cost": len(next_path),
                    "actions": next_path,
                    "final_state": next_state,
                }

            visited.add(next_key)
            queue.append((next_state, next_path))

    return {
        "reversible": False,
        "recovery_cost": "unrecoverable",
        "actions": [],
        "final_state": None,
    }
