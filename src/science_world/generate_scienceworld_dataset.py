
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Any

from scienceworld import ScienceWorldEnv


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "scienceworld"
OUTPUT_FILE = OUTPUT_DIR / "scienceworld_recovery_expanded_v1.jsonl"

# Broader task coverage for the first dataset expansion.
# If a task name is unsupported by the installed ScienceWorld version,
# main() catches the error for that task and continues with the next one.
TASK_NAMES = [
    "boil",
    "melt",
    "freeze",
    "change-the-state-of-matter",
    "power-component",
    "power-component-more-complex",
    "test-conductivity",
    "test-conductivity-of-unknown-substances",
    "use-thermometer",
    "measure-melting-point-known-substance",
    "measure-melting-point-unknown-substance",
    "chemistry-mix",
    "chemistry-mix-paint",
    "mix-paint",
    "grow-plant",
    "find-plant",
    "find-animal",
    "identify-life-stages-1",
    "identify-life-stages-2",
    "mendelian-genetics",
    "mendelian-genetics-unknown-plant",
    "inclined-plane",
    "lifespan",
    "weigh",
]

TASK_VARIATION = 0
ENV_STEP_LIMIT = 100

BINARY_HORIZON = 3
TRINARY_HORIZON = 5

MAX_EXPANSIONS_PER_EXAMPLE = 30
MAX_BRANCHING_ACTIONS = 10

# Sample more positions across each task's gold trajectory.
SAMPLE_POSITIONS = [2, 5, 8, 11, 14, 17, 20, 25]
SEED = 42

BASIC_ACTIONS = [
    "look around",
    "inventory",
    "task",
    "wait",
    "wait1",
]

MOVEMENT_PREFIXES = (
    "pick up ",
    "put down ",
    "open ",
    "close ",
)


# =========================================================
# ENVIRONMENT HELPERS
# =========================================================

def open_environment(
    task_name: str,
    variation: int = TASK_VARIATION,
) -> ScienceWorldEnv:
    env = ScienceWorldEnv(
        taskName=task_name,
        serverPath=None,
        envStepLimit=ENV_STEP_LIMIT,
    )
    env.load(task_name, variation, "", generateGoldPath=True)
    env.reset()
    return env


def close_environment(env: ScienceWorldEnv | None) -> None:
    if env is not None and hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass


def get_gold_sequence(env: ScienceWorldEnv) -> list[str]:
    if hasattr(env, "get_gold_action_sequence"):
        actions = env.get_gold_action_sequence()
    elif hasattr(env, "getGoldActionSequence"):
        actions = env.getGoldActionSequence()
    else:
        raise AttributeError("ScienceWorld gold action sequence API not found.")

    return list(actions or [])


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    if isinstance(value, set):
        return sorted(
            (to_jsonable(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )

    return str(value)


def get_world_state(env: ScienceWorldEnv) -> Any:
    if hasattr(env, "getObjectTree"):
        return to_jsonable(env.getObjectTree())

    if hasattr(env, "get_object_tree"):
        return to_jsonable(env.get_object_tree())

    raise AttributeError(
        "No getObjectTree() or get_object_tree() method found."
    )


# =========================================================
# STATE COMPARISON
# =========================================================

def states_equivalent(
    left: Any,
    right: Any,
    float_tolerance: float = 0.05,
) -> bool:
    """Compare states recursively with a small numeric tolerance."""

    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False

        return all(
            states_equivalent(left[key], right[key], float_tolerance)
            for key in left
        )

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False

        return all(
            states_equivalent(a, b, float_tolerance)
            for a, b in zip(left, right)
        )

    # Booleans must be compared exactly because bool subclasses int.
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) or isinstance(right, float):
            return abs(left - right) <= float_tolerance

    return left == right


def without_temperature_fields(value: Any) -> Any:
    """Ignore temperature fields when checking movement-only actions."""

    if isinstance(value, dict):
        return {
            key: without_temperature_fields(val)
            for key, val in value.items()
            if "temperature" not in str(key).replace("-", "_").lower()
        }

    if isinstance(value, list):
        return [without_temperature_fields(item) for item in value]

    if isinstance(value, tuple):
        return tuple(without_temperature_fields(item) for item in value)

    return value


def recovery_states_match(
    current: Any,
    target: Any,
    candidate_action: str,
) -> bool:
    action = " ".join(candidate_action.lower().split())

    if action.startswith(MOVEMENT_PREFIXES):
        current = without_temperature_fields(current)
        target = without_temperature_fields(target)

    return states_equivalent(current, target)


# =========================================================
# STATE RECONSTRUCTION
# =========================================================

def unpack_step(result: Any) -> tuple[bool, Any]:
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        raise TypeError(
            "Unexpected env.step() result. Expected at least "
            "(observation, reward, done)."
        )

    done = bool(result[2])
    info = result[3] if len(result) > 3 else None
    return done, info


def replay_history(
    env: ScienceWorldEnv,
    task_name: str,
    variation: int,
    history: list[str],
) -> dict[str, Any]:
    """Reconstruct a state by reloading and replaying an action history."""

    env.load(task_name, variation, "", generateGoldPath=True)
    env.reset()

    executed = 0
    done = False
    error = None

    for action in history:
        if done:
            break

        try:
            result = env.step(action)
            done, _ = unpack_step(result)
            executed += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break

    return {
        "state": get_world_state(env),
        "executed": executed,
        "done": done,
        "error": error,
        "complete": error is None and executed == len(history),
    }


# =========================================================
# ACTION CANDIDATES
# =========================================================

def build_action_pool(gold_actions: list[str]) -> list[str]:
    ordered = []
    seen = set()

    for action in BASIC_ACTIONS + gold_actions:
        action = " ".join(str(action).split())

        if action and action not in seen:
            seen.add(action)
            ordered.append(action)

    return ordered


def get_targeted_recovery_actions(
    candidate_action: str,
    gold_actions: list[str],
) -> list[str]:
    """Generate likely inverse actions, then other grounded actions."""

    action = " ".join(candidate_action.split())
    targeted = []

    if action.startswith("pick up "):
        obj = action[len("pick up "):]
        targeted.append(f"put down {obj}")
    
    elif action.startswith("put down "):
        obj = action[len("put down "):]
        targeted.append(f"pick up {obj}")
    
    elif action.startswith("move "):
        # Moving an object to a named destination may be reversible
        # by moving it back to its previous destination.
        targeted.append("pick up " + action[len("move "):].split(" to ")[0])

    elif action.startswith("activate "):
        targeted.append(f"deactivate {action[len('activate '):]}")

    elif action.startswith("deactivate "):
        targeted.append(f"activate {action[len('deactivate '):]}")

    elif action.startswith("open "):
        targeted.append(f"close {action[len('open '):]}")

    elif action.startswith("close "):
        targeted.append(f"open {action[len('close '):]}")

    ordered = []
    seen = set()

    for candidate in targeted + list(reversed(gold_actions)):
        candidate = " ".join(candidate.split())

        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)

    return ordered


def select_branching_actions(
    action_pool: list[str],
    prefix: list[str],
    width: int = MAX_BRANCHING_ACTIONS,
) -> list[str]:
    return action_pool[:width]


# =========================================================
# RECOVERY SEARCH
# =========================================================


def find_object_locations(state: Any, object_name: str) -> list[str]:
    """Find paths containing a named object without printing the full state."""
    locations = []

    def walk(node: Any, path: str = "root") -> None:
        if isinstance(node, dict):
            name = str(node.get("name", "")).lower()

            if name == object_name.lower():
                locations.append(path)

            for key, value in node.items():
                walk(value, f"{path} -> {key}")

        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(state)
    return locations

def search_recovery(
    env: ScienceWorldEnv,
    task_name: str,
    variation: int,
    prefix: list[str],
    target_state: Any,
    action_pool: list[str],
) -> dict[str, Any]:
    """
    Bounded breadth-first search for a verified recovery path.

    The target is the world state immediately before the candidate
    action. The prefix already includes the candidate action.

    A recovery must contain at least one action and must restore the
    target state according to recovery_states_match().
    """

    start = replay_history(
        env, task_name, variation, prefix
    )

    if not start["complete"]:
        return {
            "status": "inconclusive",
            "reason": "start_state_reconstruction_failed",
            "recovery_actions": None,
            "recovery_cost": None,
            "expanded_nodes": 0,
        }

    candidate_action = prefix[-1] if prefix else ""
    is_movement = candidate_action.lower().startswith(
        MOVEMENT_PREFIXES
    )

    # Prevent movement actions from being incorrectly labelled
    # as zero-cost recoveries.
    if (
        not is_movement
        and recovery_states_match(
            start["state"],
            target_state,
            candidate_action,
        )
    ):
        return {
            "status": "recovered",
            "reason": "state_already_restored",
            "recovery_actions": [],
            "recovery_cost": 0,
            "expanded_nodes": 0,
        }

    if start["done"]:
        return {
            "status": "not_evaluable",
            "reason": "task_terminated_before_recovery",
            "recovery_actions": None,
            "recovery_cost": None,
            "expanded_nodes": 0,
        }

    gold_actions = [
        action
        for action in action_pool
        if action not in BASIC_ACTIONS
    ]

    targeted = get_targeted_recovery_actions(
        candidate_action,
        gold_actions,
    )

    # For movement actions, first test the likely inverse.
    # Other actions can use the broader candidate pool.
    if is_movement:
        actions_to_try = targeted + action_pool
    else:
        actions_to_try = targeted + action_pool

    ordered_pool = []
    seen_actions = set()

    for action in actions_to_try:
        if action not in seen_actions:
            seen_actions.add(action)
            ordered_pool.append(action)

    queue = deque([[]])
    seen_paths = {()}
    expanded = 0

    while queue and expanded < MAX_EXPANSIONS_PER_EXAMPLE:
        recovery_path = queue.popleft()

        if len(recovery_path) >= TRINARY_HORIZON:
            continue

        node_history = prefix + recovery_path

        node = replay_history(
            env,
            task_name,
            variation,
            node_history,
        )

        expanded += 1

        if not node["complete"]:
            continue

        # Compact debugging for the metal-pot inverse action.
        if (
            candidate_action == "pick up metal pot"
            and recovery_path == ["put down metal pot"]
        ):
            print(
                "Target pot location:",
                find_object_locations(
                    target_state, "metal pot"
                ),
            )
            print(
                "Current pot location:",
                find_object_locations(
                    node["state"], "metal pot"
                ),
            )
            print(
                "States match:",
                recovery_states_match(
                    node["state"],
                    target_state,
                    candidate_action,
                ),
            )

        # Never accept an empty recovery path.
        if recovery_path and recovery_states_match(
            node["state"],
            target_state,
            candidate_action,
        ):
            verified = replay_history(
                env,
                task_name,
                variation,
                node_history,
            )

            if (
                verified["complete"]
                and recovery_states_match(
                    verified["state"],
                    target_state,
                    candidate_action,
                )
            ):
                return {
                    "status": "recovered",
                    "reason": "verified_state_match",
                    "recovery_actions": recovery_path,
                    "recovery_cost": len(recovery_path),
                    "expanded_nodes": expanded,
                }

        if node["done"]:
            continue

        candidates = select_branching_actions(
            ordered_pool,
            node_history,
            width=MAX_BRANCHING_ACTIONS,
        )

        for action in candidates:
            new_path = recovery_path + [action]
            key = tuple(new_path)

            if key not in seen_paths:
                seen_paths.add(key)
                queue.append(new_path)

    return {
        "status": "inconclusive",
        "reason": (
            "expansion_limit_reached"
            if expanded >= MAX_EXPANSIONS_PER_EXAMPLE
            else "no_path_found_within_bounded_search"
        ),
        "recovery_actions": None,
        "recovery_cost": None,
        "expanded_nodes": expanded,
    }

# =========================================================
# LABELING
# =========================================================

def assign_labels(search_result: dict[str, Any]) -> dict[str, Any]:
    status = search_result["status"]

    if status == "not_evaluable":
        return {
            "binary_label": "not_evaluable",
            "trinary_label": "not_evaluable",
            "label_status": "unresolved",
        }

    if status != "recovered":
        return {
            "binary_label": "inconclusive",
            "trinary_label": "inconclusive",
            "label_status": "unresolved",
        }

    cost = search_result["recovery_cost"]

    binary = (
        "reversible"
        if cost <= BINARY_HORIZON
        else "irreversible"
    )

    trinary = (
        "reversible"
        if cost <= BINARY_HORIZON
        else "partially_reversible"
    )

    return {
        "binary_label": binary,
        "trinary_label": trinary,
        "label_status": "verified",
    }


# =========================================================
# EXPERIMENT
# =========================================================

def run_candidate_experiment(
    env: ScienceWorldEnv,
    task_name: str,
    variation: int,
    gold_actions: list[str],
    action_index: int,
    action_pool: list[str],
) -> dict[str, Any]:
    history_before = gold_actions[:action_index]
    candidate_action = gold_actions[action_index]

    before = replay_history(
        env, task_name, variation, history_before
    )

    base = {
        "id": f"scienceworld:{task_name}:v{variation}:a{action_index}",
        "scenario_id": f"{task_name}:variation:{variation}",
        "environment": "scienceworld",
        "source": "gold_action_sequence",
        "horizon": BINARY_HORIZON,
        "task_name": task_name,
        "variation": variation,
        "action_index": action_index,
        "history_before": history_before,
        "candidate_action": candidate_action,
    }

    if not before["complete"]:
        return {
            **base,
            "target_state_available": False,
            "candidate_executed": False,
            "search_status": "inconclusive",
            "search_reason": "pre_action_reconstruction_failed",
            "recovery_actions": None,
            "recovery_cost": None,
            **assign_labels({"status": "inconclusive"}),
        }

    if before["done"]:
        return {
            **base,
            "target_state_available": True,
            "candidate_executed": False,
            "search_status": "not_evaluable",
            "search_reason": "task_already_terminated",
            "recovery_actions": None,
            "recovery_cost": None,
            **assign_labels({"status": "not_evaluable"}),
        }

    target_state = before["state"]
    candidate_history = history_before + [candidate_action]

    after = replay_history(
        env, task_name, variation, candidate_history
    )

    if not after["complete"]:
        return {
            **base,
            "target_state_available": True,
            "candidate_executed": False,
            "search_status": "inconclusive",
            "search_reason": "candidate_execution_failed",
            "candidate_error": after["error"],
            "recovery_actions": None,
            "recovery_cost": None,
            **assign_labels({"status": "inconclusive"}),
        }

    search_result = search_recovery(
        env=env,
        task_name=task_name,
        variation=variation,
        prefix=candidate_history,
        target_state=target_state,
        action_pool=action_pool,
    )

    return {
        **base,
        "target_state_available": True,
        "candidate_executed": True,
        "candidate_done": after["done"],
        "pre_state": target_state,
        "post_state": after["state"],
        "search_status": search_result["status"],
        "search_reason": search_result["reason"],
        "recovery_actions": search_result["recovery_actions"],
        "recovery_cost": search_result["recovery_cost"],
        "expanded_nodes": search_result["expanded_nodes"],
        "binary_horizon": BINARY_HORIZON,
        "trinary_search_horizon": TRINARY_HORIZON,
        "state_comparison": (
            "temperature-insensitive for pick-up/drop/open/close; "
            "full tolerant tree comparison otherwise"
        ),
        **assign_labels(search_result),
    }


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    for task_name in TASK_NAMES:
        print(f"\n{'=' * 65}")
        print(f"Task: {task_name}")
        print(f"{'=' * 65}")

        env = None

        try:
            env = open_environment(task_name, TASK_VARIATION)
            gold_actions = get_gold_sequence(env)
            action_pool = build_action_pool(gold_actions)

            print(f"Gold actions: {len(gold_actions)}")
            print(f"Unique action candidates: {len(action_pool)}")

            for action_index in SAMPLE_POSITIONS:
                if action_index >= len(gold_actions):
                    print(
                        f"Skipping index {action_index}: "
                        "outside gold sequence."
                    )
                    continue

                print(
                    f"\nTesting action {action_index}: "
                    f"{gold_actions[action_index]!r}"
                )

                record = run_candidate_experiment(
                    env=env,
                    task_name=task_name,
                    variation=TASK_VARIATION,
                    gold_actions=gold_actions,
                    action_index=action_index,
                    action_pool=action_pool,
                )

                records.append(record)

                print(
                    f"  status={record['search_status']}, "
                    f"cost={record['recovery_cost']}, "
                    f"binary={record['binary_label']}, "
                    f"trinary={record['trinary_label']}"
                )

        except Exception as exc:
            print(
                f"ERROR in task {task_name}: "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            close_environment(env)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output_file": str(OUTPUT_FILE),
        "examples": len(records),
        "search_statuses": {},
        "binary_labels": {},
        "trinary_labels": {},
    }

    for record in records:
        for field, summary_field in [
            ("search_status", "search_statuses"),
            ("binary_label", "binary_labels"),
            ("trinary_label", "trinary_labels"),
        ]:
            value = record.get(field, "missing")
            summary[summary_field][value] = (
                summary[summary_field].get(value, 0) + 1
            )

    print(f"\n{'=' * 65}")
    print("EXPANDED DATASET SUMMARY")
    print(f"{'=' * 65}")
    print(json.dumps(summary, indent=2))
    print(f"\nDataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
