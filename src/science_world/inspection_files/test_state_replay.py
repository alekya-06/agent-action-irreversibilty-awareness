
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from scienceworld import ScienceWorldEnv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TASK_NAME = "melt"
VARIATION = 0
SIMPLIFICATIONS = ""
STEP_LIMIT = 100

ACTION_HISTORY = [
    "open door to kitchen",
    "go to kitchen",
]

OUTPUT_DIR = Path("src/sandbox/science_world/state_replay_results")
OUTPUT_FILE = OUTPUT_DIR / "replay_diagnostic.json"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def canonicalize(value: Any) -> Any:
    """Sort dictionary keys for consistent comparison."""
    if isinstance(value, dict):
        return {
            key: canonicalize(value[key])
            for key in sorted(value)
        }
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def capture_state(env: ScienceWorldEnv) -> dict[str, Any]:
    """Capture the observable state available through the API."""
    return {
        "object_tree": canonicalize(env.getObjectTree()),
        "task_progress": env.get_goal_progress(),
        "inventory": env.inventory(),
        "agent_location": env.look(),
        "num_moves": env.get_num_moves(),
    }


def run_history(
    env: ScienceWorldEnv,
    actions: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reset and replay the same action sequence."""
    env.reset()
    history = []

    for index, action in enumerate(actions, start=1):
        observation, reward, done, info = env.step(action)

        history.append({
            "step": index,
            "action": action,
            "observation": observation,
            "reward": reward,
            "done": done,
            "score": info.get("score"),
            "goal_progress": env.get_goal_progress(),
        })

        print(
            f"Step {index}: {action!r} | "
            f"reward={reward} | "
            f"score={info.get('score')} | done={done}"
        )

        if done and index < len(actions):
            raise RuntimeError(
                f"Episode ended at step {index}, "
                "before all actions were replayed."
            )

    return capture_state(env), history


def compare_values(
    a: Any,
    b: Any,
    path: str = "$",
    differences: list | None = None,
) -> list:
    """Return all differences, preserving their full paths."""
    if differences is None:
        differences = []

    if type(a) is not type(b):
        differences.append((path, a, b, "type"))
        return differences

    if isinstance(a, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in a:
                differences.append(
                    (f"{path}.{key}", None, b[key], "missing")
                )
            elif key not in b:
                differences.append(
                    (f"{path}.{key}", a[key], None, "missing")
                )
            else:
                compare_values(
                    a[key], b[key], f"{path}.{key}", differences
                )

    elif isinstance(a, list):
        if len(a) != len(b):
            differences.append(
                (f"{path}.length", len(a), len(b), "length")
            )
        for i, (x, y) in enumerate(zip(a, b)):
            compare_values(x, y, f"{path}[{i}]", differences)

    elif a != b:
        is_numeric = (
            isinstance(a, (int, float))
            and not isinstance(a, bool)
            and isinstance(b, (int, float))
            and not isinstance(b, bool)
        )
        differences.append((
            path, a, b, "numeric" if is_numeric else "value"
        ))

    return differences


def get_temperature_differences(differences: list) -> list:
    return [
        (path, a, b, abs(a - b))
        for path, a, b, kind in differences
        if (
            kind == "numeric"
            and "temperatureC" in path
            and isinstance(a, (int, float))
            and isinstance(b, (int, float))
        )
    ]


def get_non_temperature_differences(differences: list) -> list:
    return [
        item for item in differences
        if "temperatureC" not in item[0]
    ]


# --------------------------------------------------
# Main diagnostic
# --------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = ScienceWorldEnv("", envStepLimit=STEP_LIMIT)

    try:
        env.load(TASK_NAME, VARIATION, SIMPLIFICATIONS)

        print("\n=== Replay 1 ===")
        state_1, history_1 = run_history(env, ACTION_HISTORY)

        print("\n=== Replay 2 ===")
        state_2, history_2 = run_history(env, ACTION_HISTORY)

        tree_diffs = compare_values(
            state_1["object_tree"],
            state_2["object_tree"],
        )

        all_diffs = compare_values(state_1, state_2)

        temp_diffs = get_temperature_differences(tree_diffs)
        non_temp_diffs = get_non_temperature_differences(tree_diffs)

        max_temp_diff = max(
            (item[3] for item in temp_diffs),
            default=0.0,
        )

        print("\n=== REPLAY DIAGNOSTIC ===")
        print("Object trees exactly identical:", not tree_diffs)
        print("Total object-tree differences:", len(tree_diffs))
        print("Temperature differences:", len(temp_diffs))
        print("Non-temperature differences:", len(non_temp_diffs))
        print(f"Maximum temperature difference: {max_temp_diff:.8f} °C")

        print("\n=== OTHER STATE FIELDS ===")
        for field in (
            "task_progress",
            "inventory",
            "agent_location",
            "num_moves",
        ):
            identical = state_1[field] == state_2[field]
            print(f"{field}: {'MATCH' if identical else 'DIFFER'}")

        print("\n=== DIFFERENCE TYPES ===")
        print(Counter(item[3] for item in all_diffs))

        print("\n=== NON-TEMPERATURE OBJECT-TREE DIFFERENCES ===")
        if not non_temp_diffs:
            print("None found.")
        else:
            for path, a, b, kind in non_temp_diffs[:50]:
                print(f"\n{path} [{kind}]")
                print("  Replay 1:", repr(a)[:200])
                print("  Replay 2:", repr(b)[:200])

        print("\n=== LARGEST TEMPERATURE DIFFERENCES ===")
        for path, a, b, delta in sorted(
            temp_diffs, key=lambda item: item[3], reverse=True
        )[:10]:
            print(f"{path}")
            print(f"  Replay 1: {a}")
            print(f"  Replay 2: {b}")
            print(f"  Absolute difference: {delta:.8f} °C")

        result = {
            "task": TASK_NAME,
            "variation": VARIATION,
            "actions": ACTION_HISTORY,
            "state_1": state_1,
            "state_2": state_2,
            "history_1": history_1,
            "history_2": history_2,
            "object_tree_exact_match": not tree_diffs,
            "object_tree_difference_count": len(tree_diffs),
            "temperature_difference_count": len(temp_diffs),
            "non_temperature_difference_count": len(non_temp_diffs),
            "max_temperature_difference_c": max_temp_diff,
            "field_matches": {
                field: state_1[field] == state_2[field]
                for field in (
                    "task_progress",
                    "inventory",
                    "agent_location",
                    "num_moves",
                )
            },
            "object_tree_differences": [
                {
                    "path": path,
                    "replay_1": a,
                    "replay_2": b,
                    "kind": kind,
                }
                for path, a, b, kind in tree_diffs
            ],
        }

        OUTPUT_FILE.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\nFull diagnostic saved to: {OUTPUT_FILE.resolve()}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
