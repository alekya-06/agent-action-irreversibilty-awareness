from __future__ import annotations

import json
import math
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scienceworld import ScienceWorldEnv


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "replay_audit_results"
TASK_VARIATION = 0
ENV_STEP_LIMIT = 100

TASKS: dict[str, list[str]] = {
    "melt": [
        "open door to kitchen",
        "go to kitchen",
        "look around",
        "pick up thermometer",
        "open cupboard",
        "pick up metal pot",
        "look around",
        "move metal pot to sink",
        "activate sink",
        "deactivate sink",
        "pick up metal pot",
        "focus on substance in metal pot",
        "pour metal pot into metal pot",
        "pick up metal pot",
        "open freezer",
        "move metal pot to freezer",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "wait",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "wait",
        "examine ice",
        "use thermometer in inventory on ice",
        "wait",
        "pick up metal pot",
        "pick up metal pot",
        "move metal pot to stove",
        "activate stove",
        "examine ice",
        "use thermometer in inventory on ice",
        "examine ice",
        "use thermometer in inventory on substance in metal pot",
        "wait1",
    ],
    "boil": [
        "open door to kitchen",
        "go to kitchen",
        "look around",
        "pick up thermometer",
        "open cupboard",
        "pick up metal pot",
        "look around",
        "move metal pot to sink",
        "activate sink",
        "deactivate sink",
        "pick up metal pot",
        "focus on substance in metal pot",
        "pour metal pot into metal pot",
        "pick up metal pot",
        "move metal pot to stove",
        "activate stove",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine substance in metal pot",
        "use thermometer in inventory on substance in metal pot",
        "examine steam",
        "use thermometer in inventory on steam",
        "wait1",
    ],
    "power-component": [
        "open door to workshop",
        "go to workshop",
        "look around",
        "focus on red light bulb",
        "connect battery anode to black wire terminal 1",
        "connect battery cathode to orange wire terminal 1",
        "connect black wire terminal 2 to cathode in red light bulb",
        "connect orange wire terminal 2 to anode in red light bulb",
        "wait1",
        "wait1",
        "look around",
    ],
}


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------

def to_jsonable(value: Any, depth: int = 0) -> Any:
    """Convert values into JSON-compatible representations."""
    if depth > 20:
        return f"<max-depth:{type(value).__name__}>"

    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)

    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item, depth + 1)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item, depth + 1) for item in value]

    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item(), depth + 1)
        except Exception:
            pass

    return repr(value)


# ---------------------------------------------------------------------
# Actual ScienceWorld world-state capture
# ---------------------------------------------------------------------

def get_world_state(env: ScienceWorldEnv) -> dict[str, Any] | None:
    """
    Capture the complete ScienceWorld object tree.

    IMPORTANT:
    env.getObjectTree() is the Python wrapper. It creates/reads the
    objectTree.json file internally. Do not call
    env.server.getObjectTree() with no arguments.
    """
    try:
        tree = env.getObjectTree()

        if not isinstance(tree, dict):
            return None

        return to_jsonable(tree)

    except Exception as exc:
        print(f"Warning: world-state capture failed: {exc!r}")
        return None


def get_progress(env: ScienceWorldEnv) -> Any:
    for method_name in ("get_goal_progress", "getGoalProgressStr"):
        method = getattr(env, method_name, None)

        if callable(method):
            try:
                return to_jsonable(method())
            except Exception as exc:
                return {"capture_error": repr(exc)}

    return None


def get_possible_actions(env: ScienceWorldEnv) -> Any:
    for method_name in (
        "get_possible_actions",
        "getPossibleActions",
    ):
        method = getattr(env, method_name, None)

        if callable(method):
            try:
                return to_jsonable(method())
            except Exception as exc:
                return {"capture_error": repr(exc)}

    return None


def get_gold_sequence(env: ScienceWorldEnv) -> Any:
    for method_name in (
        "get_gold_action_sequence",
        "getGoldActionSequence",
    ):
        method = getattr(env, method_name, None)

        if callable(method):
            try:
                return to_jsonable(method())
            except Exception as exc:
                return {"capture_error": repr(exc)}

    return None


# ---------------------------------------------------------------------
# Run one replay
# ---------------------------------------------------------------------

def run_once(
    task_name: str,
    actions: list[str],
    run_id: str,
) -> dict[str, Any]:

    env = ScienceWorldEnv("", envStepLimit=ENV_STEP_LIMIT)

    result: dict[str, Any] = {
        "task": task_name,
        "run_id": run_id,
        "variation": TASK_VARIATION,
        "requested_actions": len(actions),
        "executed_actions": 0,
        "steps": [],
        "initial_state": None,
        "final_state": None,
        "world_state_available": False,
        "error": None,
        "stopped_early": False,
        "stop_reason": None,
        "stop_step": None,
        "stop_action": None,
    }

    try:
        env.load(
            task_name,
            TASK_VARIATION,
            "",
            generateGoldPath=True,
        )

        initial_observation, initial_info = env.reset()

        result["task_description"] = to_jsonable(
            env.get_task_description()
        )
        result["gold_sequence"] = get_gold_sequence(env)
        result["initial_observation"] = to_jsonable(initial_observation)
        result["initial_info"] = to_jsonable(initial_info)
        result["initial_progress"] = get_progress(env)

        # Capture after reset. This call creates objectTree.json.
        result["initial_state"] = get_world_state(env)

        result["world_state_available"] = (
            result["initial_state"] is not None
        )

        for step_number, action in enumerate(actions, start=1):
            # Capture the world immediately before the action.
            state_before = get_world_state(env)

            try:
                observation, reward, done, info = env.step(action)
            except Exception as exc:
                result["error"] = {
                    "step": step_number,
                    "action": action,
                    "exception": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                result["stop_reason"] = "step_exception"
                break

            # Capture the world immediately after the action.
            state_after = get_world_state(env)
            progress = get_progress(env)

            step_record = {
                "step": step_number,
                "action": action,
                "observation": to_jsonable(observation),
                "reward": to_jsonable(reward),
                "done": bool(done),
                "info": to_jsonable(info),
                "progress": progress,
                "possible_actions_after": get_possible_actions(env),
                "state_before": state_before,
                "state_after": state_after,
                "world_state_available": (
                    state_before is not None
                    and state_after is not None
                ),
            }

            result["steps"].append(step_record)

            if done:
                result["stop_reason"] = "done_true"
                result["stop_step"] = step_number
                result["stop_action"] = action
                result["stop_reward"] = to_jsonable(reward)
                result["stop_progress"] = progress
                result["stopped_early"] = (
                    step_number < len(actions)
                )
                break

        if result["stop_reason"] is None:
            if result["error"] is None:
                result["stop_reason"] = (
                    "all_requested_actions_executed"
                )

            result["stopped_early"] = (
                len(result["steps"]) < len(actions)
            )

        result["executed_actions"] = len(result["steps"])
        result["final_progress"] = get_progress(env)
        result["final_state"] = get_world_state(env)

        if result["final_state"] is None:
            result["world_state_available"] = False

    except Exception as exc:
        result["error"] = {
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
        result["stop_reason"] = "setup_or_capture_exception"

    finally:
        try:
            env.close()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------
# Recursive structural comparison
# ---------------------------------------------------------------------

def diff_values(
    a: Any,
    b: Any,
    path: str = "$",
) -> list[dict[str, Any]]:
    """Return paths where two JSON-compatible values differ."""
    differences: list[dict[str, Any]] = []

    # Missing state is unknown, not evidence of a mismatch.
    if a is None or b is None:
        if a is not b:
            differences.append({
                "path": path,
                "type": "comparison_unavailable",
                "run_a_available": a is not None,
                "run_b_available": b is not None,
            })
        return differences

    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))

        for key in keys:
            child_path = f"{path}.{key}"

            if key not in a:
                differences.append({
                    "path": child_path,
                    "type": "missing_in_run_a",
                    "run_b": b[key],
                })
            elif key not in b:
                differences.append({
                    "path": child_path,
                    "type": "missing_in_run_b",
                    "run_a": a[key],
                })
            else:
                differences.extend(
                    diff_values(a[key], b[key], child_path)
                )

        return differences

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            differences.append({
                "path": f"{path}.__length__",
                "type": "length_mismatch",
                "run_a": len(a),
                "run_b": len(b),
            })

        for index, (left, right) in enumerate(zip(a, b)):
            differences.extend(
                diff_values(left, right, f"{path}[{index}]")
            )

        return differences

    if type(a) is not type(b) or a != b:
        differences.append({
            "path": path,
            "type": "value_mismatch",
            "run_a": a,
            "run_b": b,
        })

    return differences


def compare_optional_states(a: Any, b: Any) -> dict[str, Any]:
    """Compare states without claiming unavailable states are different."""
    if a is None or b is None:
        return {
            "available": False,
            "equal": None,
            "difference_count": None,
            "differences": [],
        }

    differences = diff_values(a, b)

    return {
        "available": True,
        "equal": not differences,
        "difference_count": len(differences),
        "differences": differences[:50],
    }


def compare_runs(
    run_a: dict[str, Any],
    run_b: dict[str, Any],
) -> dict[str, Any]:

    count_a = len(run_a["steps"])
    count_b = len(run_b["steps"])

    comparisons: dict[str, Any] = {
        "same_executed_step_count": count_a == count_b,
        "run_a_executed": count_a,
        "run_b_executed": count_b,
        "same_stop_reason": (
            run_a.get("stop_reason") == run_b.get("stop_reason")
        ),
        "same_stop_step": (
            run_a.get("stop_step") == run_b.get("stop_step")
        ),
        "initial_state_comparison": compare_optional_states(
            run_a.get("initial_state"),
            run_b.get("initial_state"),
        ),
        "final_state_comparison": compare_optional_states(
            run_a.get("final_state"),
            run_b.get("final_state"),
        ),
        "step_differences": [],
        "world_state_comparison_available": True,
    }

    if (
        run_a.get("initial_state") is None
        or run_b.get("initial_state") is None
        or run_a.get("final_state") is None
        or run_b.get("final_state") is None
    ):
        comparisons["world_state_comparison_available"] = False

    for index, (step_a, step_b) in enumerate(
        zip(run_a["steps"], run_b["steps"]),
        start=1,
    ):
        # Compare behavior independently of world-state availability.
        behavior_a = {
            key: step_a.get(key)
            for key in (
                "action",
                "observation",
                "reward",
                "done",
                "info",
                "progress",
            )
        }
        behavior_b = {
            key: step_b.get(key)
            for key in (
                "action",
                "observation",
                "reward",
                "done",
                "info",
                "progress",
            )
        }

        behavior_differences = diff_values(
            behavior_a,
            behavior_b,
        )

        state_before = compare_optional_states(
            step_a.get("state_before"),
            step_b.get("state_before"),
        )
        state_after = compare_optional_states(
            step_a.get("state_after"),
            step_b.get("state_after"),
        )

        if not state_before["available"] or not state_after["available"]:
            comparisons["world_state_comparison_available"] = False

        step_different = (
            bool(behavior_differences)
            or state_before["equal"] is False
            or state_after["equal"] is False
        )

        if step_different:
            comparisons["step_differences"].append({
                "step": index,
                "action_a": step_a.get("action"),
                "action_b": step_b.get("action"),
                "behavior_differences": behavior_differences[:20],
                "state_before_comparison": state_before,
                "state_after_comparison": state_after,
            })

    comparisons["first_differing_step"] = (
        comparisons["step_differences"][0]["step"]
        if comparisons["step_differences"]
        else None
    )

    comparisons["identical_recorded_behavior"] = (
        count_a == count_b
        and all(
            not diff_values(
                {
                    key: step_a.get(key)
                    for key in (
                        "action",
                        "observation",
                        "reward",
                        "done",
                        "info",
                        "progress",
                    )
                },
                {
                    key: step_b.get(key)
                    for key in (
                        "action",
                        "observation",
                        "reward",
                        "done",
                        "info",
                        "progress",
                    )
                },
            )
            for step_a, step_b in zip(
                run_a["steps"],
                run_b["steps"],
            )
        )
    )

    initial_comparison = comparisons["initial_state_comparison"]
    final_comparison = comparisons["final_state_comparison"]

    comparisons["identical_initial_world_state"] = (
        initial_comparison["equal"]
    )
    comparisons["identical_final_world_state"] = (
        final_comparison["equal"]
    )

    comparisons["identical_recorded_world_trajectory"] = (
        comparisons["world_state_comparison_available"]
        and not comparisons["step_differences"]
        and count_a == count_b
    )

    return comparisons


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    for task_name, actions in TASKS.items():
        print("\n" + "=" * 72)
        print(f"TASK: {task_name}")
        print("=" * 72)

        run_a = run_once(task_name, actions, "A")
        run_b = run_once(task_name, actions, "B")
        comparison = compare_runs(run_a, run_b)

        write_json(
            OUTPUT_DIR / f"{task_name}_run_A.json",
            run_a,
        )
        write_json(
            OUTPUT_DIR / f"{task_name}_run_B.json",
            run_b,
        )
        write_json(
            OUTPUT_DIR / f"{task_name}_comparison.json",
            comparison,
        )

        summary[task_name] = {
            "requested_actions": len(actions),
            "run_a_executed": run_a["executed_actions"],
            "run_b_executed": run_b["executed_actions"],
            "run_a_stop_reason": run_a["stop_reason"],
            "run_b_stop_reason": run_b["stop_reason"],
            "run_a_stop_step": run_a.get("stop_step"),
            "run_b_stop_step": run_b.get("stop_step"),
            "run_a_stop_action": run_a.get("stop_action"),
            "run_b_stop_action": run_b.get("stop_action"),
            "initial_world_state_comparison": (
                comparison["initial_state_comparison"]
            ),
            "final_world_state_comparison": (
                comparison["final_state_comparison"]
            ),
            "world_state_comparison_available": (
                comparison["world_state_comparison_available"]
            ),
            "first_differing_step": (
                comparison["first_differing_step"]
            ),
            "identical_recorded_behavior": (
                comparison["identical_recorded_behavior"]
            ),
            "identical_recorded_world_trajectory": (
                comparison["identical_recorded_world_trajectory"]
            ),
            "run_a_error": run_a["error"],
            "run_b_error": run_b["error"],
        }

        print(f"Requested actions: {len(actions)}")
        print(
            f"Executed steps: A={run_a['executed_actions']}, "
            f"B={run_b['executed_actions']}"
        )
        print(
            f"Stop reason: A={run_a['stop_reason']}, "
            f"B={run_b['stop_reason']}"
        )
        print(
            f"Stop step: A={run_a.get('stop_step')}, "
            f"B={run_b.get('stop_step')}"
        )
        print(
            "Initial world states equal: "
            f"{comparison['identical_initial_world_state']}"
        )
        print(
            "Final world states equal: "
            f"{comparison['identical_final_world_state']}"
        )
        print(
            "World-state comparison available: "
            f"{comparison['world_state_comparison_available']}"
        )
        print(
            "First differing step: "
            f"{comparison['first_differing_step']}"
        )
        print(
            "Identical recorded behavior: "
            f"{comparison['identical_recorded_behavior']}"
        )
        print(
            "Identical recorded world trajectory: "
            f"{comparison['identical_recorded_world_trajectory']}"
        )

        if run_a["error"] or run_b["error"]:
            print("WARNING: At least one run encountered an error.")

    write_json(OUTPUT_DIR / "summary.json", summary)

    print("\n" + "=" * 72)
    print("AUDIT COMPLETE")
    print(f"Results directory: {OUTPUT_DIR.resolve()}")
    print("Start by inspecting summary.json.")
    print("=" * 72)


if __name__ == "__main__":
    main()