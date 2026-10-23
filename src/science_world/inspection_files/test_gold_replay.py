
from __future__ import annotations

import json
from scienceworld import ScienceWorldEnv


TASKS = {
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


def run_once(task_name: str, actions: list[str]) -> list[dict]:
    env = ScienceWorldEnv("", envStepLimit=100)
    results = []

    try:
        env.load(task_name, 0, "", generateGoldPath=True)
        env.reset()

        for step_no, action in enumerate(actions, start=1):
            observation, reward, done, info = env.step(action)

            # Store stable, public-facing outputs.
            try:
                progress = env.get_goal_progress()
            except Exception:
                progress = None

            results.append({
                "step": step_no,
                "action": action,
                "observation": observation,
                "reward": reward,
                "done": done,
                "valid": info.get("valid"),
                "progress": progress,
            })

            if done:
                break

    finally:
        env.close()

    return results


def main():
    for task_name, actions in TASKS.items():
        print("\n" + "=" * 70)
        print(f"TASK: {task_name}")
        print("=" * 70)

        run_a = run_once(task_name, actions)
        run_b = run_once(task_name, actions)

        print(f"Run A steps: {len(run_a)}")
        print(f"Run B steps: {len(run_b)}")

        first_difference = None

        for i, (a, b) in enumerate(zip(run_a, run_b), start=1):
            # Ignore valid-action ordering only if it differs by ordering.
            valid_a = a["valid"]
            valid_b = b["valid"]

            if isinstance(valid_a, list) and isinstance(valid_b, list):
                valid_a = sorted(map(str, valid_a))
                valid_b = sorted(map(str, valid_b))

            comparable_a = {**a, "valid": valid_a}
            comparable_b = {**b, "valid": valid_b}

            if comparable_a != comparable_b:
                first_difference = (i, comparable_a, comparable_b)
                break

        if len(run_a) != len(run_b):
            print("RESULT: DIFFERENT NUMBER OF STEPS")
        elif first_difference is None:
            print("RESULT: IDENTICAL RECORDED OUTPUTS")
        else:
            step_no, a, b = first_difference
            print(f"RESULT: FIRST DIFFERENCE AT STEP {step_no}")
            print("Run A:")
            print(json.dumps(a, indent=2, default=str))
            print("Run B:")
            print(json.dumps(b, indent=2, default=str))


if __name__ == "__main__":
    main()
