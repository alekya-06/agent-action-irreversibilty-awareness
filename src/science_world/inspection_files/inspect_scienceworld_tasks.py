
from __future__ import annotations

import json
from pathlib import Path

from scienceworld import ScienceWorldEnv


TASKS = [
    ("melt", 0),
    ("boil", 0),
    ("power-component", 0),
]

OUTPUT_DIR = Path("src/sandbox/science_world/state_replay_results")
STEP_LIMIT = 100


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    env = ScienceWorldEnv("", envStepLimit=STEP_LIMIT)

    try:
        for task_name, variation in TASKS:
            print("\n" + "=" * 70)
            print(f"TASK: {task_name} | VARIATION: {variation}")
            print("=" * 70)

            env.load(task_name, variation, "")
            observation, info = env.reset()

            task_description = info.get("taskDesc", "Not provided")

            print("\nTask description:")
            print(task_description)

            print("\nInitial observation:")
            print(observation)

            print("\nInitial inventory:")
            print(env.inventory())

            print("\nInitial location:")
            print(env.look())

            object_tree = env.getObjectTree()

            results.append({
                "task": task_name,
                "variation": variation,
                "task_description": task_description,
                "initial_observation": observation,
                "initial_inventory": env.inventory(),
                "initial_location": env.look(),
                "object_tree": object_tree,
            })

    finally:
        env.close()

    output_file = OUTPUT_DIR / "task_inspection.json"
    output_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"\nInspection saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
