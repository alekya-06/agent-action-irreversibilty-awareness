
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

def show_actions(env, label):
    actions = env.get_valid_action_object_combinations()

    print(f"\n--- {label} ---")
    print(f"Valid combinations: {len(actions)}")

    for action in actions:
        print(action)

    return actions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    env = ScienceWorldEnv("", envStepLimit=100)

    try:
        for task_name, variation in TASKS:
            print("\n" + "=" * 70)
            print(f"TASK: {task_name} | VARIATION: {variation}")
            print("=" * 70)

            env.load(task_name, variation, "")
            observation, info = env.reset()

            print("\nTask description:")
            print(env.get_task_description())

            # Get valid actions at the initial state.
            actions = show_actions(env, "Initial state")

            # Save the task's reference sequence, if available.
            try:
                gold_sequence = env.get_gold_action_sequence()
            except Exception as exc:
                gold_sequence = None
                print(f"\nGold sequence unavailable: {exc}")

            print("\nReference action sequence:")
            print(gold_sequence)

            results.append({
                "task": task_name,
                "variation": variation,
                "initial_observation": observation,
                "valid_actions": actions,
                "gold_action_sequence": gold_sequence,
            })

    finally:
        env.close()

    output_file = OUTPUT_DIR / "valid_actions_inspection.json"
    output_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"\nSaved results to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
