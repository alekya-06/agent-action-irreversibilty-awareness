
from __future__ import annotations

from scienceworld import ScienceWorldEnv


TASKS = [
    ("melt", 0),
    ("boil", 0),
    ("power-component", 0),
]


def main():
    env = ScienceWorldEnv("", envStepLimit=100)

    try:
        for task_name, variation in TASKS:
            print("\n" + "=" * 65)
            print(f"TASK: {task_name} | VARIATION: {variation}")
            print("=" * 65)

            # Request gold-path generation when loading the task.
            env.load(
                task_name,
                variation,
                "",
                generateGoldPath=True,
            )

            env.reset()

            print("\nTask description:")
            print(env.get_task_description())

            try:
                sequence = env.get_gold_action_sequence()
                print("\nGold action sequence:")
                if isinstance(sequence, (list, tuple)):
                    for i, action in enumerate(sequence, start=1):
                        print(f"{i:02d}. {action}")
                else:
                    print(sequence)
            except Exception as exc:
                print(f"Could not retrieve gold sequence: {exc}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
