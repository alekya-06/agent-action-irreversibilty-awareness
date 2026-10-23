
import json
from pathlib import Path

RESULTS_DIR = (
    Path(__file__).resolve().parent / "replay_audit_results"
)

TASKS = ["melt", "boil", "power-component"]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    for task in TASKS:
        print("\n" + "=" * 72)
        print(f"TASK: {task}")
        print("=" * 72)

        for run_id in ("A", "B"):
            path = RESULTS_DIR / f"{task}_run_{run_id}.json"
            result = load_json(path)
            steps = result.get("steps", [])

            print(f"\n--- RUN {run_id} ---")
            print("Task description:", result.get("task_description"))
            print("Stop reason:", result.get("stop_reason"))
            print("Stop step:", result.get("stop_step"))
            print("Stop action:", result.get("stop_action"))
            print("Final progress:", result.get("final_progress"))
            print("Error:", result.get("error"))

            if not steps:
                print("No steps recorded.")
                continue

            final_step = steps[-1]

            print("\nLast executed step:")
            print("Step:", final_step.get("step"))
            print("Action:", final_step.get("action"))
            print("Reward:", final_step.get("reward"))
            print("Done:", final_step.get("done"))
            print("Observation:", final_step.get("observation"))
            print("Progress:", final_step.get("progress"))
           # print("Info:", json.dumps(
           #     final_step.get("info"),
           #     indent=2,
           #     ensure_ascii=False,
           # ))


if __name__ == "__main__":
    main()
