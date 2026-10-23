
from __future__ import annotations

import json
from pathlib import Path
from scienceworld import ScienceWorldEnv


TASKS = [
    ("melt", 0, ["water", "ice", "freezer", "stove", "fridge"]),
    ("boil", 0, ["water", "ice", "stove", "pot", "burner"]),
    (
        "power-component",
        0,
        ["red light bulb", "light bulb", "battery", "wire", "switch"],
    ),
]

OUTPUT_DIR = Path("src/sandbox/science_world/state_replay_results")



def find_objects(tree, keywords, path="$"):
    """Find actual objects by their name, not by arbitrary property keys."""
    matches = []

    if isinstance(tree, dict):
        # Object-tree nodes have a name field and a UUID.
        if "name" in tree and "uuid" in tree:
            name = str(tree["name"]).lower()

            if any(keyword in name for keyword in keywords):
                matches.append({
                    "path": path,
                    "name": tree["name"],
                    "uuid": tree["uuid"],
                    "isDeleted": tree.get("isDeleted"),
                    "propMaterial": tree.get("propMaterial"),
                    "propDevice": tree.get("propDevice"),
                    "propHeatSource": tree.get("propHeatSource"),
                    "propCoolingSource": tree.get("propCoolingSource"),
                    "propElectricalConnection": tree.get(
                        "propElectricalConnection"
                    ),
                })

        for key, value in tree.items():
            find_path = f"{path}.{key}"
            matches.extend(find_objects(value, keywords, find_path))

    elif isinstance(tree, list):
        for index, item in enumerate(tree):
            matches.extend(
                find_objects(item, keywords, f"{path}[{index}]")
            )

    return matches

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    env = ScienceWorldEnv("", envStepLimit=100)

    try:
        for task_name, variation, keywords in TASKS:
            print("\n" + "=" * 70)
            print(f"TASK: {task_name} | VARIATION: {variation}")
            print("=" * 70)

            env.load(task_name, variation, "")
            observation, info = env.reset()

            print("\nTask description:")
            print(info.get("taskDesc"))

            print("\nInitial observation:")
            print(observation)

            print("\nInitial inventory:")
            print(env.inventory())

            print("\nObject-tree matches:")
            tree = env.getObjectTree()
            matches = find_objects(tree, keywords)

            if matches:
                for match in matches[:50]:
                    print(f"\nPath: {match['path']}")
                    print(json.dumps(
                        match,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )[:1200])

            else:
                print("No keyword matches found.")

            all_results.append({
                "task": task_name,
                "variation": variation,
                "task_description": info.get("taskDesc"),
                "initial_observation": observation,
                "matches": matches,
            })

    finally:
        env.close()

    output_file = OUTPUT_DIR / "target_inspection.json"
    output_file.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"\nSaved inspection to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
