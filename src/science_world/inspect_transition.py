from __future__ import annotations
from typing import Any
from scienceworld import ScienceWorldEnv

TASK_NAME = "melt"
VARIATION = 0
DIFFICULTY = ""


def walk_tree(node: Any, path: str = ""):
    """Traverse the ScienceWorld object tree safely."""
    if not isinstance(node, dict):
        return

    name = node.get("name", "unknown")
    current_path = f"{path}/{name}"

    yield current_path, node

    contents = node.get("contents", {})
    if isinstance(contents, dict):
        for child in contents.values():
            if isinstance(child, dict):
                yield from walk_tree(child, current_path)


def inspect_water_objects(env: ScienceWorldEnv) -> None:
    """Print water objects and their physical properties."""
    tree = env.getObjectTree()

    print("\n=== WATER OBJECTS ===")
    found = False

    for path, obj in walk_tree(tree):
        material = obj.get("propMaterial") or {}

        if (
            isinstance(material, dict)
            and material.get("substanceName", "").lower() == "water"
        ):
            found = True
            print({
                "path": path,
                "uuid": obj.get("uuid"),
                "temperatureC": material.get("temperatureC"),
                "stateOfMatter": material.get("stateOfMatter"),
                "isDeleted": obj.get("isDeleted"),
            })

    if not found:
        print("No water objects found with the expected material properties.")


def run_action(env: ScienceWorldEnv, action: str):
    """Execute one action and print its result."""
    observation, reward, done, info = env.step(action)

    print(f"\nACTION: {action}")
    print(f"OBSERVATION: {observation}")
    print(f"REWARD: {reward}")
    print(f"DONE: {done}")

    valid = info.get("valid", [])

    print("RELEVANT VALID ACTIONS:")
    matches = [
        candidate
        for candidate in valid
        if any(
            word in candidate.lower()
            for word in (
                "bathroom",
                "hallway",
                "kitchen",
                "water",
                "focus on",
                "open",
                "go to",
            )
        )
    ]

    if matches:
        for candidate in matches:
            print(f"  {candidate}")
    else:
        print("  No matching actions found.")

    return observation, reward, done, info


def find_action(info: dict, candidates: tuple[str, ...]):
    """Find an exact action from the current valid-action list."""
    valid = info.get("valid", [])
    valid_by_lower = {action.lower(): action for action in valid}

    for candidate in candidates:
        if candidate.lower() in valid_by_lower:
            return valid_by_lower[candidate.lower()]

    return None


def main() -> None:
    env = ScienceWorldEnv("", envStepLimit=100)

    try:
        env.load(TASK_NAME, VARIATION, DIFFICULTY)
        observation, info = env.reset()

        print("TASK:", env.get_task_description())
        print("\n=== INITIAL OBSERVATION ===")
        print(observation)

        # Record the initial water objects before taking any actions.
        inspect_water_objects(env)

        current_info = info

        # Step 1: Open the kitchen door if it is available.
        action = find_action(
            current_info,
            ("open door to kitchen",),
        )

        if action:
            observation, _, done, current_info = run_action(env, action)
            if done:
                return
        else:
            print("\nKitchen door opening action not found.")
            print("Current observation:", observation)
            print("Current valid actions:", current_info.get("valid", []))
            return

        # Step 2: Enter the kitchen.
        action = find_action(
            current_info,
            ("go to kitchen", "go to door to kitchen"),
        )

        if not action:
            print("\nKitchen navigation action not found.")
            print("Current observation:", observation)
            print("Current valid actions:", current_info.get("valid", []))
            return

        observation, _, done, current_info = run_action(env, action)
        if done:
            return

        # Step 3: Open the bathroom door if available.
        action = find_action(
            current_info,
            ("open bathroom door",),
        )

        if not action:
            print("\nBathroom door action not found from this location.")
            print("Current observation:", observation)
            print("Current valid actions:", current_info.get("valid", []))
            return

        observation, _, done, current_info = run_action(env, action)
        if done:
            return

        # Step 4: Look for a navigation action into the bathroom.
        action = find_action(
            current_info,
            ("go to bathroom", "go to bathroom door"),
        )

        if not action:
            print("\nNo bathroom entry action found.")
            print("Current observation:", observation)
            print("Current valid actions:", current_info.get("valid", []))
            return

        observation, _, done, current_info = run_action(env, action)
        if done:
            return

        # Step 5: Inspect the resulting room.
        print("\n=== CURRENT ROOM OBSERVATION ===")
        print(current_info.get("look", observation))

        # Step 6: Focus on water only if it is currently offered.
        action = find_action(
            current_info,
            ("focus on substance in toilet",),
        )
        
        if action:
            observation, _, done, current_info = run_action(env, action)
            if done:
                return
        else:
            print("Focus action for the toilet substance was not available.")
            print("Current valid actions:", current_info.get("valid", []))

        if action:
            observation, _, done, current_info = run_action(env, action)
        else:
            print("\nNo exact 'focus on water' action is currently suggested.")
            print(
                "Use the current observation and valid actions to identify "
                "the task-relevant substance; do not assume the toilet water "
                "is the intended target."
            )

        # Show the final observation and the current task status.
        print("\n=== FINAL OBSERVATION ===")
        print(current_info.get("look", observation))
        print("\n=== TASK STATUS ===")
        print("Score:", current_info.get("score"))
        print("Task completed:", done)

    finally:
        env.close()


if __name__ == "__main__":
    main()
