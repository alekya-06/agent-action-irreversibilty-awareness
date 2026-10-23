
from scienceworld import ScienceWorldEnv
import json

env = ScienceWorldEnv("", envStepLimit=100)

try:
    env.load("melt", 0, "")
    observation, info = env.reset()

    print("=== TASK ===")
    print(env.get_task_description())

    print("\n=== POSSIBLE OBJECTS ===")
    print(env.get_possible_objects())

    print("\n=== VALID ACTIONS ===")
    valid_actions = env.get_valid_action_object_combinations()
    for action in valid_actions:
        print(action)

    print("\n=== TASK VARIATIONS ===")
    print("Maximum variations:", env.get_max_variations("melt"))
    print("Train variations:", len(env.get_variations_train()))
    print("Dev variations:", len(env.get_variations_dev()))
    print("Test variations:", len(env.get_variations_test()))

    print("\n=== OBJECT TREE ===")
    tree = env.getObjectTree()
    print(json.dumps(tree, indent=2)[:10000])

finally:
    env.close()
