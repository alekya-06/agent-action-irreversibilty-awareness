
from scienceworld import ScienceWorldEnv

env = ScienceWorldEnv("", envStepLimit=100)

try:
    env.load("melt", 0, "", generateGoldPath=True)
    env.reset()

    gold = env.get_gold_action_sequence()
    possible = env.get_possible_actions()

    print("GOLD TYPE:", type(gold).__name__)
    print("GOLD:", repr(gold)[:3000])

    print("\nPOSSIBLE ACTIONS TYPE:", type(possible).__name__)
    print("POSSIBLE ACTIONS:", repr(possible)[:5000])

finally:
    try:
        env.close()
    except Exception:
        pass
