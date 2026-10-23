
import inspect
from scienceworld import ScienceWorldEnv

env = ScienceWorldEnv("", envStepLimit=100)

try:
    print("=== Public methods related to actions ===")
    for name in dir(env):
        if any(term in name.lower() for term in (
            "action", "valid", "command", "step", "task", "goal"
        )):
            attribute = getattr(env, name, None)
            if callable(attribute):
                try:
                    signature = inspect.signature(attribute)
                except (TypeError, ValueError):
                    signature = "(signature unavailable)"

                print(f"{name}{signature}")

    print("\n=== step() implementation ===")
    try:
        print(inspect.getsource(type(env).step))
    except (OSError, TypeError):
        print("Could not retrieve step() source.")

    print("\n=== Package location ===")
    print(inspect.getfile(type(env)))

finally:
    env.close()
