import inspect
from scienceworld import ScienceWorldEnv

for name in dir(ScienceWorldEnv):
    if "object" in name.lower() or "state" in name.lower():
        method = getattr(ScienceWorldEnv, name)
        if callable(method):
            try:
                print(f"\n--- {name} ---")
                print(inspect.signature(method))
                print(inspect.getsource(method))
            except (TypeError, OSError, ValueError):
                pass