
from __future__ import annotations

import re

from scienceworld import ScienceWorldEnv


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

TASK_NAME = "melt"
TASK_VARIATION = 0
ENV_STEP_LIMIT = 100


# ---------------------------------------------------------
# ACTION-TEMPLATE MATCHING
# ---------------------------------------------------------

import re


def matches_action_template(
    candidate_action: str,
    templates: list[str],
) -> bool:
    """Match a grounded action against generic ScienceWorld templates."""

    candidate_action = " ".join(candidate_action.split())

    # Known wording variants in the gold action sequence.
    normalized_action = re.sub(
        r"^pour (.+?) into (.+)$",
        r"pour \1 in \2",
        candidate_action,
    )

    # Some gold actions use 'examine', while the available template
    # list exposes generic look actions instead.
    effective_templates = list(templates)

    if any(
        template.startswith("look at ")
        or template == "look at OBJ"
        for template in templates
    ):
        effective_templates.append("examine OBJ")

    for template in effective_templates:
        template = " ".join(template.split())
        parts = template.split("OBJ")

        pattern = "^" + r".+?".join(
            re.escape(part) for part in parts
        ) + "$"

        if re.fullmatch(pattern, normalized_action):
            return True

    return False

# ---------------------------------------------------------
# ENVIRONMENT HELPERS
# ---------------------------------------------------------

def open_environment(
    task_name: str,
    variation: int = 0,
) -> ScienceWorldEnv:
    """Create, load, and reset a ScienceWorld environment."""
    env = ScienceWorldEnv(
        taskName=task_name,
        serverPath=None,
        envStepLimit=ENV_STEP_LIMIT,
    )

    env.load(
        task_name,
        variation,
        "",
        generateGoldPath=True,
    )

    env.reset()

    return env


def get_possible_actions(env: ScienceWorldEnv) -> list[str]:
    """Retrieve the generic action templates exposed by the environment."""
    if hasattr(env, "get_possible_actions"):
        actions = env.get_possible_actions()
    elif hasattr(env, "getPossibleActions"):
        actions = env.getPossibleActions()
    else:
        raise AttributeError(
            "Could not find get_possible_actions() "
            "or getPossibleActions() on this environment."
        )

    if actions is None:
        return []

    return list(actions)


def get_gold_sequence(env: ScienceWorldEnv) -> list[str]:
    """Retrieve the gold action sequence from the loaded task."""
    if hasattr(env, "get_gold_action_sequence"):
        actions = env.get_gold_action_sequence()
    elif hasattr(env, "getGoldActionSequence"):
        actions = env.getGoldActionSequence()
    else:
        raise AttributeError(
            "Could not find a gold-action-sequence method."
        )

    if actions is None:
        return []

    return list(actions)


# ---------------------------------------------------------
# DIAGNOSTIC
# ---------------------------------------------------------

def main() -> None:
    env = None

    try:
        env = open_environment(TASK_NAME, TASK_VARIATION)

        possible_actions = get_possible_actions(env)
        gold_actions = get_gold_sequence(env)

        print("=" * 70)
        print("SCIENCEWORLD ACTION-TEMPLATE DIAGNOSTIC")
        print("=" * 70)

        print(f"\nTask: {TASK_NAME}")
        print(f"Variation: {TASK_VARIATION}")
        print(f"Number of action templates: {len(possible_actions)}")
        print(f"Number of gold actions: {len(gold_actions)}")

        print("\nAvailable action templates:")
        for action in possible_actions:
            print(f"  {action}")

        print("\nGold actions and template matches:")

        unmatched = []

        for index, action in enumerate(gold_actions):
            matched = matches_action_template(
                action,
                possible_actions,
            )

            status = "MATCH" if matched else "NO MATCH"

            print(
                f"{index:>3}. [{status}] {action}"
            )

            if not matched:
                unmatched.append(action)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print(f"Gold actions checked: {len(gold_actions)}")
        print(
            f"Template matches: "
            f"{len(gold_actions) - len(unmatched)}"
        )
        print(f"Unmatched actions: {len(unmatched)}")

        if unmatched:
            print("\nUnmatched actions:")
            for action in unmatched:
                print(f"  - {action}")

            print(
                "\nIMPORTANT: An unmatched action is not necessarily "
                "invalid. It may use different wording from the "
                "generic action templates."
            )
        else:
            print(
                "\nAll gold actions matched a generic action template."
            )

        print(
            "\nThis diagnostic checks template compatibility only. "
            "It does not execute actions or verify their effects."
        )

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
