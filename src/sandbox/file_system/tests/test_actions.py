import os
import sys

parent_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(parent_dir)

from state import FileSystemState
from actions import get_permissible_actions


def make_filesystem():
    return FileSystemState(
        directories={"/", "/reports", "/archive"},
        files={
            "/reports/final.csv": "report data",
            "/reports/draft.csv": "draft data",
        },
    )


def test_all_four_action_types_are_represented():
    fs = make_filesystem()

    actions = get_permissible_actions(fs)
    action_types = {action["type"] for action in actions}

    assert "delete_file" in action_types
    assert "move_file" in action_types
    assert "delete_directory" in action_types
    assert "rename" in action_types


def test_move_to_same_location_is_not_permissible():
    fs = make_filesystem()

    actions = get_permissible_actions(fs)

    assert not any(
        action["type"] == "move_file"
        and action["source"] == action["destination"]
        for action in actions
    )


def test_action_generation_does_not_mutate_filesystem():
    fs = make_filesystem()
    original = fs.clone()

    get_permissible_actions(fs)

    assert fs.exactly_equals(original)
    assert fs.validate()


def test_generated_actions_are_permissible():
    fs = make_filesystem()

    actions = get_permissible_actions(fs)

    for action in actions:
        # Every generated action must successfully execute
        # on the original filesystem state.
        new_fs, _ = __import__(
            "transitions"
        ).transition(fs, action)

        assert new_fs.validate()


def test_generated_actions_are_unique():
    fs = make_filesystem()

    actions = get_permissible_actions(fs)

    action_keys = [
        str(sorted(action.items()))
        for action in actions
    ]

    assert len(action_keys) == len(set(action_keys))


def test_action_generation_returns_a_list():
    fs = make_filesystem()

    actions = get_permissible_actions(fs)

    assert isinstance(actions, list)
    assert len(actions) > 0
