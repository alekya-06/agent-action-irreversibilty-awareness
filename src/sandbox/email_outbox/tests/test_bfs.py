
from bfs import bfs_recovery


def get_decrement_actions(state):
    return [{"step": -1}] if state > 0 else []


def apply_step(state, action):
    return state + action["step"]


def test_bfs_finds_shortest_path():
    result = bfs_recovery(
        initial_state=3,
        target_state=0,
        get_actions=get_decrement_actions,
        apply_action=apply_step,
        max_depth=3,
    )

    assert result["reversible"] is True
    assert result["recovery_cost"] == 3
    assert len(result["actions"]) == 3


def test_bfs_fails_when_horizon_is_too_short():
    result = bfs_recovery(
        initial_state=3,
        target_state=0,
        get_actions=get_decrement_actions,
        apply_action=apply_step,
        max_depth=2,
    )

    assert result["reversible"] is False
    assert result["recovery_cost"] == "unrecoverable"


def test_bfs_returns_zero_cost_for_restored_state():
    result = bfs_recovery(
        initial_state=0,
        target_state=0,
        get_actions=get_decrement_actions,
        apply_action=apply_step,
    )

    assert result["reversible"] is True
    assert result["recovery_cost"] == 0


def test_negative_horizon_is_rejected():
    try:
        bfs_recovery(
            1, 0, get_decrement_actions, apply_step, max_depth=-1
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Negative horizon was accepted")
