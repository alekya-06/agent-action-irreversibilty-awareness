from science_world.audit_scienceworld_replay import compare_runs, diff_values
from copy import deepcopy

from science_world.audit_scienceworld_replay import diff_values


def make_test_world():
    """A minimal world tree with one electrical device."""
    return {
        "uuid": "100",
        "name": "universe",
        "isDeleted": False,
        "contents": {
            "workshop-101": {
                "uuid": "101",
                "name": "workshop",
                "isDeleted": False,
                "contents": {
                    "red-light-bulb-102": {
                        "uuid": "102",
                        "name": "red light bulb",
                        "isDeleted": False,
                        "propDevice": {
                            "isActivable": False,
                            "isActivated": False,
                            "isBroken": False,
                        },
                        "contents": {},
                    }
                },
            }
        },
    }


def test_detects_changed_device_state():
    # Create two initially identical world states.
    world_a = make_test_world()
    world_b = deepcopy(world_a)

    # Introduce one controlled difference.
    world_b["contents"]["workshop-101"]["contents"][
        "red-light-bulb-102"
    ]["propDevice"]["isActivated"] = True

    differences = diff_values(world_a, world_b)

    # The difference must be detected.
    assert differences, "The changed device state was not detected!"

    # It must identify the correct property.
    changed_paths = [difference["path"] for difference in differences]

    expected_path = (
        "$.contents.workshop-101.contents."
        "red-light-bulb-102.propDevice.isActivated"
    )

    assert expected_path in changed_paths, (
        f"Expected property path not found. Got: {changed_paths}"
    )

    # It must report the original and modified values.
    matching_difference = next(
        difference
        for difference in differences
        if difference["path"] == expected_path
    )

    assert matching_difference["run_a"] is False
    assert matching_difference["run_b"] is True

    print("PASS: Changed device state detected correctly.")
    print(f"Property: {expected_path}")
    print("Run A: False")
    print("Run B: True")


def test_identical_world_states_have_no_differences():
    world_a = make_test_world()
    world_b = deepcopy(world_a)

    differences = diff_values(world_a, world_b)

    assert differences == [], (
        f"Identical states should have no differences. Got: {differences}"
    )

    print("PASS: Identical world states produce no differences.")


def test_detects_multiple_differences():
    world_a = make_test_world()
    world_b = deepcopy(world_a)

    bulb = world_b["contents"]["workshop-101"]["contents"][
        "red-light-bulb-102"
    ]
    bulb["propDevice"]["isActivated"] = True
    bulb["propDevice"]["isBroken"] = True

    differences = diff_values(world_a, world_b)
    changed_paths = {difference["path"] for difference in differences}

    expected_paths = {
        "$.contents.workshop-101.contents.red-light-bulb-102."
        "propDevice.isActivated",
        "$.contents.workshop-101.contents.red-light-bulb-102."
        "propDevice.isBroken",
    }

    assert expected_paths.issubset(changed_paths), (
        f"Expected both changes. Got: {changed_paths}"
    )

    print("PASS: Multiple changed properties are detected.")


def make_test_run():
    """Create a minimal recorded run with one action."""
    initial_state = make_test_world()
    final_state = deepcopy(initial_state)

    return {
        "initial_state": initial_state,
        "final_state": final_state,
        "stop_reason": "done_true",
        "stop_step": 1,
        "steps": [
            {
                "action": "activate bulb",
                "observation": "The bulb activates.",
                "reward": 1.0,
                "done": True,
                "info": {},
                "progress": "1/1",
                "state_before": deepcopy(initial_state),
                "state_after": deepcopy(final_state),
            }
        ],
    }


def test_compare_runs_identical_trajectories():
    run_a = make_test_run()
    run_b = deepcopy(run_a)

    result = compare_runs(run_a, run_b)

    assert result["identical_recorded_behavior"] is True
    assert result["identical_initial_world_state"] is True
    assert result["identical_final_world_state"] is True
    assert result["world_state_comparison_available"] is True
    assert result["step_differences"] == []
    assert result["first_differing_step"] is None
    assert result["identical_recorded_world_trajectory"] is True

    print("PASS: Identical trajectories are classified correctly.")


def test_compare_runs_detects_world_state_difference():
    run_a = make_test_run()
    run_b = deepcopy(run_a)

    # Change only the world state after the action.
    bulb = run_b["steps"][0]["state_after"]["contents"][
        "workshop-101"
    ]["contents"]["red-light-bulb-102"]

    bulb["propDevice"]["isActivated"] = True
    run_b["final_state"] = deepcopy(run_b["steps"][0]["state_after"])

    result = compare_runs(run_a, run_b)

    assert result["identical_recorded_behavior"] is True
    assert result["world_state_comparison_available"] is True
    assert result["step_differences"], "Expected a detected step difference."
    assert result["first_differing_step"] == 1
    assert result["identical_final_world_state"] is False
    assert result["identical_recorded_world_trajectory"] is False

    state_differences = result["step_differences"][0][
        "state_after_comparison"
    ]["differences"]

    changed_paths = {
        difference["path"] for difference in state_differences
    }

    expected_path = (
        "$.contents.workshop-101.contents."
        "red-light-bulb-102.propDevice.isActivated"
    )

    assert expected_path in changed_paths, (
        f"Expected changed state path not found: {changed_paths}"
    )

    print("PASS: World-state differences are detected at the correct step.")


def test_compare_runs_handles_unavailable_world_states():
    run_a = make_test_run()
    run_b = deepcopy(run_a)

    # Simulate an audit where the environment did not provide world states.
    run_a["initial_state"] = None
    run_a["final_state"] = None
    run_b["initial_state"] = None
    run_b["final_state"] = None

    for run in (run_a, run_b):
        run["steps"][0]["state_before"] = None
        run["steps"][0]["state_after"] = None

    result = compare_runs(run_a, run_b)

    assert result["identical_recorded_behavior"] is True
    assert result["world_state_comparison_available"] is False
    assert result["identical_initial_world_state"] is None
    assert result["identical_final_world_state"] is None
    assert result["step_differences"] == []
    assert result["first_differing_step"] is None
    assert result["identical_recorded_world_trajectory"] is False

    print("PASS: Unavailable world states are not treated as differences.")

if __name__ == "__main__":
    test_detects_changed_device_state()
