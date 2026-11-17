from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from fr3_contact_state import ContactStateEstimator, STATES


CONFIG = json.loads(Path(__file__).parents[1].joinpath("fr3_four_wall_scene_config.json").read_text())


def sample(**changes):
    value = dict(time_s=1., left_wrench_world=[0.] * 6, right_wrench_world=[0.] * 6,
                 tcp_positions_m={"left": [-.25, .8, .787], "right": [.25, .8, .787]},
                 cup_normals_world={"left": [1., 0, 0], "right": [-1., 0, 0]})
    value.update(changes)
    return value


def camera(position=(0, .8, .787), **changes):
    value = dict(status="detected", position_m=list(position), capture_time_s=.95,
                 received_time_s=.97, frame_id=4)
    value.update(changes)
    return value


def stable(estimator, values):
    for index in range(3):
        result = estimator.update(**{**values, "time_s": values["time_s"] + index / 120})
    return result


def front_sample(y=.8, x=0., right_loaded=False):
    return sample(left_wrench_world=[0, -2., 0, 0, 0, 0],
                  right_wrench_world=[2. if right_loaded else 0, 0, 0, 0, 0, 0],
                  tcp_positions_m={"left": [x, y - .075, .791],
                                   "right": [x + .075, y, .787]},
                  cup_normals_world={"left": [0, 1., 0], "right": [-1., 0, 0]})


def test_no_load_without_visual_separation_is_unknown_not_no_contact():
    estimator = ContactStateEstimator(CONFIG)
    result = stable(estimator, sample())
    assert result["state"] == "UNKNOWN"
    assert result["recommended_skills"] == ["hold_reobserve"]
    result = estimator.update(**sample(time_s=2., camera=camera(),
        tcp_positions_m={"left": [-.075, .8, .787], "right": [.075, .8, .787]}))
    assert result["raw_state"] == "UNKNOWN"


def test_fresh_camera_and_separated_unloaded_tcps_confirm_no_contact():
    result = stable(ContactStateEstimator(CONFIG), sample(camera=camera()))
    assert result["state"] == "NO_TOOL_CONTACT"
    assert result["observability"] == "observed"
    assert "align" in result["recommended_skills"]


@pytest.mark.parametrize("observation", [camera(received_time_s=1.1), camera(capture_time_s=1.1, received_time_s=1.2),
                                        camera(capture_time_s=.5, received_time_s=.6),
                                        camera(status="marker_not_visible"), camera(position_m=[np.nan, 0, 0])])
def test_future_stale_missing_and_nonfinite_camera_are_not_observed(observation):
    result = stable(ContactStateEstimator(CONFIG), sample(camera=observation))
    assert result["state"] == "UNKNOWN"
    assert not result["features"]["camera_fresh"]


def test_last_camera_pose_is_not_held_indefinitely_after_occlusion():
    estimator = ContactStateEstimator(CONFIG)
    assert stable(estimator, sample(camera=camera()))["state"] == "NO_TOOL_CONTACT"
    result = estimator.update(**sample(time_s=1.4, camera=camera()))
    assert result["state"] == "UNKNOWN"


def test_occluded_bilateral_suspended_payload_uses_force_and_not_phase():
    result = stable(ContactStateEstimator(CONFIG), sample(
        left_wrench_world=[0, 0, -6., 0, 0, 0], right_wrench_world=[0, 0, -6.2, 0, 0, 0]))
    assert result["state"] == "BILATERAL_SUSPENDED"
    assert result["observability"] == "partial"
    assert result["features"]["combined_vertical_force_N"] == -12.2


def test_opposed_side_loads_at_support_height():
    result = stable(ContactStateEstimator(CONFIG), sample(
        left_wrench_world=[-1., 0, 0, 0, 0, 0], right_wrench_world=[1., 0, 0, 0, 0, 0],
        tcp_positions_m={"left": [-.075, -.3, .787], "right": [.075, -.3, .787]}))
    assert result["state"] == "BILATERAL_SUPPORTED"


@pytest.mark.parametrize("y,x,right_loaded,state", [(.8, 0, False, "LEFT_FLOOR_CONTACT"),
    (1.076, 0, False, "LEFT_REAR_CONTACT"), (1.076, -.1, True, "DUAL_REAR_CONTACT"),
    (1.076, -.245, True, "DUAL_CORNER_CONTACT")])
def test_loaded_encoder_contact_geometry_distinguishes_constraints(y, x, right_loaded, state):
    result = stable(ContactStateEstimator(CONFIG), front_sample(y, x, right_loaded))
    assert result["state"] == state


def test_coarse_visual_wall_proximity_does_not_prove_wall_contact():
    value = front_sample(y=1.06)
    result = stable(ContactStateEstimator(CONFIG), {**value, "camera": camera((0, 1.076, .773))})
    assert result["state"] == "LEFT_FLOOR_CONTACT"


def test_released_corner_requires_visual_corner_and_both_tools_outside():
    value = sample(camera=camera((-.247, 1.074, .776)),
                   tcp_positions_m={"left": [-.245, .966, .791], "right": [-.135, 1.076, .787]})
    result = stable(ContactStateEstimator(CONFIG), value)
    assert result["state"] == "RELEASED_CORNER"


def test_camera_loaded_contact_conflict_and_raw_overload_are_immediate():
    estimator = ContactStateEstimator(CONFIG)
    value = front_sample()
    result = estimator.update(**{**value, "camera": camera((.5, .8, .787))})
    assert result["state"] == "INCONSISTENT"
    assert result["observability"] == "conflicting"
    result = estimator.update(**{**value, "time_s": 2., "left_wrench_world": [0, -31., 0, 0, 0, 0]})
    assert result["state"] == "INCONSISTENT"


def test_ablations_do_not_inspect_excluded_modalities():
    force = ContactStateEstimator(CONFIG, mode="force_only")
    result = stable(force, {**front_sample(1.076), "camera": {"invalid": object()}})
    assert result["state"] == "LEFT_REAR_CONTACT"
    vision = ContactStateEstimator(CONFIG, mode="vision_only")
    result = stable(vision, sample(camera=camera(), left_wrench_world=None, right_wrench_world=None))
    assert result["state"] == "NO_TOOL_CONTACT"
    assert not any("force" in key for key in result["features"])


def test_task_phase_truth_known_support_and_duplicate_time_are_rejected():
    estimator = ContactStateEstimator(CONFIG)
    for key in ("phase", "truth_position_m", "known_support", "skill_id"):
        with pytest.raises(TypeError):
            estimator.update(**sample(), **{key: "forbidden"})
    estimator.update(**sample())
    with pytest.raises(ValueError, match="increasing"):
        estimator.update(**sample())


def test_skill_library_states_and_entry_targets_are_consistent():
    library = json.loads(Path(__file__).parents[1].joinpath("fr3_contact_skills.json").read_text())
    assert {state["id"] for state in library["states"]} == set(STATES)
    assert len(library["skills"]) == 10
    assert len({skill["id"] for skill in library["skills"]}) == 10
    for skill in library["skills"]:
        assert set(skill["entry_states"] + skill["target_states"]).issubset(STATES)
        assert skill["runtime_existing_functions"] and skill["success"] and skill["abort"]
    changed = deepcopy(CONFIG)
    changed["task"] = {"phase": "arbitrary forbidden task label"}
    assert stable(ContactStateEstimator(changed), front_sample())["state"] == "LEFT_FLOOR_CONTACT"


def test_estimator_thresholds_are_the_documented_library_values():
    library = json.loads(Path(__file__).parents[1].joinpath("fr3_contact_skills.json").read_text())
    estimator = ContactStateEstimator(CONFIG)
    assert estimator.thresholds == library["observation_contract"]
    assert estimator.thresholds["rear_side_geometry_band_m"] == .0002
    assert stable(estimator, front_sample())["state"] == "LEFT_FLOOR_CONTACT"
    estimator = ContactStateEstimator(CONFIG)
    estimator.thresholds["contact_normal_threshold_N"] = 2.1
    assert stable(estimator, front_sample())["state"] == "UNKNOWN"
    estimator = ContactStateEstimator(CONFIG)
    estimator.thresholds["camera_max_age_s"] = .01
    assert stable(estimator, sample(camera=camera()))["state"] == "UNKNOWN"
