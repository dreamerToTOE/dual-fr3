from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_post_release_probe_score import boundary, compare_trial, fixed_prestate, reference_fields


def trial():
    return [{"step": step, "window": "side_stroke" if step < 3 else "handoff",
             "segment": "pre" if 7 <= step < 19 else "out_neg" if step == 19 else
                        "cross_pos" if 20 <= step < 36 else "post" if 36 <= step < 50 else "",
             "truth_side": step < 7 or step >= 20, "truth_side_N": 0.5 if step < 7 or step >= 20 else 0.,
             "truth_left_cup": step < 7, "truth_left_detached": step >= 7, "passive_side": None, "active_side": None,
             "command_dx_m": 0.00025 if step >= 20 else -0.00025 if step == 19 else 0.,
             "probe_before_loaded": True if step == 47 else None,
             "probe_response": {} if step == 47 else None}
            for step in range(1, 61)]


def test_fresh_contact_cannot_validate_retrospective_bearing_or_reload():
    rows = trial()
    result = compare_trial(rows, 19, 120, False)
    assert result["preexisting_side_load"]["label"] == "clear"
    assert result["new_bearing_during_intervention_if_preclear"]["first_12step_onset"] == 20
    assert result["new_bearing_before_positive_command_if_preclear"]["positive_ticks"] == 0
    compare = result["preexisting_state_comparison"]
    assert compare["passive_at_last_premotion_sample"]["outcome"] == "abstain"
    assert compare["frozen_pulse_retrospective"]["outcome"] == "false_bearing"
    assert result["premotion_events_only"]["first_sustained_reload_within_handoff"] is None
    reload = result["events_over_observed_motion"]["first_sustained_reload_within_handoff"]
    assert reload["onset_period"] == "during_intervention" and reload["fusion"]["confirm_step"] is None


def test_premotion_reference_requires_contiguous_clear_left_cup_and_known_side():
    rows = trial()
    rows[8]["truth_side"] = None
    assert fixed_prestate(rows, 19)["label"] == "mixed_or_unobserved"
    rows[8]["truth_left_detached"] = False
    assert fixed_prestate(rows, 19)["label"] == "left_cup_clear_not_verified"
    del rows[8]
    assert fixed_prestate(rows, 19)["label"] == "incomplete_reference"


def test_probe_on_first_nonzero_command_must_match_recorded_boundary():
    marker = {"phase": "upper_B_u4_side_probe", "status": "MOTION_COMMAND", "sim_time_s": 18 / 120}
    header = {"hz": 120, "phase_times_for_scoring_only": [marker]}
    assert boundary(trial(), header) == 19
    marker["sim_time_s"] += 1 / 120
    with pytest.raises(ValueError, match="command marker"):
        boundary(trial(), header)


def test_zero_left_cup_impulse_alone_does_not_indicate_detachment():
    pair = {"label": "left_cup", "collider0": "/World/Cube_10", "collider1": "/World/left_cup",
            "current_step_observed": True, "normal_impulse_Ns": 0.0, "touching": True}
    physical = {"attachments": {"left": False}, "cube_contacts": {"valid": True, "pairs": [pair]}}
    assert reference_fields(physical, 120)["truth_left_detached"] is False
    pair["touching"] = False
    assert reference_fields(physical, 120)["truth_left_detached"] is True
    physical["attachments"]["left"] = True
    assert reference_fields(physical, 120)["truth_left_detached"] is False
