import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_left_unload_score import ACTIVE, evaluate, event_timing, first_streak, left_cup_reference


def test_missing_reference_is_unknown_and_measured_lip_remains_loaded():
    cup = {"label": "left_cup", "collider0": ACTIVE, "collider1": "left",
           "current_step_observed": True, "normal_impulse_Ns": .010,
           "normal_axis_impulse_abs_Ns": [.008, .006, 0.]}
    def ref(pairs, valid=True):
        return left_cup_reference({"cube_contacts": {"valid": valid, "pairs": pairs}}, 120)
    assert ref([]) == {"truth_left_cup": False, "truth_left_cup_x_N": 0}
    assert ref([], False)["truth_left_cup"] is None
    stale = dict(cup, current_step_observed=False)
    assert ref([stale])["truth_left_cup"] is None
    assert ref([cup, stale]) == {"truth_left_cup": True, "truth_left_cup_x_N": None}
    assert ref([cup, cup]) == {"truth_left_cup": True, "truth_left_cup_x_N": 1.92}


def test_transient_tool_release_and_early_or_missed_external_release():
    rows = []
    side = [True]*5 + [False]*4 + [True]*3
    for i, v in enumerate(side, 1):
        rows.append({"step": i, "window": "handoff" if i < 10 else "rear_stroke",
                     "truth_side": v, "truth_left_cup": i < 2 or i == 4,
                     "truth_side_N": .09 if v and i < 4 else .2 if v else 0.,
                     "baseline_side": False if i < 4 or i >= 10 else None,
                     "fusion_side": v if i != 6 else None,
                     "truth_state": "S5" if v else "S1", "baseline_state": "UNKNOWN", "fusion_state": "S5",
                     "unload_event": i == 6})
    out = evaluate(rows, 120)
    assert out["events"]["left_cup_clear_confirm_step"] == 7
    e = out["events"]["from_unload_command_3"]
    assert (e["last_observed_bearing_before_clear_step"], e["true_clear_onset_step"],
            e["true_clear_confirm_step"], e["true_reload_confirm_step"]) == (5, 6, 8, 12)
    assert e["baseline"]["early_declaration"] and e["baseline"]["missed_first_clear_bout"]
    assert e["fusion"]["clear_confirm_within_first_true_bout_step"] == 9
    assert e["fusion"]["matched_clear_delay_from_true_onset_s"] == 3/120
    assert e["fusion"]["bearing_confirm_within_first_true_reload_bout_step"] == 12
    assert out["events"]["from_unload_command_12"]["true_clear_confirm_step"] is None
    assert out["windows"]["handoff"]["bearing_below_point_one_N"] == 3
    assert out["windows"]["handoff"]["fusion"]["weak_side"]["true"] == 2
    assert out["windows"]["handoff"]["fusion"]["side"]["negative"] == 4
    assert evaluate([], 120)["events"] is None


def test_window_entered_during_clearance_is_left_censored():
    current = [{"step": i, "truth_side": False, "baseline_side": False, "fusion_side": None} for i in range(4, 8)]
    e = event_timing(current, 3, 120, 3, [{"step": 1, "truth_side": True}, {"step": 3, "truth_side": False}])
    assert e["initial_loss_unwitnessed"] and e["baseline"]["first_declaration_minus_true_confirm_s"] is None
    assert first_streak(current[:1]+current[2:], "truth_side", False, 3) is None
