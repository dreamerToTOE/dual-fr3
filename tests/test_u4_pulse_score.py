import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_pulse_score import episode, scores
from u4_handover_score import anchors


def trial(values):
    segments = ["pre"] * 3 + ["out_neg"] * 3 + ["cross_pos"] * 3 + ["post"] * 3
    return [{"step": i + 1, "segment": segment, "truth_side": v, "truth_side_N": .2 if v else 0.,
             "truth_box_x_m": .14 + i * 1e-5, "truth_box_y_m": 1.01,
             "truth_state": "S5" if v else "S1", "active_state": "S5", "active_side": True,
             "probe_response": {} if i == len(values)-1 else None,
             "probe_before_loaded": False if i == len(values)-1 else None}
            for i, (v, segment) in enumerate(zip(values, segments, strict=True))]


def test_onset_during_probe_is_distinct_from_preexisting_contact():
    r = trial([False]*6 + [True]*6)
    event = episode(r)
    assert event["pre_end_loaded"] is False and event["post_end_loaded"] is True
    assert event["sustained_contact_during_previously_clear"] is True
    assert event["retrospective_decision_step"] == 12 and event["retrospective_correct"] is True
    assert not event["complete_sequence"]
    r[:3] = trial([True]*3 + [False]*3 + [True]*6)[:3]
    event = episode(r)
    assert event["pre_end_loaded"] is True and event["first_reload_after_unload"] == 9
    assert event["sustained_contact_during_previously_clear"] is None and event["retrospective_correct"] is False
    event = episode(trial([False]*6 + [True, False, True] + [False]*3))
    assert event["during_single_tick_load"] == 7 and event["during_load_ticks"] == 2
    assert event["single_tick_contact_during_previously_clear"] and not event["sustained_contact_during_previously_clear"]
    incoming = [dict(r[0], step=i, segment="", truth_side=True) for i in range(-11,1)]
    event = episode(incoming + trial([False]*6+[True]*6))
    assert event["before_command_hold_12"]["known_loaded"] == 12 and event["load_changed_during_pre_hold"]
    assert event["retrospective_correct"] and not event["retrospective_matches_preintervention"]


def test_fpr_and_weak_recall_treat_abstentions_as_unsolved():
    rows = trial([False]*6 + [True]*6)
    rows[0]["truth_side"] = None
    rows[0]["truth_state"] = "UNKNOWN"
    rows[-1]["active_side"] = None
    rows[-1]["active_state"] = "UNKNOWN"
    metric = scores(rows, "active")
    assert metric["side"]["negative"] == 5 and metric["side"]["fp"] == 5
    assert metric["states"]["S5"]["fpr"] == 1 and metric["side"]["coverage"] == 10/11
    assert metric["weak_side"] == {"true": 6, "tp": 5, "unknown": 1, "recall": 5/6}


def test_truncated_trial_ends_in_observed_stroke():
    partial = {"incomplete_run": {"status": "STOPPED"}, "stop_step": 101, "hz": 10,
               "phase_times_for_scoring_only": [{"phase": name, "status": "RUNNING", "sim_time_s": time}
                 for name, time in (("upper_B_side", 2.), ("upper_B_side_unload", 4.), ("upper_B_rear", 6.))]}
    assert anchors(partial) == [20,40,60,101]
