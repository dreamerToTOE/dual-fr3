import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_handover_score import ACTIVE, SIDE, events, load_at, object_scores, state_scores


def test_bearing_and_abstention_are_timed_separately():
    horizontal = {"horizontal_objects": [SIDE], "ambiguous_contacts": []}
    pair = {"collider0": ACTIVE, "collider1": SIDE, "normal_impulse_Ns": .02,
            "normal_axis_impulse_abs_Ns": [.02, 0., 0.], "current_step_observed": True}
    snap = {"valid": True, "pairs": [pair]}
    assert load_at(snap, horizontal, SIDE, 0, 120) == (True, 2.4)
    pair["current_step_observed"] = False
    assert load_at(snap, horizontal, SIDE, 0, 120) == (None, None)
    snap["pairs"] = []
    assert load_at(snap, horizontal, SIDE, 0, 120) == (False, 0.)
    rows = [{"step": i, "truth_side": t, "fusion_side": p}
            for i, (t, p) in enumerate([(None, True), (False, True), (True, None),
                                         (True, True), (True, True), (True, True)])]
    score = object_scores(rows, "fusion", "side", 10)
    assert score["tp"] == 3 and score["fp"] == 1 and score["abstain_on_positive"] == 1
    assert score["truth_unknown"] == 1 and score["coverage_on_known"] == .8
    assert score["truth_events"]["first_step"] == 2 and score["confirmation_minus_first_bearing_s"] == .3
    assert events([rows[2], rows[4], rows[5]], "truth_side")["first_3step_confirm"] is None


def test_state_precision_uses_all_known_states_and_exact_objects():
    states = [("S0", "S6"), ("S6", "S6"), ("S6", "UNKNOWN"), ("UNKNOWN", "S6")]
    rows = [{"truth_state": t, "fusion_state": p, "truth_objects": "side|rear", "fusion_objects": "side"}
            for t, p in states]
    scores = state_scores(rows, "fusion")
    assert scores["totals"]["S6"] == {"true": 2, "predicted_on_known": 2, "tp": 1, "tp_exact_objects": 0,
                                       "unknown_on_true": 1, "precision": .5, "recall": .5}


def test_reload_waits_for_witnessed_unload_and_keeps_abstentions():
    from u4_handover_score import reacquisition
    truth = [True, False, None, False, False, False, True, True, True]
    est = [True, None, None, None, None, None, None, True, True]
    rows = [{"window": "handoff" if tick <= 6 else "rear_stroke", "step": tick, "truth_side": t,
             "baseline_side": p, "fusion_side": p} for tick,(t,p) in enumerate(zip(truth,est), 1)]
    result = reacquisition(rows, 120)
    assert (result["three_step_unloaded"], result["first_subsequent_load"]) == (6, 7)
    assert not result["fusion"]["negative_seen_before_reload"]
    assert result["fusion"]["predicted_events"]["first_3step_confirm"] is None
