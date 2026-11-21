from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_pulse_summary import comparison, group_results


def test_partial_trajectory_remains_in_denominator_without_inventing_s6():
    scores = {"states": {"S5": {"tp": 3, "true": 7, "fp": 1, "negative": 1},
                         "S6": {"tp": 0, "true": 0, "fp": 0, "negative": 8}}}
    row = {"episode": {"segments": {}, "complete_sequence": False},
           "complete_rear": False, "physical_status": "STOPPED",
           "changed_state_predictions": 0, "changed_side_predictions": 0,
           "windows": {"rear_stroke": {"active": scores, "passive": scores}}}
    totals = group_results([("late_partial", row, {}), ("late_raw_only", None, {"status":"STOPPED"})])
    assert totals["outcomes"]["attempts"] == totals["outcomes"]["unreached"] == 2
    assert totals["same_trajectory"]["rear_S5"]["passive"]["positive"] == 7
    assert totals["same_trajectory"]["rear_S6"]["delta_recall_pp"] is None
    assert totals["same_trajectory"]["rear_S6"]["delta_fpr_pp"] == 0.


def test_weak_recall_and_unloaded_false_positives_have_different_denominators():
    a = {"weak_side": {"true": 10, "tp": 1}, "side": {"negative": 4, "fp": 0}}
    b = {"weak_side": {"true": 10, "tp": 2}, "side": {"negative": 4, "fp": 1}}
    pair = comparison([a],[b],"handoff_weak",("passive","active"))
    assert pair["delta_recall_pp"] == 10. and pair["delta_fpr_pp"] == 25.
