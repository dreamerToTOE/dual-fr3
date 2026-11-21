from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_sustained_score import trial_events


def test_short_calls_and_early_bearing_do_not_count_as_sustained_transitions():
    truth = [True]*4 + [False]*4 + [True]*7 + [False]*13 + [True]*12
    rows = [{"step":i,"window":"handoff", "truth_side":state, "baseline_side": i >= 27 or i < 4,
             "fusion_side": state} for i,state in enumerate(truth,1)]
    events = trial_events(rows,120)
    n12 = events["first"]["12"]
    assert n12["loss"]["start"] == 16 and n12["loss"]["ticks"] == 13
    assert n12["loss"]["baseline"]["confirm_step"] is None
    assert n12["loss"]["fusion"]["confirm_step"] == 18
    assert n12["reload"]["start"] == 29 and n12["reload"]["baseline"]["confirm_step"] is None
    assert n12["reload"]["fusion"]["confirm_step"] == 31
    assert events["alerts"]["fusion"]["clear_1_to_5"] == 1
    assert events["alerts"]["baseline"]["bearing_false"] == 1
    assert events["first"]["24"]["loss"] is None
