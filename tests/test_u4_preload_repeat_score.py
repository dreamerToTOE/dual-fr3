from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_preload_repeat_score import analyze


def series(truth, handoff_end, before=2):
    return [{"step": tick, "window": "side_stroke" if tick <= before else
             "handoff" if tick <= handoff_end else "rear_stroke",
             "truth_left_cup": tick <= before+2, "truth_side": state,
             "baseline_side": state, "fusion_side": state,
             "left_compression_N": .35}
            for tick,state in enumerate(truth,1)]


def test_clear_must_have_twelve_confirmed_steps_before_right_push():
    rows = series([True]*5+[False]*14+[True]*14,16)
    report = analyze(rows,120,None)
    assert report["handoff_bouts"]["longest"]["clear"]==11
    assert report["handoff_loss_eligible"] is False
    assert report["first_sustained_reload_including_rear"] is None


def test_preloaded_handoff_with_sustained_clear_and_later_rear_reload():
    rows = series([True]*8+[False]*14+[True]*14,22)
    rows[7]["fusion_side"] = False  # Declaration began one step before the observed loss.
    report = analyze(rows,120,None)
    event = report["first_sustained_handoff_loss"]
    assert (event["start"],event["end"],event["sustained_confirm_step"])==(9,22,20)
    assert event["baseline"]["confirm_step"]==11 and event["fusion"]["confirm_step"] is None
    assert report["errors"]["handoff"]["fusion"]["false_clear_alerts"]==1
    assert report["first_sustained_reload_within_handoff"] is None
    assert report["first_sustained_reload_including_rear"]["start"]==23
    assert report["from_left_cup_clear"]["longest"]["clear"]==14


def test_capture_end_censors_absent_reload_and_keeps_observed_release():
    rows = series([True]*8+[False]*14,22)
    report = analyze(rows,120,{"status":"STOPPED"})
    assert report["first_sustained_handoff_loss"]["right_censored_at_stop"]
    assert report["reload_right_censored_at_stop"]
    assert report["first_sustained_reload_including_rear"] is None
    assert not report["right_push_observed"]


def test_left_cup_clear_uses_physical_three_samples_and_preserves_weak_bearing():
    rows = series([True]*100,80)
    report = analyze(rows,120,None)
    assert report["left_cup_clear_confirm_step"]==7
    assert report["first_12_from_left_cup_clear"]=="bearing"
    assert report["from_left_cup_clear"]["from_start"]=={"bearing":74}
    assert report["first_1p5s_from_left_cup_clear"]["ticks"]==74
    assert report["errors"]["handoff"]["fusion"]["false_clear_alerts"]==0


def test_preload_transition_may_be_confirmed_during_first_handoff_steps():
    rows = series([True]*7+[False]*14+[True]*14,25,before=17)
    report = analyze(rows,120,None,preload_start=4)
    assert not report["handoff_loss_eligible"]
    early = report["preload"]["first_sustained_preload_loss"]
    assert early["onset_observed"]
    assert (early["start"],early["end"],early["sustained_confirm_step"])==(8,21,19)
    assert (early["preload_ticks"],early["handoff_ticks"])==(10,4)
    assert report["preload"]["first_sustained_reload_including_rear"]["start"]==22


def test_preload_force_stop_preserves_observed_clear_even_without_handoff():
    rows = series([True]*5+[False]*15,30,before=23)
    report = analyze(rows,120,{"status":"STOPPED"},preload_start=4)
    assert not report["handoff_observed"]
    early = report["preload"]["first_sustained_preload_loss"]
    assert early["ticks"]==15 and early["right_censored_at_stop"]
    assert report["preload"]["errors"]["baseline"]["truth_clear_steps"]==15


def test_observed_loss_can_skip_censored_clear_at_start_of_handoff():
    rows = series([None] * 2 + [False] * 12 + [True] * 13 + [False] * 15 + [True] * 15, 48)
    inherited = analyze(rows, 120, None)["first_sustained_handoff_loss"]
    assert inherited["start"] == 3 and not inherited["onset_observed"]
    targeted = analyze(rows, 120, None, require_observed_loss=True)
    event = targeted["first_sustained_handoff_loss"]
    assert event["start"] == 28 and event["onset_observed"]
