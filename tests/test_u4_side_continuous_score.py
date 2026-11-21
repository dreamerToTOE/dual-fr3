from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_side_continuous_score import alerts, event_matches


def test_three_step_alert_checks_truth_at_start_and_does_not_repeat():
    cases = [(False, True)] + [(False, False)]*3 + [(None, None)] + [(True, None)]*3 + [(False, False)]*3
    rows = [{"step": i, "fusion_side": prediction, "truth_side": true}
            for i, (prediction, true) in enumerate(cases)]
    result = alerts(rows, "fusion", 3)
    assert result["clear"] == {"correct": 1, "incorrect": 1, "unverifiable": 0}
    assert result["bearing"] == {"correct": 0, "incorrect": 0, "unverifiable": 1}
    assert alerts([rows[0]] + rows[2:4], "fusion", 3)["clear"]["incorrect"] == 0


def test_early_clear_held_through_first_loss_is_not_a_matched_declaration():
    rows = [{"step": i, "window": "handoff", "truth_side": i < 5 or i >= 9,
             "baseline_side": i >= 9, "fusion_side": i < 5 or i >= 9}
            for i in range(1,13)]
    event = {"minimum_consecutive_steps": 3, "true_clear_onset_step": 5,
             "clear_bout_last_observed_step": 8, "true_reload_confirm_step": 11,
             "baseline": {}, "fusion": {}}
    current = event_matches(rows, {"unload_command_start_step": 0,
        "from_unload_command_3": event, "from_unload_command_12": event})["from_unload_command_3"]
    assert current["baseline"]["new_clear_declaration_step"] is None
    assert current["fusion"]["new_clear_declaration_step"] == 7
    assert current["fusion"]["new_reload_declaration_step"] == 11
