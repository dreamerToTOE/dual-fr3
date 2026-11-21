from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from seven_replay import BACK, metrics, target_events


def test_abstention_and_wrong_object_are_not_matches():
    states = [("UNKNOWN", "S1"), ("S1", "S5"), ("S1", "S5"), ("S1", "S5"),
              ("S5", "UNKNOWN"), ("S5", "S5"), ("S5", "S5")]
    rows = [{"step": i, "time_s": i / 120, "ref": ref, "pred": pred,
             "reference_objects": [BACK] if ref == "S5" else [],
             "boundaries": [BACK] if i in (2, 3, 4) else []} for i, (ref, pred) in enumerate(states, 1)]
    result = metrics(rows, 120)
    assert result["correct"] == 2 and result["reference_known"] == 6
    assert result["coverage_on_known"] == 5 / 6 and result["accuracy_including_abstentions"] == 2 / 6
    assert result["exact_horizontal_objects_and_class"] == [0, 3]
    assert result["switches"][0]["delay_ms"] == 8.33
    header = {"hz": 120, "case": "u5", "phase_times_for_scoring_only": [
        {"phase": "upper_B_rear", "status": "RUNNING", "sim_time_s": 0},
        {"phase": "upper_B_rear", "status": "FORCE_CONFIRMED", "sim_time_s": 7 / 120}]}
    check = target_events(header, rows)["upper_B_rear"]
    assert check["first_reference_step"] == 5 and check["early_episodes"] == 1
