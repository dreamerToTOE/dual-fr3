from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_side_continuous_score import alerts


def test_three_step_alert_checks_truth_at_start_and_does_not_repeat():
    cases = [(False, True)] + [(False, False)]*3 + [(None, None)] + [(True, None)]*3 + [(False, False)]*3
    rows = [{"step": i, "fusion_side": prediction, "truth_side": true}
            for i, (prediction, true) in enumerate(cases)]
    result = alerts(rows, "fusion", 3)
    assert result["clear"] == {"correct": 1, "incorrect": 1, "unverifiable": 0}
    assert result["bearing"] == {"correct": 0, "incorrect": 0, "unverifiable": 1}
    assert alerts([rows[0]] + rows[2:4], "fusion", 3)["clear"]["incorrect"] == 0
