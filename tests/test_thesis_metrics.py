import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "scenes/w760_300mm_two_layer/experiments"))

from thesis_metrics import confusion_metrics, event_metrics
from task_state_labels import TASK_STATES, task_state


def test_confusion_metrics_reports_precision_recall_f1_and_unknown():
    trace = [
        {"truth": "S1", "pred": "S1"},
        {"truth": "S1", "pred": "UNKNOWN"},
        {"truth": "S2", "pred": "S1"},
        {"truth": "S2", "pred": "S2"},
    ]
    metrics = confusion_metrics(trace)
    assert metrics["S1"]["tp"] == 1
    assert metrics["S1"]["truth"] == 2
    assert metrics["S1"]["predicted"] == 2
    assert metrics["S1"]["recall"] == .5
    assert metrics["S1"]["precision"] == .5
    assert metrics["S1"]["f1"] == .5
    assert metrics["S1"]["unknown"] == 1


def test_event_metrics_uses_transition_start_and_excludes_early_prediction():
    trace = [
        {"step": 0, "time_s": 0.0, "truth": "S1", "pred": "S0"},
        {"step": 1, "time_s": 1 / 120, "truth": "S1", "pred": "S1"},
        {"step": 2, "time_s": 2 / 120, "truth": "S1", "pred": "S1"},
        {"step": 3, "time_s": 3 / 120, "truth": "S2", "pred": "S1"},
        {"step": 4, "time_s": 4 / 120, "truth": "S2", "pred": "S2"},
        {"step": 5, "time_s": 5 / 120, "truth": "S2", "pred": "S2"},
    ]
    events = event_metrics(trace)
    assert events["S1"]["detected"] == 1
    assert events["S1"]["median_delay_ms"] == pytest.approx(1000 / 120)
    assert events["S2"]["detected"] == 1
    assert events["S2"]["median_delay_ms"] == pytest.approx(1000 / 120)


def test_merged_push_labels_keep_dual_bearing_and_unknown_distinct():
    raw_pairs = [("S1", "S5"), ("S5", "S1"), ("S5", "UNKNOWN"), ("S6", "S5")]
    trace = [{"step": i, "time_s": i / 120, "truth": task_state(truth), "pred": task_state(pred)}
             for i, (truth, pred) in enumerate(raw_pairs)]
    metrics = confusion_metrics(trace, TASK_STATES)
    assert "S6" not in metrics
    assert metrics["S1"]["tp"] == 2
    assert metrics["S1"]["recall"] == pytest.approx(2 / 3)
    assert metrics["S1"]["precision"] == pytest.approx(2 / 3)
    assert metrics["S1"]["unknown"] == 1
    assert metrics["S5"]["truth"] == 1
    assert metrics["S5"]["tp"] == 0
    events = event_metrics(trace, states=TASK_STATES)
    assert events["S1"]["events"] == 1
    assert events["S1"]["median_delay_ms"] == 0
