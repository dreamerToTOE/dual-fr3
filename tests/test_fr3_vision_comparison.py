import json

import numpy as np
import pytest

from fr3_vision_comparison import VisionComparison, match_reference, summarize


def test_metrics_keep_missing_observations_out_of_errors():
    records = [
        {"phase": "A", "truth_position_m": [0, 0, 1], "camera_position_m": [.003, .004, 1]},
        {"phase": "A", "truth_position_m": [0, 0, 1], "camera_position_m": None},
        {"phase": "B1", "truth_position_m": None, "camera_position_m": [0, 0, 1]},
        {"phase": "B1", "truth_position_m": [0, 0, 1], "camera_position_m": [0, 0, .988]},
    ]
    summary = summarize(records)
    overall = summary["overall"]
    assert overall["attempts"] == 4
    assert overall["camera_detections"] == 3
    assert overall["matched"] == 2
    assert overall["misses"] == 1
    assert overall["unpaired"] == 1
    assert overall["detection_rate"] == .75
    assert overall["euclidean_mean_m"] == pytest.approx(.0085)
    assert overall["euclidean_rmse_m"] == pytest.approx(np.sqrt((.005 ** 2 + .012 ** 2) / 2))
    assert np.allclose(overall["xyz_bias_m"], [.0015, .002, -.006])
    assert np.allclose(overall["xyz_mae_m"], [.0015, .002, .006])
    assert summary["by_phase"]["A"]["euclidean_mean_m"] == pytest.approx(.005)
    assert summary["by_phase"]["A"]["euclidean_mean_m"] * 1000 == pytest.approx(5)
    assert summarize([records[1]])["overall"]["euclidean_mean_m"] is None


def test_independent_camera_stream_contains_no_reference_or_error(tmp_path):
    writer = VisionComparison(tmp_path, {"scene": "synthetic"})
    writer.append(sample_id=1, phase="B2", physics_step=8, capture_time_s=8 / 120,
                  frame_id=4, truth_position_m=[1, 2, 3], camera_position_m=[1, 2.003, 3.004],
                  frame_metadata={"truth_debug": [1, 2, 3]}, reference_debug="do not leak")
    writer.append(sample_id=2, phase="B2", physics_step=16, capture_time_s=16 / 120,
                  truth_position_m=[1, 2, 3], status="marker_not_visible")
    with pytest.raises(ValueError, match="Duplicate"):
        writer.append(sample_id=2, phase="B2", physics_step=16, capture_time_s=16 / 120,
                      truth_position_m=[1, 2, 3])
    result = writer.finalize()
    camera_rows = [json.loads(line) for line in (tmp_path / "camera.jsonl").read_text().splitlines()]
    assert len(camera_rows) == 2
    assert set(camera_rows[0]) == {"sample_id", "phase", "physics_step", "capture_time_s", "frame_id",
                                   "camera_position_m", "status", "source"}
    assert camera_rows[1]["camera_position_m"] is None
    reference_rows = [json.loads(line) for line in (tmp_path / "ground_truth.jsonl").read_text().splitlines()]
    assert "camera_position_m" not in reference_rows[0]
    assert result["overall"]["euclidean_max_m"] == pytest.approx(.005)
    assert result["overall"]["matched"] == 1
    assert result["overall"]["status_counts"] == {"detected": 1, "marker_not_visible": 1}
    assert (tmp_path / "comparison.csv").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["overall"]["misses"] == 1


def test_bad_position_is_rejected_before_writing_any_sample(tmp_path):
    writer = VisionComparison(tmp_path)
    with pytest.raises(ValueError, match="finite"):
        writer.append(sample_id=1, phase="A", physics_step=0, capture_time_s=0,
                      truth_position_m=[0, 0, 0], camera_position_m=[float("nan"), 0, 0])
    result = writer.finalize()
    assert result["overall"]["attempts"] == 0
    assert (tmp_path / "camera.jsonl").read_text() == ""


def test_reference_pairing_uses_capture_time_and_rejects_missing_history():
    history = [{"time_s": i / 120, "physics_step": i, "phase": "B2", "position_m": [i, 0, 0]}
               for i in range(121)]
    assert match_reference(history, .5)["physics_step"] == 60
    assert match_reference(history, .501)["physics_step"] == 60
    assert match_reference(history, -.001) is None
    assert match_reference(history, 1.001) is None
    assert match_reference(history, 1 + 1e-10)["physics_step"] == 120
    assert match_reference([], .5) is None
    assert match_reference(history, .501, max_time_delta_s=.0005) is None


def test_coarse_stage_summary_groups_transit_phases():
    rows = [{"phase": phase, "truth_position_m": [0, 0, 0], "camera_position_m": [0, .001, 0]}
            for phase in ("A_lift", "A_transfer", "B1_contact_transit", "B2_push")]
    summary = summarize(rows)
    assert summary["by_stage"]["A"]["matched"] == 2
    assert summary["by_stage"]["B1"]["euclidean_mean_m"] == .001
    assert summary["by_stage"]["B2"]["matched"] == 1
