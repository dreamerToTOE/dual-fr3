from copy import deepcopy
import gzip
import json

import numpy as np

from fr3_wrist_ft_analysis import analyze, write_analysis


def _trial():
    rows = []
    for phase in ("initialization", "A_lift", "A_over_front", "B1", "B2_push"):
        for _ in range(50):
            row = {"time_s": (len(rows) + 1) / 120, "physics_step": len(rows) + 1, "phase": phase}
            for side in ("left", "right"):
                wrench = [0.0] * 6
                if phase.startswith("A"):
                    # Deliberately unequal load sharing; sum is Cube gravity.
                    wrench[2] = -9.0 if side == "left" else -3.2625
                if phase == "B1" and side == "left":
                    wrench[1] = -2.0
                if phase == "B2_push":
                    wrench[1 if side == "left" else 0] = -0.5 if side == "left" else 3.0
                moving = phase in ("B1", "B2_push") and not (phase == "B1" and side == "right")
                row[side] = {
                    "raw_parent_on_tool_local": [-value for value in wrench],
                    "tool_gravity_local": [0.0] * 6,
                    "gravity_corrected_local": wrench.copy(), "gravity_corrected_world": wrench.copy(),
                    "sensor_position_m": [0, 0, 0], "sensor_orientation_wxyz": [1, 0, 0, 0],
                    "q_rad": [0.0] * 7, "qdot_rad_s": [0.02 if moving else 0.0] * 7,
                }
            rows.append(row)
    return rows


META = {"task_status": "PASS_B1_B2_PROXY_CONTACT", "calibration_passed": True,
        "cube_mass_kg_reference": 1.25, "gravity_world_m_s2": [0, 0, -9.81]}


def test_correct_wrench_direction_and_unequal_payload_split_pass():
    result = analyze(_trial(), META)
    assert result["status"] == "PASS_FORCE_FEEDBACK"
    assert result["acceptance"]["A_lift_payload"]["status"] == "PASS"
    assert result["acceptance"]["B1_left_push_direction"]["median_N"] == -2
    assert result["acceptance"]["B2_right_push_direction"]["median_N"] == 3
    assert not result["support_task_issue"]


def test_zero_support_force_is_task_warning_not_sensor_failure():
    rows = _trial()
    for row in rows:
        if row["phase"] == "B2_push":
            row["left"]["gravity_corrected_world"][1] = 0
    result = analyze(rows, META)
    assert result["status"] == "PASS_FORCE_FEEDBACK_WITH_SUPPORT_WARNING"
    assert result["support_task_issue"]
    assert result["acceptance"]["B2_left_support"]["status"] == "NO_SUSTAINED_FORCE"


def test_correct_median_but_intermittent_force_is_not_a_direction_mismatch():
    rows = _trial()
    for index, row in enumerate(rows):
        if row["phase"] == "B2_push":
            row["left"]["gravity_corrected_world"][1] = -1.0 if index % 2 else 0.2
    result = analyze(rows, META)
    assert result["acceptance"]["B2_left_support"]["status"] == "INTERMITTENT_FORCE"
    assert result["status"] == "PASS_FORCE_FEEDBACK_WITH_SUPPORT_WARNING"
    for index, row in enumerate(rows):
        if row["phase"] == "B2_push":
            row["right"]["gravity_corrected_world"][0] = 1.0 if index % 2 else -0.2
    assert analyze(rows, META)["status"] == "INCOMPLETE"


def test_opposing_attachment_load_is_reported_without_a_safety_threshold():
    rows = _trial()
    for row in rows:
        if row["phase"] == "A_lift":
            row["phase"] = "A_transfer"
            row["left"]["gravity_corrected_world"][:3] = [100, 0, -6.13125]
            row["right"]["gravity_corrected_world"][:3] = [-100, 0, -6.13125]
    load = analyze(rows, META)["A_attachment_internal_load"]
    assert load["peak_internal_force_norm_N"] == 100
    np.testing.assert_allclose(load["combined_force_at_peak_world_N"], [0, 0, -12.2625])


def test_release_spike_is_a_task_warning_after_sensor_checks_pass():
    rows = _trial()
    row = deepcopy(rows[-1])
    row.update(phase="B2_push_transit", physics_step=len(rows) + 1, time_s=(len(rows) + 1) / 120)
    row["right"]["gravity_corrected_world"] = [200, 0, 0, 0, 3, 0]
    rows.append(row)
    result = analyze(rows, META)
    assert result["release_transients"]["right"]["peak_force_N"] == 200
    assert result["release_task_issue"]
    assert result["status"] == "PASS_FORCE_FEEDBACK_WITH_TASK_WARNINGS"


def test_reversed_contact_force_and_payload_mismatch_fail():
    rows = _trial()
    for row in rows:
        if row["phase"] == "B2_push":
            row["right"]["gravity_corrected_world"][0] = -3
        if row["phase"] == "A_lift":
            row["left"]["gravity_corrected_world"][2] += 2
    result = analyze(rows, META)
    assert result["status"] == "SENSOR_CHECKS_FAILED"
    assert result["acceptance"]["B2_right_push_direction"]["status"] == "DIRECTION_MISMATCH"
    assert np.isclose(result["acceptance"]["A_lift_payload"]["combined_force_error_norm_p95_N"], 2)


def test_nonfinite_or_missing_physics_samples_cannot_pass():
    rows = _trial()
    rows[0]["left"]["raw_parent_on_tool_local"][0] = np.nan
    result = analyze(rows, META)
    assert result["status"] == "INVALID_DATA"
    assert result["valid_side_samples"]["left"] == len(rows) - 1
    rows = _trial()
    rows.pop(10)
    assert analyze(rows, META)["status"] == "INVALID_DATA"
    rows = _trial()
    rows[3]["time_s"] += 0.001
    assert analyze(rows, META)["status"] == "INVALID_DATA"


def test_missing_windows_and_missing_calibration_are_unverified():
    result = analyze([], META)
    assert result["status"] == "INVALID_DATA"
    assert result["acceptance"]["static_initialization"]["status"] == "UNVERIFIED"
    rows = [row for row in _trial() if row["phase"] != "A_over_front"]
    for index, row in enumerate(rows):
        row["physics_step"] = index + 1
        row["time_s"] = (index + 1) / 120
    result = analyze(rows, META)
    assert result["status"] == "INCOMPLETE"
    assert result["acceptance"]["A_over_front_payload"]["status"] == "UNVERIFIED"
    assert analyze(_trial())["calibration_status"] == "UNVERIFIED"
    assert analyze(_trial())["full_flow_status"] == "UNVERIFIED"
    assert analyze(_trial())["acceptance"]["A_lift_payload"]["status"] == "UNVERIFIED"


def test_static_gravity_residual_is_checked_separately_from_direction():
    rows = deepcopy(_trial())
    for row in rows:
        if row["phase"] == "initialization":
            row["right"]["gravity_corrected_local"][3] = 0.1
    result = analyze(rows, META)
    assert result["acceptance"]["static_initialization"]["status"] == "FAIL"
    assert result["status"] == "SENSOR_CHECKS_FAILED"


def test_gzip_recording_writes_reviewable_summary_and_plot(tmp_path):
    with gzip.open(tmp_path / "wrench.jsonl.gz", "wt") as stream:
        for row in _trial():
            stream.write(json.dumps(row) + "\n")
    (tmp_path / "metadata.json").write_text(json.dumps(META))
    result = write_analysis(tmp_path)
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["status"] == result["status"] == "PASS_FORCE_FEEDBACK"
    assert (tmp_path / "wrench.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "wrench_contact.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
