from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

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
TCP_META = {**META, "schema_version": 2, "tcp_calibration_passed": True,
            "tcp_offset_sensor_m": {"left": [0, -.155, .08], "right": [0, .155, .08]}}


def _tcp_trial():
    rows = _trial()
    axis = np.array([1, 2, 3]) / np.sqrt(14)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    for row in rows:
        for side, angle in (("left", 0.7), ("right", -0.45)):
            rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew
            value = row[side]
            world = np.array(value["gravity_corrected_world"])
            local = np.r_[rotation.T @ world[:3], rotation.T @ world[3:]]
            offset = np.array(TCP_META["tcp_offset_sensor_m"][side])
            value["gravity_corrected_local"] = local.tolist()
            value["sensor_position_m"] = [.4, -.3, 1.2]
            value["sensor_orientation_wxyz"] = [np.cos(angle / 2), *(axis * np.sin(angle / 2))]
            value["tcp_position_m"] = (value["sensor_position_m"] + rotation @ offset).tolist()
            value["tcp_gravity_corrected_local"] = np.r_[
                local[:3], local[3:] - np.cross(offset, local[:3])].tolist()
            value["tcp_gravity_corrected_world"] = np.r_[
                world[:3], world[3:] - np.cross(rotation @ offset, world[:3])].tolist()
    return rows


def test_tcp_translation_checks_both_mirrored_offsets_in_rotated_frames():
    rows = _tcp_trial()
    result = analyze(rows, TCP_META)
    check = result["tcp_transform_validation"]
    assert result["status"] == "PASS_FORCE_FEEDBACK"
    assert result["tcp_physical_calibration_status"] == "PASS"
    assert check["status"] == "PASS"
    assert check["valid_side_samples"] == {"left": len(rows), "right": len(rows)}
    assert check["by_phase"]["B2_push"]["left"]["samples_checked"] == 50
    assert max(check["max_errors"].values()) < 1e-12
    assert analyze(_trial(), META)["tcp_transform_validation"]["status"] == "UNVERIFIED"


def test_wrong_tcp_shift_sign_is_rejected_by_independent_relations():
    rows = _tcp_trial()
    value = rows[151]["left"]
    sensor = np.array(value["gravity_corrected_local"])
    offset = TCP_META["tcp_offset_sensor_m"]["left"]
    value["tcp_gravity_corrected_local"][3:] = (sensor[3:] + np.cross(offset, sensor[:3])).tolist()
    result = analyze(rows, TCP_META)
    assert result["status"] == "INVALID_DATA"
    check = result["tcp_transform_validation"]
    assert check["status"] == "FAIL"
    assert check["max_errors"]["moment_shift_local_Nm"] > 0.1
    assert check["by_phase"]["B1"]["left"]["max_errors"]["reverse_shift_local_Nm"] > 0.1


@pytest.mark.parametrize("bad", ("missing", "nonfinite", "position_shape", "force_changed"))
def test_schema_two_cannot_pass_missing_or_corrupt_tcp_feedback(bad):
    rows = _tcp_trial()
    value = rows[160]["right"]
    if bad == "missing":
        del value["tcp_gravity_corrected_local"]
    elif bad == "nonfinite":
        value["tcp_gravity_corrected_world"][5] = float("nan")
    elif bad == "position_shape":
        value["tcp_position_m"] = [0, 0]
    else:
        value["tcp_gravity_corrected_world"][0] += 1.0
    result = analyze(rows, TCP_META)
    assert result["status"] == "INVALID_DATA"
    assert result["tcp_transform_validation"]["status"] == "FAIL"
    if bad != "force_changed":
        assert not result["all_samples_finite"]


def test_tcp_offset_metadata_and_physical_calibration_are_required():
    rows = _tcp_trial()
    result = analyze(rows, {**TCP_META, "tcp_offset_sensor_m": {}})
    assert result["status"] == "INVALID_DATA"
    result = analyze(rows, {**TCP_META, "tcp_calibration_passed": None})
    assert result["tcp_transform_validation"]["status"] == "PASS"
    assert result["tcp_physical_calibration_status"] == "UNVERIFIED"
    assert result["status"] == "INCOMPLETE"
    assert analyze(rows, {**TCP_META, "tcp_calibration_passed": False})["status"] == "SENSOR_CHECKS_FAILED"


def test_tcp_recording_writes_sensor_and_tcp_comparison_plot(tmp_path):
    with gzip.open(tmp_path / "wrench.jsonl.gz", "wt") as stream:
        for row in _tcp_trial():
            stream.write(json.dumps(row) + "\n")
    (tmp_path / "metadata.json").write_text(json.dumps(TCP_META))
    result = write_analysis(tmp_path)
    assert result["tcp_transform_validation"]["status"] == "PASS"
    assert (tmp_path / "tcp_wrench.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


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


CONTROL_META = {**TCP_META, "b2_force_control_validation": {
    "held_phases": ["B2_hold", "B2_push", "B2_right_unload", "B2_right_retreat", "B2_right_release"],
    "release_phases": {"right": ["B2_right_retreat"], "left": ["B2_left_retreat"]},
    "normal_force_min_N": .1, "normal_force_max_N": 6., "min_contact_coverage": .98,
    "max_dropout_s": .05, "max_force_N": 10., "max_tcp_torque_Nm": 1.,
    "max_inward_m": .00002, "minimum_retreat_m": .03,
}}


def _scale_wrench(value, scale):
    for name in ("gravity_corrected_local", "gravity_corrected_world",
                 "tcp_gravity_corrected_local", "tcp_gravity_corrected_world"):
        value[name] = (np.asarray(value[name]) * scale).tolist()


def _move_tcp(value, delta):
    for name in ("sensor_position_m", "tcp_position_m"):
        value[name] = (np.asarray(value[name]) + delta).tolist()


def _renumber(rows):
    for index, row in enumerate(rows, 1):
        row["physics_step"], row["time_s"] = index, index / 120
    return rows


def _controlled_trial():
    rows = _tcp_trial()
    push_start = next(index for index, row in enumerate(rows) if row["phase"] == "B2_push")
    hold = []
    for _ in range(50):
        row = deepcopy(rows[push_start])
        row["phase"] = "B2_hold"
        row["left"]["qdot_rad_s"] = [0.] * 7
        hold.append(row)
    rows[push_start:push_start] = hold
    for phase, active in (("B2_right_unload", "right"), ("B2_right_retreat", "right"),
                          ("B2_right_release", "right"), ("B2_left_unload", "left"),
                          ("B2_left_retreat", "left")):
        start = deepcopy(rows[-1])
        for index in range(50):
            row = deepcopy(start)
            row["phase"] = phase
            row["left"]["qdot_rad_s"] = [0.] * 7
            if "unload" in phase:
                _scale_wrench(row[active], 1 - (index + 1) / 50)
            if "retreat" in phase:
                direction = np.array([1., 0., 0.]) if active == "right" else np.array([0., -1., 0.])
                _move_tcp(row[active], direction * (.035 * (index + 1) / 50))
            rows.append(row)
    return _renumber(rows)


def test_controlled_b2_checks_stationary_support_and_both_full_retreats():
    result = analyze(_controlled_trial(), CONTROL_META)
    assert result["status"] == "PASS_FORCE_FEEDBACK"
    check = result["b2_force_control_validation"]
    assert check["status"] == "PASS"
    assert check["held_phases"]["B2_hold"]["samples"] == 50
    assert check["held_phases"]["B2_hold"]["contact_coverage"] == 1.
    assert check["held_phases"]["B2_right_release"]["longest_contact_dropout_s"] == 0.
    for side in ("left", "right"):
        assert check["releases"][side]["status"] == "PASS"
        assert np.isclose(check["releases"][side]["final_outward_distance_m"], .035)
        assert check["releases"][side]["maximum_inward_excursion_m"] == 0.
    assert result["release_transients"]["right"]["phase"] == "B2_right_retreat"


def test_stationary_support_dropout_is_not_hidden_by_qdot_filter():
    rows = _controlled_trial()
    held = [row for row in rows if row["phase"] == "B2_hold"]
    for row in held[10:20]:
        _scale_wrench(row["left"], 0.)
    result = analyze(rows, CONTROL_META)
    assert result["acceptance"]["B2_left_support"]["status"] == "CONSISTENT"
    check = result["b2_force_control_validation"]
    assert check["status"] == "FAIL"
    assert check["held_phases"]["B2_hold"]["longest_contact_dropout_s"] == 10 / 120
    assert check["held_phases"]["B2_hold"]["contact_coverage"] == .8
    assert result["status"] == "PASS_FORCE_FEEDBACK_WITH_TASK_WARNINGS"


def test_unload_ramp_is_not_exempt_from_force_spike_envelope():
    rows = _controlled_trial()
    row = next(row for row in rows if row["phase"] == "B2_left_unload")
    _scale_wrench(row["left"], 100.)
    result = analyze(rows, CONTROL_META)
    check = result["b2_force_control_validation"]
    assert check["phase_peaks"]["B2_left_unload"]["left"]["force_norm_max_N"] > 40
    assert check["phase_peaks"]["B2_left_unload"]["left"]["status"] == "FAIL"
    assert check["status"] == "FAIL"
    assert result["status"] == "PASS_FORCE_FEEDBACK_WITH_TASK_WARNINGS"


@pytest.mark.parametrize("which", ("first_sample", "later_backtrack"))
def test_retreat_cannot_hide_inward_tcp_motion(which):
    rows = _controlled_trial()
    indices = [index for index, row in enumerate(rows) if row["phase"] == "B2_right_retreat"]
    index = indices[0] if which == "first_sample" else indices[15]
    prior = rows[index - 1]["right"]["tcp_position_m"]
    current = rows[index]["right"]["tcp_position_m"]
    _move_tcp(rows[index]["right"], np.array(prior) - current + [-.0001, 0., 0.])
    check = analyze(rows, CONTROL_META)["b2_force_control_validation"]
    assert check["status"] == "FAIL"
    assert check["releases"]["right"]["maximum_backtrack_m"] > .00009
    if which == "first_sample":
        assert check["releases"]["right"]["maximum_inward_excursion_m"] > .00009
    else:
        assert check["releases"]["right"]["maximum_inward_excursion_m"] == 0.


def test_missing_required_retreat_fails_and_schema_three_keeps_tcp_checks():
    rows = _controlled_trial()
    assert analyze(rows, {**CONTROL_META, "schema_version": 3})["tcp_transform_validation"]["status"] == "PASS"
    rows = _renumber([row for row in rows if row["phase"] != "B2_right_retreat"])
    check = analyze(rows, CONTROL_META)["b2_force_control_validation"]
    assert check["status"] == "FAIL"
    assert check["missing_required_phases"] == ["B2_right_retreat"]


A_CONTROL_META = {**CONTROL_META, "a_force_control_validation": {
    "required_phases": ["A_approach_transit", "A_attach", "A_transfer", "A_lift", "A_over_front", "A_release"],
    "max_force_N": 30., "max_tcp_torque_Nm": 3., "max_internal_force_N": 15.,
}, "b1_force_control_validation": {
    "required_phases": ["B1"], "max_force_N": 30., "max_tcp_torque_Nm": 3.,
}}


def _set_world_wrench(row, side, wrench):
    value = row[side]
    w, x, y, z = value["sensor_orientation_wxyz"]
    rotation = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                         [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                         [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])
    local = np.r_[rotation.T @ wrench[:3], rotation.T @ wrench[3:]]
    offset = np.asarray(TCP_META["tcp_offset_sensor_m"][side])
    value["gravity_corrected_world"] = list(wrench)
    value["gravity_corrected_local"] = local.tolist()
    value["raw_parent_on_tool_local"] = (-local).tolist()
    value["tcp_gravity_corrected_local"] = np.r_[local[:3], local[3:] - np.cross(offset, local[:3])].tolist()
    value["tcp_gravity_corrected_world"] = np.r_[wrench[:3], wrench[3:] - np.cross(rotation @ offset, wrench[:3])].tolist()


def _a_controlled_trial():
    rows = _controlled_trial()
    for phase in ("A_approach_transit", "A_attach", "A_transfer", "A_release"):
        start = deepcopy(next(row for row in rows if row["phase"] == "A_lift"))
        start["phase"] = phase
        rows.insert(50, start)
    return _renumber(rows)


def test_full_force_pass_requires_a_b2_payload_and_sensor_checks():
    result = analyze(_a_controlled_trial(), A_CONTROL_META)
    assert result["a_force_control_validation"]["status"] == "PASS"
    assert result["task_force_status"] == "FULL_FLOW_FORCE_PASS"
    assert analyze(_controlled_trial(), CONTROL_META)["task_force_status"] == "UNVERIFIED"
    result = analyze(_a_controlled_trial(), {**A_CONTROL_META, "calibration_passed": None})
    assert result["task_force_status"] != "FULL_FLOW_FORCE_PASS"


def test_a_opposing_force_is_task_failure_even_with_correct_payload_sum():
    rows = _a_controlled_trial()
    row = next(row for row in rows if row["phase"] == "A_transfer")
    for side, sign in (("left", 1), ("right", -1)):
        wrench = np.asarray(row[side]["gravity_corrected_world"])
        wrench[0] = sign * 20.
        _set_world_wrench(row, side, wrench)
    result = analyze(rows, A_CONTROL_META)
    check = result["a_force_control_validation"]
    assert check["phase_peaks"]["A_transfer"]["left"]["force_norm_max_N"] < 30.
    assert check["peak_internal_force_norm_N"] > 20.
    assert check["status"] == "FAIL"
    assert result["task_force_status"] == "FULL_FLOW_FORCE_FAIL"
    assert result["status"] == "PASS_FORCE_FEEDBACK_WITH_TASK_WARNINGS"
    assert result["calibration_status"] == "PASS"
    assert result["acceptance"]["A_lift_payload"]["status"] == "PASS"


@pytest.mark.parametrize("phase", ("A_approach_transit", "A_attach", "A_release"))
def test_a_contact_transitions_cannot_hide_force_spikes(phase):
    rows = _a_controlled_trial()
    row = next(row for row in rows if row["phase"] == phase)
    _set_world_wrench(row, "left", [31., 0., 0., 0., 0., 0.])
    result = analyze(rows, A_CONTROL_META)
    assert result["a_force_control_validation"]["status"] == "FAIL"
    assert result["a_force_control_validation"]["phase_peaks"][phase]["left"]["raw_force_norm_max_N"] == 31.
    assert result["task_force_status"] == "FULL_FLOW_FORCE_FAIL"


def test_a_tcp_moment_and_missing_transition_are_independent_failures():
    rows = _a_controlled_trial()
    row = next(row for row in rows if row["phase"] == "A_attach")
    _set_world_wrench(row, "right", [0., 0., 0., 0., 3.1, 0.])
    check = analyze(rows, A_CONTROL_META)["a_force_control_validation"]
    assert check["status"] == "FAIL"
    assert check["phase_peaks"]["A_attach"]["right"]["tcp_torque_norm_max_Nm"] == 3.1
    rows = _renumber([row for row in _a_controlled_trial() if row["phase"] != "A_release"])
    check = analyze(rows, A_CONTROL_META)["a_force_control_validation"]
    assert check["status"] == "FAIL"
    assert check["missing_required_phases"] == ["A_release"]


@pytest.mark.parametrize("contract", ({}, {"required_phases": ["A_transfer"]},
    {**A_CONTROL_META["a_force_control_validation"], "max_force_N": float("nan")},
    {**A_CONTROL_META["a_force_control_validation"], "required_phases": "A_transfer"}))
def test_invalid_a_contract_cannot_pass(contract):
    result = analyze(_a_controlled_trial(), {**A_CONTROL_META, "a_force_control_validation": contract})
    assert result["a_force_control_validation"]["status"] == "FAIL"
    assert result["task_force_status"] == "FULL_FLOW_FORCE_FAIL"


def test_full_force_pass_requires_b1_envelope_and_all_transition_samples():
    rows = _a_controlled_trial()
    metadata = {**A_CONTROL_META}
    del metadata["b1_force_control_validation"]
    assert analyze(rows, metadata)["task_force_status"] == "UNVERIFIED"
    row = next(row for row in rows if row["phase"] == "B1")
    _set_world_wrench(row, "left", [0., -31., 0., 0., 0., 0.])
    result = analyze(rows, A_CONTROL_META)
    assert result["b1_force_control_validation"]["status"] == "FAIL"
    assert result["b1_force_control_validation"]["phase_peaks"]["B1"]["left"]["force_norm_max_N"] == 31.
    assert result["task_force_status"] == "FULL_FLOW_FORCE_FAIL"


def test_b1_transition_moment_spike_is_not_hidden_by_moving_window():
    rows = _a_controlled_trial()
    row = deepcopy(next(row for row in rows if row["phase"] == "B1"))
    row["phase"] = "B1_contact_transit"
    row["left"]["qdot_rad_s"] = [0.] * 7
    _set_world_wrench(row, "left", [0., 0., 0., 0., 3.1, 0.])
    rows.insert(50, row)
    result = analyze(_renumber(rows), A_CONTROL_META)
    assert result["b1_force_control_validation"]["status"] == "FAIL"
    assert result["calibration_status"] == "PASS"
    assert result["status"] == "PASS_FORCE_FEEDBACK_WITH_TASK_WARNINGS"
