import gzip
import json

import pytest

from fr3_row_acceptance import force_checks


def recording(directory, change=None, missing_step=None):
    """Five continuous 120 Hz traces with alternating arm roles and explicit unloads."""
    phases, sequence, frames = [], [], []
    for cube_id in range(1, 6):
        start = (cube_id - 1) * 500
        support = "right" if cube_id in (2, 4) else "left"
        push = "left" if support == "right" else "right"
        sequence.append({"cube_id": cube_id, "support_side": support,
                         "push_side": None if cube_id == 5 else push})
        stages = [(0, "supply_arrival", "READY_FOR_RGB_LOCALIZATION"),
                  (30, "B2_hold", "FORCE_CONFIRMED"),
                  (470, "B2_push_release", "PUSH_ARM_RETRACTED_SUPPORT_HOLDING"),
                  (479, "B2_support_unload", "RUNNING")]
        for offset, phase, status in stages:
            phases.append({"cube_id": cube_id, "phase": phase, "status": status,
                           "sim_time_s": (start + offset) / 120})
        for offset in range(500):
            step = start + offset
            phase = "A_lift" if offset < 10 else "B1_push" if offset < 20 else "B2_push"
            frame = {"physics_step": step, "time_s": step / 120, "phase": phase,
                     "left": {"tcp_gravity_corrected_world": [0.] * 6},
                     "right": {"tcp_gravity_corrected_world": [0.] * 6}}
            if phase == "A_lift":
                frame["left"]["tcp_gravity_corrected_world"] = [1., 0., -6., 0., 0., 0.]
                frame["right"]["tcp_gravity_corrected_world"] = [-1., 0., -6., 0., 0., 0.]
            elif phase == "B1_push":
                frame[support]["tcp_gravity_corrected_world"][1] = -2.
            elif offset <= 479:
                frame[support]["tcp_gravity_corrected_world"][1] = -2.
                if cube_id < 5:
                    frame[push]["tcp_gravity_corrected_world"][0] = 1.
            else:
                frame["phase"] = "B2_support_unload"
            if change:
                change(frame, cube_id, offset, support, push)
            if step != missing_step:
                frames.append(frame)
    with gzip.open(directory / "wrench.jsonl.gz", "wt") as stream:
        for frame in frames:
            stream.write(json.dumps(frame) + "\n")
    return {"phases": phases, "packing_sequence": sequence}


def test_alternating_support_roles_and_unload_exclusion(tmp_path):
    report = recording(tmp_path)
    result = force_checks(tmp_path, report)
    assert result["status"] == "PASS", result["issues"]
    assert [cube["support_side"] for cube in result["cubes"]] == ["left", "right", "left", "right", "left"]
    assert all(cube["support_held"]["samples"] == 450 for cube in result["cubes"])
    assert all(cube["support_held"]["contact_coverage"] == 1 for cube in result["cubes"])


def test_mirrored_support_dropout_cannot_hide_in_high_coverage(tmp_path):
    def dropout(frame, cube_id, offset, support, push):
        if cube_id == 2 and 100 <= offset < 107:
            frame[support]["tcp_gravity_corrected_world"][1] = 0.

    result = force_checks(tmp_path, recording(tmp_path, dropout))
    mirrored = result["cubes"][1]["support_held"]
    assert mirrored["contact_coverage"] > .98
    assert mirrored["longest_contact_dropout_s"] == pytest.approx(7 / 120)
    assert mirrored["status"] == "FAIL"


def test_supported_cube_overload_is_not_valid_contact(tmp_path):
    def overload(frame, cube_id, offset, support, push):
        if cube_id == 4 and 100 <= offset < 120:
            frame[support]["tcp_gravity_corrected_world"][1] = -6.1

    result = force_checks(tmp_path, recording(tmp_path, overload))
    stats = result["cubes"][3]["support_held"]
    assert stats["contact_coverage"] == 1
    assert stats["normal_force_in_band_proportion"] < .98
    assert stats["status"] == "FAIL"


def test_opposing_transport_forces_cannot_cancel_in_sum(tmp_path):
    def internal(frame, cube_id, offset, support, push):
        if cube_id == 1 and offset == 2:
            frame["left"]["tcp_gravity_corrected_world"][:3] = [20., 0., 0.]
            frame["right"]["tcp_gravity_corrected_world"][:3] = [-20., 0., 0.]

    result = force_checks(tmp_path, recording(tmp_path, internal))
    transport = result["cubes"][0]["stage_peaks"]["A"]
    assert transport["force_norm_max_N"] == {"left": 20., "right": 20.}
    assert transport["internal_force_norm_max_N"] == 20.
    assert transport["status"] == "FAIL"


@pytest.mark.parametrize("vector", ([8., 8., 0., 0., 0., 0.], [0., 0., 0., 0., 1.1, 0.]))
def test_b2_resultant_and_tcp_torque_limits(tmp_path, vector):
    def overload(frame, cube_id, offset, support, push):
        if cube_id == 3 and offset == 100:
            frame[push]["tcp_gravity_corrected_world"] = vector

    result = force_checks(tmp_path, recording(tmp_path, overload))
    assert result["cubes"][2]["stage_peaks"]["B2"]["status"] == "FAIL"


def test_invalid_wrench_and_missing_tick_are_not_ignored(tmp_path):
    def invalid(frame, cube_id, offset, support, push):
        if cube_id == 5 and offset == 50:
            frame[support]["tcp_gravity_corrected_world"][0] = float("nan")

    result = force_checks(tmp_path, recording(tmp_path, invalid, missing_step=1220))
    assert result["status"] == "FAIL"
    assert any("1 invalid wrenches and 1 120 Hz clock errors" in issue for issue in result["issues"])
    assert result["cubes"][2]["support_held"]["status"] == "FAIL"


def test_missing_role_or_held_endpoint_cannot_pass(tmp_path):
    report = recording(tmp_path)
    del report["packing_sequence"][1]["support_side"]
    report["phases"] = [row for row in report["phases"]
                        if not (row["cube_id"] == 4 and row["phase"] == "B2_support_unload")]
    result = force_checks(tmp_path, report)
    assert result["status"] == "FAIL"
    assert result["cubes"][1]["support_held"]["status"] == "FAIL"
    assert result["cubes"][3]["support_held"]["status"] == "FAIL"
