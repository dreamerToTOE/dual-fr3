import json
import math
from pathlib import Path

from fr3_contact_monitor import is_unexpected_robot_contact
from fr3_five_cube_row import row_config


def test_five_row_fits_and_contact_permission_tracks_active_body():
    base = json.loads((Path(__file__).resolve().parents[1] / "fr3_four_wall_scene_config.json").read_text())
    old_config = json.dumps(base)
    config = row_config(base)
    assert json.dumps(base) == old_config
    assert len(config["cube_sequence"]) == len(config["goal_cube_centers_m"]) == 5
    size = config["cube_size_m"]
    ordered = sorted(config["goal_cube_centers_m"])
    assert all(math.isclose(actual[0], expected, abs_tol=1e-12)
               for actual, expected in zip(config["goal_cube_centers_m"], [-.305, .305, -.155, .155, 0.]))
    walls = {name: (center, extent) for name, center, extent in config["static_boxes"]}
    left_face = walls["wall_left"][0][0] + walls["wall_left"][1][0] / 2
    right_face = walls["wall_right"][0][0] - walls["wall_right"][1][0] / 2
    assert math.isclose(right_face - left_face, 5 * size + .010)
    assert math.isclose(ordered[2][0] - ordered[1][0] - size, .005)
    assert math.isclose(ordered[3][0] - ordered[2][0] - size, .005)
    for first, second in zip(ordered, ordered[1:]):
        assert second[0] - first[0] >= size - 1e-10
    assert [item["push_direction"] for item in config["cube_sequence"]] == [-1, 1, -1, 1, 0]
    assert [item["landing_x_m"] for item in config["cube_sequence"]] == [0., .005, -.005, .145, 0.]
    assert config["task"]["phase_B2"]["rail_assist"]["base_y_m"] == .8
    assert config["cube_sequence"][0]["approach_standoff_m"] == .03
    assert config["cube_sequence"][-1]["push_side"] is None
    cup = "/World/left_fr3/fr3_link8/ABSideTool/cup_r1_c1"
    assert not is_unexpected_robot_contact(cup, "/World/Cube_2", {"left"}, -.0005, "/World/Cube_2")
    assert is_unexpected_robot_contact(cup, "/World/Cube_1", {"left"}, -.0005, "/World/Cube_2")
    assert is_unexpected_robot_contact(cup, "/World/Cube_2", {"left"}, -.004, "/World/Cube_2")


def test_neighbor_contact_tracker_keeps_other_bodies_distinct():
    from fr3_contact_monitor import CubeContactTracker

    tracker = CubeContactTracker("/World/Cube_3")
    tracker.update("/World/Cube_1", "/World/Cube_3", event_type=0, physics_step=1,
                   separation_m=-.0001, normal_impulse_Ns=.001)
    tracker.update("/World/Cube_2", "/World/Environment/floor", event_type=0, physics_step=1,
                   separation_m=-.0001, normal_impulse_Ns=.001)
    frame = tracker.snapshot(1)
    assert frame["loaded_contacts"]["cube_neighbor"]
    assert len(frame["pairs"]) == 1


def test_b2_mirror_preserves_role_paths_and_is_reversible():
    import numpy as np
    import pytest
    from fr3_bilateral_b2 import b2_role_seeds, mirror_joints

    plan = json.loads((Path(__file__).resolve().parents[1] / "validation/fr3_b2_collision_plan.json").read_text())
    right = b2_role_seeds(plan, "right")
    left = b2_role_seeds(plan, "left")
    assert left["support_side"] == "right"
    for phase in ("high_transit", "descent", "approach", "push", "support"):
        np.testing.assert_allclose(mirror_joints(left[phase]), right[phase])
    assert left["push_base_contact_m"][0] == -right["push_base_contact_m"][0]
    assert left["push_base_contact_m"][3] == 0.
    assert left["cube_final_m"][0] == -right["cube_final_m"][0]
    assert left["collision_validation"] == "replan_and_validate_in_current_scene"
    with pytest.raises(ValueError):
        mirror_joints(np.zeros(6))


def test_row_acceptance_rejects_five_b1_pushes_without_supply_transport():
    from fr3_row_acceptance import sequence_checks

    report = {
        "status": "PASS_FIVE_CUBE_SENSOR_SEQUENCE", "attachment_model": "Isaac_Sim_Surface_Gripper",
        "surface_gripper_validated": True, "physx_contacts": {"unexpected_contact_count": 0},
        "completed_cubes": [{"cube_id": value} for value in range(1, 6)],
        "packing_sequence": [{"push_direction": value, "goal_center_m": [value * .175, 1.076, .787]}
                             for value in (-1, 1, -1, 1, 0)],
        "phases": [{"cube_id": value, "phase": "B1_push", "status": "REACHED_WITH_PHYSX_CONTACT"}
                   for value in range(1, 6)],
    }
    issues, starts = sequence_checks(report)
    assert not starts
    assert any("A_attach" in issue for issue in issues)
    assert any("A_landing" in issue for issue in issues)
    assert any("B2_push_release" in issue for issue in issues)


def test_row_sequence_accepts_complete_transport_then_alternating_pushing():
    from fr3_row_acceptance import sequence_checks

    report = {
        "status": "PASS_FIVE_CUBE_SENSOR_SEQUENCE", "attachment_model": "Isaac_Sim_Surface_Gripper",
        "surface_gripper_validated": True, "physx_contacts": {"unexpected_contact_count": 0},
        "completed_cubes": [{"cube_id": value} for value in range(1, 6)],
        "packing_sequence": [{"push_direction": value, "goal_center_m": [value * .175, 1.076, .787]}
                             for value in (-1, 1, -1, 1, 0)],
        "phases": [],
    }
    for cube_id in range(1, 6):
        stages = [
            ("supply_arrival", "READY_FOR_RGB_LOCALIZATION", [0, -.3, .787]),
            ("A_attach", "SURFACE_GRIPPERS_CLOSED", [0, -.3, .787]),
            ("A_lift", "REACHED_WITH_SURFACE_GRIPPERS", [0, -.3, 1.107]),
            ("A_over_front", "REACHED_WITH_SURFACE_GRIPPERS", [0, .4, 1.107]),
            ("A_landing", "REACHED_WITH_SURFACE_GRIPPERS", [0, .736, .787]),
            ("A_release", "REACHED_AFTER_SUCTION_RELEASE", [0, .736, .787]),
            ("B1_push", "REACHED_WITH_PHYSX_CONTACT", [0, 1.076, .787]),
            ("B2_hold", "FORCE_CONFIRMED", [0, 1.076, .787]),
        ]
        if cube_id < 5:
            stages.append(("B2_high_approach", "RUNNING", [0, 1.076, .787]))
        stages.append(("B2_push_release", "CENTER_CUBE_STRAIGHT_PUSH_COMPLETE" if cube_id == 5
                       else "REACHED_WITH_PHYSX_CONTACT", [0, 1.076, .787]))
        for index, (phase, status, position) in enumerate(stages):
            report["phases"].append({"cube_id": cube_id, "phase": phase, "status": status,
                                     "sim_time_s": cube_id * 200 + index, "cube_position_m": position,
                                     "direction": -1 if cube_id in (1, 3) else 1, "progress_m": .175})
    issues, starts = sequence_checks(report)
    assert issues == []
    assert len(starts) == 5
    report["prepared_scene_conditions"] = {"cubes": [{"cube_id": 1}]}
    assert "the run included pre-staged Cubes" in sequence_checks(report)[0]


def test_final_row_rechecks_earlier_dynamic_cubes_after_last_insertion():
    from copy import deepcopy
    from fr3_row_acceptance import final_row_checks

    centers = [-.345, .345, -.195, .195, 0.]
    rows = [{"cube_id": index + 1, "position_m": [x, 1.076, .787],
             "rotation_world": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
            for index, x in enumerate(centers)]
    report = {"final_row_observations": {"time_s": 1001., "source": "rear_RGB_marker_support_plane",
                                         "cubes": rows},
              "completed_cubes": [{"cube_id": row["cube_id"], "target_center_m": row["position_m"].copy()}
                                  for row in rows],
              "phases": [{"phase": "B2_push_release", "sim_time_s": 1000.}]}
    assert final_row_checks(report) == []
    moved = deepcopy(report)
    moved["final_row_observations"]["cubes"][0]["position_m"][1] -= .02
    assert any("Cube_1 final row position" in issue for issue in final_row_checks(moved))
    stale = deepcopy(report)
    stale["final_row_observations"]["time_s"] = 999.
    assert any("last release" in issue for issue in final_row_checks(stale))
    del report["final_row_observations"]
    assert final_row_checks(report)


def test_only_current_active_target_impulse_during_paused_seat_counts(tmp_path):
    import gzip
    import json

    from fr3_row_acceptance import physical_contact_checks

    def contact(first, second):
        return {"collider0": first, "collider1": second,
                "current_step_observed": True, "normal_impulse_Ns": .04}

    targets = ["/World/Environment/wall_left", "/World/Environment/wall_right",
               "/World/Cube_1", "/World/Cube_2", "/World/Environment/wall_back"]
    report = {"phases": [], "simulation_clock": {"control_steps": 300}}
    starts = [1, 61, 121, 181, 241]
    frames = []
    for cube_id, (start, target) in enumerate(zip(starts, targets), start=1):
        active = f"/World/Cube_{cube_id}"
        probe = [6., 3., 3.] if cube_id <= 2 else [8.] * 30
        stages = [(0, "supply_arrival", "READY_FOR_RGB_LOCALIZATION"),
                  (2, "A_attach", "SURFACE_GRIPPERS_CLOSED"),
                  (5, "A_release", "RELEASING_SURFACE_GRIPPERS"),
                  (7, "B1_touch", "TACTILE_APPROACH"),
                  (10, "B2_hold", "FORCE_CONFIRMED" if cube_id == 5 else "MAINTAINED_DURING_APPROACH")]
        if cube_id < 5:
            stages.extend([(12, "B2_contact_sync", "FORCE_CONFIRMED"),
                           (13, "B2_push_seat", "RUNNING"),
                           (44, "B2_push_seat", "FORCE_CONFIRMED"),
                           (44, "B2_push_unload", "RUNNING")])
        for offset, phase, status in stages:
            report["phases"].append({"cube_id": cube_id, "phase": phase, "status": status,
                                     "sim_time_s": (start + offset) / 120,
                                     "probe_normal_samples_N": probe})
        contacts = {1: [contact(active, "/World/Environment/supply")],
                    9: [contact(active, "/World/Environment/wall_back")]}
        if cube_id < 5:
            loaded_offsets = (42, 44) if cube_id <= 2 else range(15, 45)
            contacts.update({offset: [contact(active, target)] for offset in loaded_offsets})
        for offset in range(60):
            frames.append({"physics_step": start + offset,
                           "attachments": {"left": offset == 4, "right": offset == 4},
                           "cube_contacts": {"valid": True, "pairs": contacts.get(offset, [])}})

    directory = tmp_path / "contact_states"
    directory.mkdir()

    def audit():
        with gzip.open(directory / "reference.jsonl.gz", "wt") as stream:
            stream.writelines(json.dumps(row) + "\n" for row in frames)
        return physical_contact_checks(tmp_path, report, starts)

    issues, evidence = audit()
    assert not issues
    assert [row["target_impulse_frames"] for row in evidence] == [2, 2, 30, 30, 1]

    frames[120 + 14]["cube_contacts"]["pairs"] = [contact("/World/Cube_3", "/World/Cube_1")]
    frames[120 + 15]["cube_contacts"]["pairs"] = [contact("/World/Cube_3", "/World/Cube_2"),
                                                   contact("/World/Cube_4", "/World/Cube_1")]
    frames[120 + 16]["cube_contacts"] = {"valid": False, "pairs": [contact("/World/Cube_3", "/World/Cube_1")]}
    frames[42]["cube_contacts"]["pairs"] = [contact("/World/Cube_1", "/World/Environment/wall_right")]
    frames[44]["cube_contacts"]["pairs"] = []
    issues, evidence = audit()
    assert issues == ["Cube_1 loaded the intended target in 0 of 1 required confirmation frames",
                      "Cube_3 loaded the intended target in 28 of 30 required confirmation frames"]
    assert evidence[0]["target_impulse_frames"] == 0
    assert evidence[2]["target_impulse_frames"] == 28
