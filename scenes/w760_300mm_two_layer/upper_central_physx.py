"""Run the dual-side-suction placement of one upper cube in Isaac Sim."""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path
import sys

from source_hooks import replace_once
from continuous_side_capture import patch_capture
from continuous_side_row import patch_a


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_release_checkpoint(directory):
    report = json.loads((directory / "runner_report.json").read_text())
    assert report["error"] == "Lower Cube_5 did not carry the upper box after both grippers opened"
    assert report["physx_contacts"]["unexpected_contact_count"] == 0
    release = next(row for row in report["phases"] if row["phase"] == "upper_release")
    release_tick = round(120 * release["sim_time_s"])
    with gzip.open(directory / "contact_states/reference.jsonl.gz", "rt") as stream:
        samples = [row for line in stream if (row := json.loads(line))["physics_step"] > release_tick]
    assert len(samples) == 60 and all(not any(row["attachments"].values()) for row in samples)
    assert all(any(pair["current_step_observed"] and (pair["normal_impulse_Ns"] or 0.) > 1e-5
                   and {pair["collider0"], pair["collider1"]} == {"/World/Cube_5", "/World/Cube_6"}
                   for pair in row["cube_contacts"]["pairs"]) for row in samples)
    with gzip.open(directory / "wrench.jsonl.gz", "rt") as stream:
        last = json.loads(deque(stream, maxlen=1)[0])
    assert last["physics_step"] == release_tick + 60
    assert max(abs(vel) for side in ("left", "right") for vel in last[side]["qdot_rad_s"]) < .01
    return {
        "source": str(directory), "position_m": report["stopped_cube_position_m"],
        "orientation_wxyz": report["stopped_cube_orientation_wxyz"], "control_pose": report["control_pose"],
        "q_rad": {side: last[side]["q_rad"] for side in ("left", "right")},
        "neighbors": report["neighbor_observations"][-1]["neighbors"],
        "target_m": next(row["target_m"] for row in report["phases"] if row["phase"] == "upper_supply"),
    }


def build(source):
    source = patch_a(source)
    source = replace_once(source, '_300mm_first_layer_continuous_side', '_300mm_second_layer_single_side')
    source = replace_once(source, '        "first_layer_side_only": True, "visual_mode": not args.headless,',
                          '        "upper_layer_side_only": True, "visual_mode": not args.headless,')
    source = replace_once(source, '''        "continuous_run": {"source": "one timeline, original supply spawn", "restored_cube_ids": [],
                           "side_guides": SIDE_GUIDE_NAMES},''',
                          '''        "checkpoint": {"restored_lower_ids": [1, 2, 3, 4, 5], "additional_lower_rows": 0,
                       "active_cube_path": "/World/Cube_6", "support_path": "/World/Cube_5"},''')
    source = replace_once(source, 'W760 300mm wall continuous five-cube row; opposing vertical face suction only',
                          'W760 300mm walls, one upper cube, five restored dynamic lower cubes, side suction')
    source = replace_once(source, '        active_cube_path = "/World/Cube_1" if args.five_cube_row else "/World/Cube"\n',
                          '        active_cube_path = "/World/Cube_6"\n')
    source = replace_once(source, 'UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.35, 0.05)])',
                          '''UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(0.97, 0.48, 0.25)])
        if RELEASE_CHECKPOINT is not None:
            xform = UsdGeom.Xformable(prim)
            translate = next(op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            translate.Set(Gf.Vec3d(*RELEASE_CHECKPOINT["position_m"]))
            w, x, y, z = RELEASE_CHECKPOINT["orientation_wxyz"]
            xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))''')
    source = replace_once(source, 'CONFIG["vision"]["aruco_marker_size_m"], CONFIG["cube_sequence"][0]["marker_id"])',
                          'CONFIG["vision"]["aruco_marker_size_m"], 28)')
    source = replace_once(source, '        cube = world.scene.add(SingleRigidPrim(active_cube_path, name="task_cube"))',
                          '''        lower_bodies = {}
        for saved in GUIDE["lower_row_poses"]:
            index = saved["cube_id"]
            path = f"/World/Cube_{index}"
            lower_prim = _box(stage, path, saved["position_m"], [CONFIG["cube_size_m"]] * 3,
                              collision=True, dynamic=True, mass=CONFIG["physics"]["cube_mass_kg"])
            w, x, y, z = saved["orientation_wxyz"]
            UsdGeom.Xformable(lower_prim).AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            UsdGeom.Gprim(lower_prim).CreateDisplayColorAttr([Gf.Vec3f(*CONFIG["row_colors_rgb"][index - 1])])
            add_cube_marker(stage, path, CONFIG["cube_size_m"], CONFIG["vision"]["aruco_marker_size_m"],
                            CONFIG["cube_sequence"][index - 1]["marker_id"])
            lower_bodies[str(index)] = world.scene.add(SingleRigidPrim(path, name=f"lower_cube_{index}"))
        cube = world.scene.add(SingleRigidPrim(active_cube_path, name="task_cube"))''')
    start = source.index('            record("A_release", "RELEASING_SURFACE_GRIPPERS")', source.index("        def run_cube():"))
    keep = source.index('        placed_bodies = [cube]\n', start)
    source = source[:start] + '            finish_upper_placement(hover_load)\n' + source[keep:]
    source = replace_once(source, '''                if phase["carrying"]:
                    follow_side_guide(phase)
            finish_upper_placement(hover_load)''',
                          '''                if phase["carrying"]:
                    follow_side_guide(phase)
                    if phase["phase"] == "high_transfer_to_rear":
                        hover_load = held_vertical_load()
            finish_upper_placement(hover_load)''')
    source = source.replace('        placed_bodies = [cube]\n', '', 1)
    start = source.index('        sequence = CONFIG.get("cube_sequence", [current_task])')
    end = source.index('    except Exception as exc:\n        report["status"]', start)
    source = source[:start] + (HERE / "upper_central_physx_body.txt").read_text() + source[end:]
    source = replace_once(source, '    return 0 if report["status"] in ("PASS_FIVE_CUBE_SENSOR_SEQUENCE", "PASS_A_B1_B2_SENSOR_CONTROL") else 1',
                          '    return 0 if report["status"] in ("PASS_W760_UPPER_DUAL_SIDE", "PASS_W760_UPPER_RELEASE_RETREAT") else 1')
    source = patch_capture(source)
    start = source.index('            if cube_id in (2, 4)')
    end = source.index('            from omni.kit.viewport.utility', start)
    source = source[:start] + '''            labels = {
                ("A_lift_and_align", "CLEAR_PHYSX_EXECUTION"): "upper_dual_side_lift",
                ("A_high_transfer_to_rear", "CLEAR_PHYSX_EXECUTION"): "upper_over_rear_support",
                ("upper_support_confirmed", "FORCE_TRANSFERRED"): "upper_load_transfer_before_release",
                ("upper_stacking_complete", "PASS_W760_UPPER_DUAL_SIDE"): "upper_cube_final",
                ("upper_stacking_complete", "PASS_W760_UPPER_RELEASE_RETREAT"): "upper_cube_final",
            }
            label = labels.get((phase, state))
            if label is None:
                return

''' + source[end:]
    return replace_once(source, 'if label == "five_cube_final_row":',
                        'if label == "upper_cube_final":')
