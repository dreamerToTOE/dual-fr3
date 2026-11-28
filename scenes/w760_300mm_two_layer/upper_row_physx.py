"""Run five upper-layer boxes in the W760 high-wall tray using side suction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from source_hooks import replace_once
from upper_central_physx import build as build_central


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def row_guide_paths():
    from upper_slot4_candidate import GUIDE as slot4_guide

    guides = {slot: HERE / f"upper_row_runtime_slot{slot}.json" for slot in range(1, 6)}
    guides[3] = HERE / "upper_row_runtime_slot3_grip39_forward3_candidate.json"
    guides[4] = slot4_guide
    return guides


def bevel_support_top(stage, prim, size, bevel):
    from pxr import Gf, UsdGeom, UsdPhysics

    inset = bevel / size
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    layers = ((.5, -.5), (.5, .5 - inset), (.5 - inset, .5))
    mesh = UsdGeom.Mesh.Define(stage, prim.GetPath())
    mesh.CreatePointsAttr([Gf.Vec3f(width * x, width * y, z)
                           for width, z in layers for x, y in corners])
    faces = [(3, 2, 1, 0), (8, 9, 10, 11)]
    for row in (0, 4):
        for index in range(4):
            next_index = (index + 1) % 4
            faces.append((row + index, row + next_index, row + 4 + next_index, row + 4 + index))
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateExtentAttr([Gf.Vec3f(-.5), Gf.Vec3f(.5)])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")


def build(source):
    source = build_central(source)
    source = replace_once(source, '        world.reset()', '''        # Stack stabilization reduces apparent payload weight during slow, force-controlled transport.
        physics_scene = next(prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene))
        scene_settings = PhysxSchema.PhysxSceneAPI.Apply(physics_scene)
        scene_settings.CreateEnableStabilizationAttr(False)
        world.reset()
        report["physx_stabilization_enabled"] = scene_settings.GetEnableStabilizationAttr().Get()
        if report["physx_stabilization_enabled"] is not False:
            raise RuntimeError("Native upper-row force measurements require full payload gravity")''')
    source = replace_once(source, '_300mm_second_layer_single_side', '_300mm_second_layer_five_cube')
    source = replace_once(source, 'W760 300mm walls, one upper cube, five restored dynamic lower cubes, side suction',
                          'W760 300mm walls, five upper boxes, measured lower rear row and one center support')
    source = replace_once(source, '''        "checkpoint": {"restored_lower_ids": [1, 2, 3, 4, 5], "additional_lower_rows": 0,
                       "active_cube_path": "/World/Cube_6", "support_path": "/World/Cube_5"},''',
                          '''        "checkpoint": {"restored_lower_ids": [1, 2, 3, 4, 5],
                       "extra_lower_support_id": 6, "starting_slot": START_SLOT,
                       "fixture_upper_slots": CHECKPOINT.get("fixture_upper_slots", [])},''')
    source = replace_once(source, '        CONFIG = row_config(CONFIG)', '''        CONFIG = row_config(CONFIG)
        CONFIG["cube_sequence"].extend({"cube_id": i, "marker_id": i + 22} for i in range(6, 12))
        CONFIG["row_colors_rgb"].extend([(0.53, 0.43, 0.72), (0.89, 0.47, 0.27), (0.36, 0.66, 0.83),
                                         (0.92, 0.76, 0.40), (0.50, 0.72, 0.45), (0.78, 0.60, 0.82)])''')
    source = replace_once(source, '        active_cube_path = "/World/Cube_6"', '''
        active_cube_path = f"/World/Cube_{6 + START_SLOT}"'''.lstrip("\n"))
    source = replace_once(source, 'CONFIG["vision"]["aruco_marker_size_m"], 28)',
                          'CONFIG["vision"]["aruco_marker_size_m"], 28 + START_SLOT)')
    source = replace_once(source, '        lower_bodies = {}', '''        if CUBE_CONTACT_OFFSET_M is not None:
            report["cube_contact_offset_m"] = CUBE_CONTACT_OFFSET_M
            PhysxSchema.PhysxCollisionAPI(stage.GetPrimAtPath(active_cube_path)).CreateContactOffsetAttr(
                CUBE_CONTACT_OFFSET_M)
        if UPPER_REST_OFFSET_M is not None:
            report["upper_rest_offset_m"] = UPPER_REST_OFFSET_M
            PhysxSchema.PhysxCollisionAPI(stage.GetPrimAtPath(active_cube_path)).GetRestOffsetAttr().Set(
                UPPER_REST_OFFSET_M)
        lower_bodies = {}''')
    source = replace_once(source, '            w, x, y, z = saved["orientation_wxyz"]', '''            if CUBE_CONTACT_OFFSET_M is not None:
                PhysxSchema.PhysxCollisionAPI(lower_prim).CreateContactOffsetAttr(CUBE_CONTACT_OFFSET_M)
            if index <= 6 and SUPPORT_TOP_BEVEL_M is not None:
                bevel_support_top(stage, lower_prim, CONFIG["cube_size_m"], SUPPORT_TOP_BEVEL_M)
                report["support_top_bevel_m"] = SUPPORT_TOP_BEVEL_M
            if index > 6 and UPPER_REST_OFFSET_M is not None:
                PhysxSchema.PhysxCollisionAPI(lower_prim).GetRestOffsetAttr().Set(UPPER_REST_OFFSET_M)
            w, x, y, z = saved["orientation_wxyz"]''')
    source = replace_once(source, '''        lower_bodies = {}
        for saved in GUIDE["lower_row_poses"]:''', '''        lower_bodies = {}
        upper_bodies = {}
        for saved in INITIAL_BOXES:''')
    source = replace_once(source, '''            lower_bodies[str(index)] = world.scene.add(SingleRigidPrim(path, name=f"lower_cube_{index}"))
        cube = world.scene.add(SingleRigidPrim(active_cube_path, name="task_cube"))''',
                          '''            storage = lower_bodies if index <= 6 else upper_bodies
            storage[str(index)] = world.scene.add(SingleRigidPrim(path, name=f"initial_cube_{index}"))
        cube = world.scene.add(SingleRigidPrim(active_cube_path, name="task_cube"))
        upper_bodies[str(6 + START_SLOT)] = cube''')
    source = replace_once(source, '                    supported_pose = vision_estimator.resting_pose(observation, CONFIG["floor_top_m"])',
                          '                    supported_pose = vision_estimator.resting_pose(observation, SUPPORT_TOP)')
    source = replace_once(source, '                        pose = vision_estimator.resting_pose(result, CONFIG["floor_top_m"])',
                          '''                        observation_plane = CONFIG["floor_top_m"] if index < 6 else SUPPORT_TOP
                        pose = vision_estimator.resting_pose(result, observation_plane)''')
    source = replace_once(source, '''            approach = {side: cartesian_joint_target(
                            side, grasp_seeds[side], supply + (half + .035) * (supply_rotation @ normal),
                            desired_rotation, retarget_guide=True)''',
                          '''            face_centers = {side: supply + supply_rotation @ GUIDE["grasp_tcp_offsets_world"][side]
                            for side in robots}
            approach = {side: cartesian_joint_target(
                            side, grasp_seeds[side], face_centers[side] + .035 * (supply_rotation @ normal),
                            desired_rotation, retarget_guide=True)''')
    source = replace_once(source, '''            approach_targets = {side: supply + (half + .007) * (supply_rotation @ normal)
                                for side, normal in face_normals.items()}''',
                          '''            approach_targets = {
                side: supply + supply_rotation @ (np.asarray(GUIDE["grasp_tcp_offsets_world"][side]) + .007 * normal)
                for side, normal in face_normals.items()}''')
    source = replace_once(source, '''                    follow_side_guide(phase)
                    if phase["phase"] == "high_transfer_to_rear":
                        hover_load = held_vertical_load()''',
                          '''                    if phase["phase"] == "vertical_descent":
                        hover_load = held_vertical_load()
                        centers = np.asarray(phase["cube_centers_m"], dtype=float).copy()
                        centers[:, 2] += np.linspace(0., .00045, len(centers))
                        phase = {**phase, "cube_centers_m": centers.tolist()}
                    follow_side_guide(phase)''')
    source = replace_once(source, '''                       and {row["collider0"], row["collider1"]} == {active_cube_path, "/World/Cube_5"})''',
                          '''                       and active_cube_path in (row["collider0"], row["collider1"])
                       and ({row["collider0"], row["collider1"]} - {active_cube_path}) <= LOWER_PATHS)''')
    grasp_log = '            report["target_observations"].append({"phase": "A_carry", "source": "encoder_fk_from_tactile_grasp"})'
    source = replace_once(source, grasp_log, '''            if current_task.get("slot") == 3:
                entry = report["upper_row"]["slot3_guide"]
                entry["rgb_fk_grasp_edge_margins_m"] = {
                    side: slot3_cup_edge_margin(tcp, rot, supply, supply_rotation)
                    for side, (tcp, rot) in tcps.items()}
                observed, orientation = cube.get_world_pose()
                entry["physics_reference_grasp_edge_margins_m"] = {
                    side: slot3_cup_edge_margin(tcp, rot, observed, quat_to_rot_matrix(orientation))
                    for side, (tcp, rot) in tcps.items()}
''' + grasp_log)
    start = source.index('        def finish_upper_placement(hover_load):')
    end = source.index('        def support_window():', start)
    source = source[:start] + (HERE / "upper_row_seating_body.txt").read_text() + "\n" + source[end:]
    source = replace_once(source, 'prefetched_cache=None, sampled_trajectory=None):',
                          'prefetched_cache=None, sampled_trajectory=None, sample_order=None, adjust_targets=None):')
    source = replace_once(source, '            for index in range(len(times)):\n                for side in sides:',
                          '            indices = range(len(times)) if sample_order is None else sample_order(len(times))\n'
                          '            for index in indices:\n'
                          '                commands = {side: samples[side][index] for side in sides}\n'
                          '                if adjust_targets is not None:\n'
                          '                    commands = adjust_targets(index, commands)\n'
                          '                for side in sides:')
    source = replace_once(source, '                    command_positions(side, samples[side][index], velocity=velocity)',
                          '                    command_positions(side, commands[side], velocity=velocity)')
    source = replace_once(source, '                    position[1] = y',
                          '                    position[0] = current_task.get("rear_to_side_hold_x_m", position[0])\n'
                          '                    position[1] = y')
    source = replace_once(source, '                if status_label is not None and tick % 12 == 0:',
                          '''                if contact_monitor.phase.startswith("upper_B_"):
                    watch = report.setdefault("upper_b_wrist_watch", {})
                    scope = {"physics_step": tick, "time_s": float(world.current_time),
                             "cube_id": current_task["cube_id"], "phase": contact_monitor.phase}
                    for arm in robots:
                        wrench = np.asarray(force_sample[arm]["tcp_gravity_corrected_world"])
                        force_N = float(np.linalg.norm(wrench[:3]))
                        torque_Nm = float(np.linalg.norm(wrench[3:]))
                        for key, value in (("peak_force_N", force_N), ("peak_torque_Nm", torque_Nm)):
                            if key not in watch or value > watch[key]["value"]:
                                watch[key] = {**scope, "arm": arm, "value": value}
                if "rear_to_side_hold_x_m" in current_task:
                    rear_bearing = any(pair["label"] == "wall_back" and pair["current_step_observed"]
                                       and (pair["normal_impulse_Ns"] or 0.) > 1e-5
                                       for pair in reference_contacts["pairs"])
                    if not rear_bearing or support_impulse() * 120 < 6.:
                        raise RuntimeError("Rear-to-side handoff lost a loaded rear or lower support")
                if status_label is not None and tick % 12 == 0:''')
    source = replace_once(source, 'for side in sides} if phase in ("A_transfer", "B2_push") else {})',
                          'for side in sides} if phase in ("A_transfer", "B2_push") or phase in '
                          '("upper_B_seam_lift", "upper_B_seam_cross", "upper_B_seam_land", '
                          '"upper_B_seam_rear_clearance") else {})')
    start = source.index('        def finish_upper_retreat(after_release):')
    end = source.index('    except Exception as exc:\n        report["status"]', start)
    source = source[:start] + (HERE / "upper_row_push_body.txt").read_text() + "\n" + (
        HERE / "upper_row_physx_body.txt").read_text() + source[end:]
    source = replace_once(source, '    return 0 if report["status"] in ("PASS_W760_UPPER_DUAL_SIDE", "PASS_W760_UPPER_RELEASE_RETREAT") else 1',
                          '    return 0 if report["status"] in ("PASS_W760_UPPER_ROW", "PARTIAL_W760_UPPER_ROW") else 1')
    start = source.index('            labels = {', source.index('        def capture_acceptance_view'))
    end = source.index('            from omni.kit.viewport.utility', start)
    source = source[:start] + '''            if (phase, state) == ("upper_row_complete", "PASS_W760_UPPER_ROW"):
                label = "upper_cube_final"
            elif (phase, state) in (("A_lift_and_align", "CLEAR_PHYSX_EXECUTION"),
                                   ("upper_slot_complete", "SEATED_STABLE"),
                                   ("upper_B_rear", "FORCE_CONFIRMED"), ("upper_B_side", "FORCE_CONFIRMED")):
                label = f"upper_{cube_id - 6}_{phase}"
            else:
                return

''' + source[end:]
    return replace_once(source, '''                        "cube_position_m": reference_center.tolist(),
                        "cube_rotation_world":''',
                        '''                        "cube_position_m": reference_center.tolist(),
                        "active_cube_id": current_task["cube_id"],
                        "cube_rotation_world":''')


def build_production():
    from upper_slot4_candidate import build as apply_slot4_guidance
    from upper_slot4_front_seal import inject as apply_front_seal

    from upper_fast_guide import inject as parallelize_guides

    source = build((REPO / "fr3_ab_visual_runner.py").read_text())
    return parallelize_guides(apply_front_seal(apply_slot4_guidance(source)))
