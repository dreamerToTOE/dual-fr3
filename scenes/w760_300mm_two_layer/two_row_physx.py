"""Pack five new floor boxes against a measured rear row, then five upper boxes in one Kit."""
import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path

# Share the tested first-layer guides and planning helpers.
sys.path.append(str(Path(__file__).resolve().parent.parent / "w760_300mm"))

from source_hooks import replace_once
from upper_row_physx import HERE, REPO, build_production, row_guide_paths

FRONT_IDS = (12, 15, 13, 14, 6)
REAR_IDS = tuple(range(1, 6))
LOWER_IDS = REAR_IDS + FRONT_IDS
SCENE = "W760_H300_rear5_front5_upper5_side_v1"


def build():
    from front_row_a import inject as add_floor_a
    from front_row_skills import inject as add_floor_b

    source = add_floor_b(add_floor_a(build_production()))
    source = replace_once(source, 'range(6, 12)', 'range(6, 16)')
    source = replace_once(source, '    app = SimulationApp(', '    wall_started = time.monotonic()\n    app = SimulationApp(')
    source = replace_once(source, 'item = {"phase": name, "status": state, "sim_time_s": tick / 120,',
                          'item = {"phase": name, "status": state, "sim_time_s": tick / 120, '
                          '"wall_elapsed_s": time.monotonic() - wall_started,')
    source = replace_once(source, '(0.78, 0.60, 0.82)])', '''(0.78, 0.60, 0.82),
                                         *(CONFIG["row_colors_rgb"][i] for i in (0, 2, 3, 1))])''')
    source = replace_once(source, 'UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(0.97, 0.48, 0.25)])',
                          'UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*CONFIG["row_colors_rgb"][FIRST_ID - 1])])')
    source = replace_once(source, 'CONFIG["scene_id"] += "_300mm_second_layer_five_cube"',
                          'CONFIG["scene_id"] = SCENE')
    source = replace_once(source, '"extra_lower_support_id": 6,', '"lower_front_slot_order": list(FRONT_IDS),')
    source = replace_once(source, 'five upper boxes, measured lower rear row and one center support',
                          'measured lower rear row, five sequential lower front boxes, five upper rear boxes')
    source = replace_once(source, 'active_cube_path = f"/World/Cube_{6 + START_SLOT}"',
                          'active_cube_path = f"/World/Cube_{FIRST_ID}"')
    source = replace_once(source, 'CONFIG["vision"]["aruco_marker_size_m"], 28 + START_SLOT)',
                          'CONFIG["vision"]["aruco_marker_size_m"], FIRST_ID + 22)')
    source = replace_once(source, 'storage = lower_bodies if index <= 6 else upper_bodies',
                          'storage = lower_bodies if index in LOWER_IDS else upper_bodies')
    source = replace_once(source, 'upper_bodies[str(6 + START_SLOT)] = cube',
                          'if START_FRONT == 6:\n            upper_bodies[str(FIRST_ID)] = cube')
    source = replace_once(source, 'if index < 6 else SUPPORT_TOP',
                          'if index + 1 in LOWER_IDS else CONFIG["floor_top_m"] + CONFIG["cube_size_m"]')
    source = replace_once(source, '        SUPPORT_TOP = CONFIG["floor_top_m"] + CONFIG["cube_size_m"]',
                          (HERE / 'two_row_front_loop.txt').read_text() +
                          '\n        SUPPORT_TOP = CONFIG["floor_top_m"] + CONFIG["cube_size_m"]')
    source = replace_once(source, 'for index in range(1, 7)}', 'for index in LOWER_IDS}')
    source = replace_once(source, '                                target = guided_tcp + (1. - u[index]) * (measured_tcp - guides_start[side])',
                          '''                                target = guided_tcp + (1. - u[index]) * (measured_tcp - guides_start[side])
                                if current_task["cube_id"] == 10 and name in ("A_withdraw_laterally", "A_withdraw_up"):
                                    # Clear the right wrist past the measured U2 corner after releasing U4.
                                    target[1] -= .003 * u[index]''')
    source = replace_once(source, '                else:\n                    outer_support = f"Cube_{slot}"',
                          '''                elif slot in (3, 4):
                    seam_x = float(final[0] - push_sign * .024)
                else:
                    outer_support = f"Cube_{slot}"''')
    source = replace_once(source, '                if push_sign * (entry_x - center[0]) > .00025:',
                          '''                if slot in (1, 3, 4) or extra:
                    # Lift before the next support edge takes load from the grounded push.
                    entry_x = float(center[0])
                    if not 0. < push_sign * (exit_x - entry_x) <= .14:
                        raise RuntimeError(f"Upper {slot}: early seam transfer starts outside the checked corridor")
                elif slot == 2 and not extra:
                    # Keep the front-row center box clear before withdrawing from the rear wall.
                    front, front_rotation = next((position, attitude) for name, position, attitude in lower_obstacles
                                                 if name.rsplit("/", 1)[-1] == "Cube_6")
                    front_edge = front[0] + half * np.abs(front_rotation[0]).sum()
                    clear_x = front_edge + half * np.abs(rotation[0]).sum()
                    if seam_x <= clear_x:
                        raise RuntimeError("Upper 2: no clear entry between the front and rear support edges")
                    entry_x = (clear_x + seam_x) / 2
                    record(approach_phase, "FRONT_SUPPORT_CLEARANCE_TARGET", entry_x_m=float(entry_x),
                           window_x_m=[float(clear_x), float(seam_x)])
                if push_sign * (entry_x - center[0]) > .00025:''')
    source = replace_once(source, 'if extra else 2.75\n                    translate_attached(across',
                          'if slot in (1, 2, 3, 4) else 2.75\n                    translate_attached(across')
    source = replace_once(source, '                lift_height = .002',
                          '''                # The rear-edge contact probe cleared at 8 mm actual height.
                lift_height = .002 if slot == 1 else .010''')
    source = replace_once(source, '                rear_clearance = .0006',
                          '                rear_clearance = 0. if side_first else .0006')
    source = replace_once(source, 'translate_attached(lifted, .75, motion_prefix + "_lift")',
                          'translate_attached(lifted, .75 * (lift_height - .0005) / .0015, motion_prefix + "_lift")')
    source = replace_once(source, 'translate_attached(cleared_rear + [0., 0., .0005], .4, motion_prefix + "_lift")',
                          'translate_attached(cleared_rear + [0., 0., .0005], .8 if slot in (3, 4) else .4, motion_prefix + "_lift")')
    source = replace_once(source, 'translate_attached(landed + [0., 0., .00045], .875, motion_prefix + "_land")',
                          'translate_attached(landed + [0., 0., .00045],\n'
                          '                                       .875 * (lift_height - .00045) / .00155, motion_prefix + "_land")')
    source = replace_once(source, '                        if airborne and support_impulse() > 1e-5:',
                          '''                        if airborne and support_impulse() > 1e-5:
                            record(motion_phase, "NATIVE_SUPPORT_CONTACT_IN_TRANSFER",
                                   lower_snapshot=lower_poses(), native_pairs=contact_monitor.cube_contact_snapshot()["pairs"])''')
    start = source.index('                balance_max_m = .00045')
    end = source.index('            def approach_front()', start)
    source = source[:start] + source[start:end].replace('slot == 2', 'slot in (2, 3, 4)') + source[end:]
    source = replace_once(source, '            contact_plan_rotation = approach_front()',
                          '''            receipt = (CHECKPOINT.get("landed") or {}).get("seam_entry_resume", {})
            completed_transport = receipt.get("completed_transport") if slot == START_SLOT else None
            contact_plan_rotation = approach_front()''')
    source = replace_once(source, '                        prepare_rear_to_side_hold()\n                        approach_side()',
                          '                        preload = 3.25 if completed_transport in (1, 2) else 1.25\n'
                          '                        prepare_rear_to_side_hold(preload)\n                        approach_side()')
    source = replace_once(source, '                side_offset[2] = .010\n                side_rotation',
                          '''                side_offset[2] = .010
                final_u2_resume = slot == 2 and completed_transport == 2
                if final_u2_resume:
                    side_offset[2] = -.018
                side_rotation''')
    source = replace_once(source, '                if slot == 3:\n                    # Both side-facing wrists',
                          '                if slot == 3 or final_u2_resume:\n                    # Both side-facing wrists')
    source = replace_once(source, 'raise RuntimeError("U3 side approach outside calibrated face orientation")',
                          'raise RuntimeError(f"Upper {slot} side approach outside calibrated face orientation")')
    source = replace_once(source, '                    align_contact_plane(push_side, aligned, "upper_B_u3_side_pitch")\n'
                          '                    record("upper_B_u3_side_pitch",',
                          '''                    pitch_phase = f"upper_B_u{slot}_side_pitch"
                    align_contact_plane(push_side, aligned, pitch_phase)
                    record(pitch_phase,''')
    source = replace_once(source, '                    push_to_boundary(support_side, 1, 1, motion_prefix + "_rear_reseat", rear_target)',
                          '''                    rear_phase = "upper_B_final_rear_reseat" if final_u2 else motion_prefix + "_rear_reseat"
                    push_to_boundary(support_side, 1, 1, rear_phase, rear_target)''')
    source = replace_once(source, '                    prepare_rear_to_side_hold(preload)\n                    touch_face(push_side,',
                          '''                    if side_first:
                        hold_front(.55)
                    else:
                        prepare_rear_to_side_hold(preload)
                    touch_face(push_side,''')
    source = replace_once(source, '''                if slot == 4:
                    push_u4_with_front_seal()
                else:
                    push_to_boundary(push_side, 0, push_sign, "upper_B_side", side_target, follow=support_side)''',
                          '''                push_to_boundary(push_side, 0, push_sign, "upper_B_side", side_target, follow=support_side)''')
    source = replace_once(source, '''                if slot in (1, 2):
                    carry_across_support_seam()
                completed_transport = CHECKPOINT.get("postseam", {}).get("transport_index")''',
                          '''                if slot in (1, 2, 3, 4) and completed_transport not in (1, 2):
                    carry_across_support_seam()''')
    source = replace_once(source, 'else observe_placed_cubes(count=6))',
                          '''else observe_known_lower())''')
    source = replace_once(source, 'if slot == START_SLOT and RELEASE_CHECKPOINT is not None:',
                          'if START_FRONT == 6 and slot == START_SLOT and RELEASE_CHECKPOINT is not None:')
    source = replace_once(source, '            if START_FRONT == 6 and slot == START_SLOT and RELEASE_CHECKPOINT is not None:',
                          '''            seam_resume = (RELEASE_CHECKPOINT or {}).get("seam_entry_resume") if slot == START_SLOT else None
            if seam_resume:
                record("upper_seam_entry_checkpoint", "GROUNDED_POSE_RESTORED", slot=slot, source=seam_resume)
            elif START_FRONT == 6 and slot == START_SLOT and RELEASE_CHECKPOINT is not None:''')
    source = replace_once(source, '            if np.linalg.norm(observed - GUIDE["landing_m"]) > .008:',
                          '''            expected = RELEASE_CHECKPOINT["position_m"] if seam_resume else GUIDE["landing_m"]
            if np.linalg.norm(observed - expected) > (.004 if seam_resume else .008):''')
    source = replace_once(source, '            save_upper_checkpoint(slot, upper_pose_records({str(task_id): cube})[0])',
                          '''            landed_now = upper_pose_records({str(task_id): cube})[0]
            if seam_resume:
                landed_now["seam_entry_resume"] = seam_resume
            save_upper_checkpoint(slot, landed_now)''')
    source = replace_once(source, 'current_task = {"cube_id": 6 + START_SLOT, "marker_id": 28 + START_SLOT}',
                          '# Keep observing the current body until upper supply arrives.')
    source = replace_once(source, '            if slot > START_SLOT:', '            if slot > START_SLOT or START_FRONT < 6:')
    source = replace_once(source, '            slot3_guide = json.loads(guide_bytes)', """            slot3_guide = json.loads(guide_bytes)
            park = next(row for row in slot3_guide["unloaded_transits"] if row["phase"] == "return_home_after_withdrawal")
            if not np.isclose(park["q_rad"]["left"][1][0], -.077954311, atol=1e-6):
                raise ValueError("Upper slot 3 parking guide changed; recheck the occupied front row")
            park["q_rad"]["left"][1][0] -= .06
            report["upper_row"]["slot3_empty_park_left_j1_offset_rad"] = -.06""")
    source = replace_once(source, '                "next_slot": next_slot,',
                          '''                "scene": SCENE, "next_front": 6, "front_completed": report["front_row"]["completed"],
                "rear_initial": rear_initial, "source_run": str(output_directory),
                "next_slot": next_slot,''')
    source = replace_once(source, 'if all_verified and not fixture else',
                          'if all_verified and not fixture and len(report["front_row"]["completed"]) == 5 else')
    source = replace_once(source, '                if tick % vision_period == 0:',
                          '                renders_before = report["simulation_clock"]["render_count"]\n'
                          '                if tick % vision_period == 0:')
    source = replace_once(source, 'if not args.headless and tick % 4 == 0:',
                          'if not args.headless and tick % 4 == 0 and report["simulation_clock"]["render_count"] == renders_before:')
    source = replace_once(source, '            elif (phase, state) in (("A_lift_and_align",',
                          '            elif phase == "front_slot_complete" or ("front_slot" in current_task and phase == "A_lift_and_align" and state == "CLEAR_PHYSX_EXECUTION"):\n'
                          '                label = f"front_{current_task.get(\'front_slot\')}_{phase}"\n'
                          '            elif (phase, state) in (("A_lift_and_align",')
    source = replace_once(source, '                try:\n                    for index in range(count):',
                          '                from two_row_physx import frame_marker_corners\n'
                          '                corners = frame_marker_corners(vision_estimator, rgba)\n'
                          '                try:\n                    for index in range(count):')
    source = replace_once(source, '                        result = vision_estimator.estimate(rgba.astype(np.uint8, copy=False))',
                          '                        result = {"corners_px": corners.get(vision_estimator.marker_id)}')
    source = replace_once(source, '                        pose = vision_estimator.resting_pose(result, observation_plane)',
                          """                        pose = vision_estimator.resting_pose(result, observation_plane)
                        if pose is not None:
                            from front_yaw_check import marker_pose_on_floor
                            pose = marker_pose_on_floor(result["corners_px"], vision_estimator, observation_plane, pose)""")
    from two_row_step_timing import inject as time_control_steps
    from front_yaw_check import inject as confirm_front_yaw
    from front_b1_checkpoint import inject as add_front_b1_boundary
    from front_row_skills import inject_side_resume
    return inject_side_resume(add_front_b1_boundary(confirm_front_yaw(time_control_steps(source))))


def frame_marker_corners(estimator, rgba):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(rgba[..., :3].astype(np.uint8, copy=False), cv2.COLOR_RGB2GRAY)
    if estimator.detector is not None:
        corners, ids, _ = estimator.detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, estimator.dictionary, parameters=estimator.parameters)
    found = {}
    if ids is not None:
        for marker, points in zip(ids.ravel(), corners, strict=True):
            found.setdefault(int(marker), points.reshape(4, 2))
    return found


def start_state(checkpoint):
    front = checkpoint.get("next_front", 1)
    upper = checkpoint.get("next_slot", 1)
    if not 1 <= front <= 6 or not 1 <= upper <= 5 or (front < 6 and upper != 1):
        raise ValueError("Invalid row or slot in checkpoint")
    if checkpoint and checkpoint.get("scene") != SCENE:
        raise ValueError("Checkpoint belongs to a different layout")
    if checkpoint.get("front_b2_sync"):
        raise ValueError("This B2 checkpoint was saved against the rear row; use the A landing or staged B1 checkpoint")
    if checkpoint:
        saved = checkpoint["initial_boxes"]
    else:
        saved = json.loads((HERE / "validation" / "rear_row_seed.json").read_text())
    expected = {*REAR_IDS, *FRONT_IDS[:front-1], *range(7, 6 + upper)}
    if len(saved) != len(expected) or {box["cube_id"] for box in saved} != expected:
        raise ValueError("Initial bodies do not match the completed slots")
    if [r["cube_id"] for r in checkpoint.get("front_completed", [])] != list(FRONT_IDS[:front-1]):
        raise ValueError("Missing measured front placements")
    if [r["slot"] for r in checkpoint.get("completed_slots", [])] != list(range(1, upper)):
        raise ValueError("Missing measured upper placements")
    landed = checkpoint.get("landed") or {}
    seam = landed.get("seam_entry_resume")
    transports = {1: (1,), 2: (None, 1, 2), 3: (None, 1), 4: (1,)}
    if seam and (front != 6 or seam.get("slot") != upper or landed.get("cube_id") != 6 + upper
                 or seam.get("completed_transport") not in transports.get(upper, ())):
        raise ValueError("Invalid upper seam checkpoint")
    return front, upper, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--close-on-finish", action="store_true")
    parser.add_argument("--stop-after-front", type=int, choices=range(1, 6))
    args = parser.parse_args()
    checkpoint = {}
    if args.resume:
        data = args.resume.read_bytes()
        checkpoint = json.loads(data)
        checkpoint["input_provenance"] = {"path": str(args.resume.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
    front, upper, saved = start_state(checkpoint)
    if args.stop_after_front is not None and args.stop_after_front < front:
        parser.error("The requested stopping slot is already complete in the checkpoint")
    first_id = FRONT_IDS[front-1] if front < 6 else 6 + upper
    output = args.output.resolve()
    snapshot = output / "runner_snapshot.py"
    source = build()
    runner = compile(source, str(snapshot), "exec")
    if args.check_only:
        print(f"Compiled. Measured starting boxes {len(saved)}, next supply Cube_{first_id}")
        return
    with open('/tmp/fr3-upper-one-kit.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "Another FR3 Isaac Sim run owns the single-instance lock.\n")
        output.mkdir(parents=True, exist_ok=False)
        snapshot.write_text(source)
        sys.path.insert(0, str(REPO))
        sys.argv = [str(REPO / "fr3_ab_visual_runner.py"), "--five-cube-row", "--output-dir", str(output),
                    "--collision-workers", "3"]
        if args.close_on_finish:
            sys.argv.append("--close-on-finish")
        exec(runner, {"__file__": sys.argv[0], "__name__": "__main__", "SCENE": SCENE, "LOWER_IDS": LOWER_IDS,
                      "FRONT_IDS": FRONT_IDS, "START_FRONT": front, "START_SLOT": upper, "FIRST_ID": first_id,
                      "STOP_AFTER_FRONT": args.stop_after_front,
                      "INITIAL_BOXES": saved, "RUNTIME_GUIDES": row_guide_paths(), "CHECKPOINT": checkpoint,
                      "RELEASE_CHECKPOINT": checkpoint.get("front_landed") or checkpoint.get("landed"),
                      "SUPPORT_TOP_BEVEL_M": None, "UPPER_REST_OFFSET_M": None, "CUBE_CONTACT_OFFSET_M": None})


if __name__ == "__main__":
    main()
