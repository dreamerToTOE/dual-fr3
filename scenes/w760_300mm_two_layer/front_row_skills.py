"""Floor-level B1/B2 for the five boxes in front of the measured rear row."""
import numpy as np
from textwrap import indent

from continuous_side_b import patch_b
from continuous_side_row import REPO, patch_a
from source_hooks import replace_once


FRONT_IDS = (12, 15, 13, 14, 6)
OUTER_REAR_GAP_M = .025


def front_geometry(slot, observations, size, floor_z, wall_goal_x):
    if not 1 <= slot <= 5:
        raise ValueError(f"Unknown front slot {slot}")
    tracked = {name: (np.asarray(p), np.asarray(r)) for name, p, r in observations}
    half = size / 2

    def edge(name, axis, sign):
        p, r = tracked[name]
        if r.shape != (3, 3) or p.shape != (3,) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(p)):
            raise ValueError(f"Invalid RGB pose for {name}")
        return float(p[axis] + sign * half * np.abs(r[axis]).sum())

    y = edge(f"Cube_{slot}", 1, -1) - half
    if slot in (1, 2):
        x = wall_goal_x
    elif slot in (3, 4):
        sign = 1 if slot == 3 else -1
        x = edge(f"Cube_{FRONT_IDS[slot - 3]}", 0, sign) + sign * half
    else:
        left = edge(f"Cube_{FRONT_IDS[2]}", 0, 1)
        right = edge(f"Cube_{FRONT_IDS[3]}", 0, -1)
        if right - left < size + .002 - 1e-12:
            raise RuntimeError(f"Front center insertion slot is only {right-left:.5f} m wide")
        x = (right + left) / 2
    return np.array([x, y, floor_z + half])


def seam_blends(start_x, distance, rear, half, sign, nominal_times):
    """Slow the approach to each measured incoming rear edge while keeping a continuous lateral path."""
    u = nominal_times / nominal_times[-1]
    nominal = u**3 * (10 + u * (-15 + 6*u))
    centers = start_x + distance * nominal
    edges = [p[0] - sign * half * (abs(r[0]).sum() + 1) for _, p, r in rear]
    nearest = np.min(np.abs(centers[:, None] - edges), axis=1)
    bump = .5 * (1 + np.cos(np.pi * np.minimum(nearest / .012, 1.)))
    factor = 1. + 11. * bump**2
    stretched = np.r_[0., np.cumsum(np.diff(nominal_times) * (factor[:-1] + factor[1:]) / 2)]
    times = np.arange(int(np.ceil(stretched[-1] * 120)) + 1) / 120
    blends = np.interp(times, stretched, nominal)
    return times, blends


def inject_side_resume(source):
    start = source.index('            allowed_contacts.discard("right")', source.index('        def run_front_b('))
    end = source.index('            if staged:\n                if slot in (1, 2, 3):\n                    unload_normal', start)
    nearby = '''            side_resume = CHECKPOINT.get("front_side") if slot == START_FRONT else None
            if side_resume:
                if slot not in (1, 2, 3) or side_resume["cube_id"] != current_task["cube_id"]:
                    raise RuntimeError("Mismatched side-seat checkpoint")
                staged = True
                b2_target = placement_target.copy()
                front_rotation = np.asarray(FRONT_ROTATION) if support_side == "left" else np.diag([-1., 1., -1.])
                front_height = .004
                cube_start = np.asarray(side_resume["cube_start_rgb"])
                progress = float(side_resume["lateral_progress_m"])
                support_motion = support_x_motion = 0.
                hold_errors, push_errors = [], []
                evidence["side_seat"] = side_resume["side_seat"]
                record("front_side_resume", "RESTORED_CONFIRMED_SIDE_SEAT", source_step=side_resume["source_step"])
            else:
'''
    source = source[:start] + nearby + indent(source[start:end], '    ') + source[end:]
    source = replace_once(source, '            if staged:\n                if slot in (1, 2, 3):\n                    unload_normal',
                          '            if staged:\n                if slot in (1, 2, 3) and not side_resume:\n                    unload_normal')
    source = replace_once(source, '''                observed, rotation = observe_resting_cube("B2_staged_rear_pose", vision_camera,
                                                           vision_estimator, CONFIG["floor_top_m"])''', '''                if slot in (1, 2, 3):
                    # Move the full arm out of the view; a straight wrist retreat can leave the marker behind its elbow.
                    reapproach_q = applied_targets[support_side].copy()
                    record("B2_stage_camera_clear", "RUNNING", arm=support_side)
                    move({support_side: np.asarray(CONFIG["fr3"]["official_start_q_rad"])})
                observed, rotation = observe_resting_cube("B2_staged_rear_pose", vision_camera,
                                                           vision_estimator, CONFIG["floor_top_m"])
                if slot in (1, 2, 3):
                    move({support_side: reapproach_q})''')
    start = source.index('                control_center, control_rotation = observe_resting_cube(\n                    "front_floor_landing_resume"')
    end = source.index('            else:\n                record("front_supply"', start)
    restore = '''                receipt = CHECKPOINT.get("front_side") or CHECKPOINT.get("front_direct_rear")
                if receipt:
                    control_center = np.asarray(receipt["estimated_center_m"], dtype=float)
                    control_rotation = np.asarray(receipt["estimated_rotation_world"], dtype=float)
                    if control_center.shape != (3,) or control_rotation.shape != (3, 3) or not (
                            np.all(np.isfinite(control_center)) and np.all(np.isfinite(control_rotation))):
                        raise RuntimeError("Invalid sensed pose in side-seat checkpoint")
                else:
'''
    source = source[:start] + restore + indent(source[start:end], '    ') + source[end:]
    source = replace_once(source, '            if not (front_slot == START_FRONT and CHECKPOINT.get("front_b1")):',
                          '            if not (front_slot == START_FRONT and (CHECKPOINT.get("front_b1") or CHECKPOINT.get("front_side") or CHECKPOINT.get("front_direct_rear"))):')
    return replace_once(source, '                    if CHECKPOINT.get("front_b1") and side == CHECKPOINT["front_b1"]["support_arm"]:',
                        '''                    if CHECKPOINT.get("front_direct_rear") and side == ("left" if front_slot == 3 else "right"):
                        tip, attitude = tcp_fk(side, q)
                        q = cartesian_joint_target(side, q, tip - [0., .0015, 0.], attitude)
                    if CHECKPOINT.get("front_b1") and side == CHECKPOINT["front_b1"]["support_arm"]:''')


def inject(source):
    source = replace_once(source, '        def contact_path(side, start, end, phase):',
                          '        def contact_path(side, start, end, phase, *, speed=.08, force_limit=None):')
    source = replace_once(source, '1.875 * float(np.linalg.norm(end - start)) / .08)',
                          '1.875 * float(np.linalg.norm(end - start)) / speed)')
    source = replace_once(source, '                if np.linalg.norm(actual - centers[index]) > .035:',
                          '''                if force_limit is not None:
                    wrench = np.asarray(last_force_sample[side]["tcp_gravity_corrected_world"])
                    if (not np.all(np.isfinite(wrench)) or np.linalg.norm(wrench[:3]) > force_limit
                            or np.linalg.norm(wrench[3:]) > 1.):
                        raise RuntimeError(f"{phase}: {side} exceeded its wrist wrench limit: {wrench}")
                if np.linalg.norm(actual - centers[index]) > .035:''')
    floor = patch_b(patch_a((REPO / "fr3_ab_visual_runner.py").read_text()))
    start = floor.index('            set_contact_drive_mode(True)\n            allowed_contacts.discard("right")')
    end = floor.index("        placed_bodies = [cube]", start)
    body = floor[start:end]
    park = floor[floor.index("        def park_for_final_rgb("):floor.index("        def settle_cube(")]
    # The upper runner has the rear D455, without the original row's extra left bay camera.
    for first, last in [('            if slot == 1:\n                # Use the centered', '            allowed_contacts.add(support_side)'),
                        ('            if slot == 1:\n                vision_camera.pause()', '            max_speed = settle_cube("B2")')]:
        start = body.index(first)
        body = body[:start] + body[body.index(last, start):]
    body = body.replace('            slot = current_task["cube_id"]\n', '')
    body = replace_once(body, 'row["cube_id"] == current_task["cube_id"]', 'row["cube_id"] == slot')
    body = replace_once(body, 'current_task["cube_id"] == 4', 'slot == 4')
    body = replace_once(body, 'np.asarray(CONFIG["task"]["phase_B1"]["target_center_m"], dtype=float)', 'placement_target.copy()')
    body = replace_once(body, 'np.asarray(CONFIG["task"]["phase_B2"]["target_center_m"], dtype=float)', 'placement_target.copy()')
    body = replace_once(body, '            stations = {support_side: support_y}',
                        '''            outer_lane_shift = OUTER_REAR_GAP_M if slot == 1 else 0.
            stations = {support_side: support_y + rail_shift - outer_lane_shift}''')
    body = replace_once(body, 'stations[push_side] = outside_plan["push_y"]',
                        'stations[push_side] = outside_plan["push_y"] + rail_shift - outer_lane_shift + (.075 if slot == 1 else 0.)')
    body = replace_once(body, 'push_base[1] = CONFIG["task"]["phase_B2"]["rail_assist"]["base_y_m"]', 'push_base[1] = CONFIG["task"]["phase_B2"]["rail_assist"]["base_y_m"] + rail_shift')
    body = replace_once(body, 'np.isclose(case["rail_y_m"], rail_states[push_side][1])', 'np.isclose(case["rail_y_m"] + rail_shift - outer_lane_shift + (.075 if slot == 1 else 0.), rail_states[push_side][1])')
    body = replace_once(body, 'if b1_progress < 0.20:', 'if b1_progress < max(.008, b1_end[1] - cube_start[1] - .010):')
    body = replace_once(body, '            if push_side is not None:\n                b1_target[0] = b1_push_start[0]',
                        '            b1_target[0] = b1_push_start[0]')
    anchor = '            b1_end = b1_target.copy()'
    body = replace_once(body, anchor, '''            if push_side is None:
                gaps = center_gaps(b1_push_start, b1_rotation)
                if min(gaps) < .001:
                    raise RuntimeError(f"Front center straight path has insufficient measured clearance: {gaps}")
                record("front_center_entry", "STRAIGHT_PATH_CLEAR", per_side_gap_m=gaps)
            initial_rear = overlapping_rear(b1_push_start, b1_rotation)
            if not initial_rear:
                raise RuntimeError("No rear box overlaps the measured front B1 lane")
            b1_target[1] = min(front_planes[name] for name in initial_rear) - half
''' + anchor)
    body = body.replace('contact_target="wall_back"', 'contact_target="rear_row"')
    body = replace_once(body, '                position, left_hold_rotation = tcp_fk(support_side, applied_targets[support_side])\n                left_hold_z = float(position[2])\n                left_hold = NormalForceController(\n', '''                probe_rear_face(support_side)
                verify_contact("first_rear", ["/World/" + name for name in initial_rear], support_side, 1, 1)
                position, left_hold_rotation = tcp_fk(support_side, applied_targets[support_side])
                left_hold_z = float(position[2])
                left_hold = NormalForceController(
''')
    body = replace_once(body, '            unload_normal(push_side, 0, push_sign, "B2_push_unload")', '''            verify_contact("side_seat", [side_path], push_side, 0, push_sign)
            unload_normal(push_side, 0, push_sign, "B2_push_unload")''')
    needle = 'unload_normal(support_side, 1, 1, "B2_support_unload")'
    assert body.count(needle) == 2
    body = '\n'.join((line[:len(line) - len(line.lstrip())] +
                        'verify_contact("rear_seat", ["/World/Cube_" + str(slot)], support_side, 1, 1)\n' + line)
                       if needle in line else line for line in body.split('\n'))
    body = body.replace('                return\n', '                return finish()\n')
    body = body.replace('current_task.get("contact_target") in ("Cube_1", "Cube_2")', 'slot in (3, 4)')
    body = replace_once(body, '            if args.five_cube_row and slot in (3, 4):\n                record("B2_post_unload_observation"',
                        '            if args.five_cube_row and slot == 4:\n                record("B2_post_unload_observation"')
    body = body.replace('observe_placed_cubes(required_names=[current_task["contact_target"]])',
                        '[row for row in observe_placed_cubes(count=15, required_names=[current_task["contact_target"]]) if row[0] != active_cube_path.rsplit("/", 1)[-1]]')
    body = body.replace('withdrawing_corner_tools = True', 'current_task["front_withdrawal"] = True')
    body = body.replace('withdrawing_corner_tools = False', 'current_task.pop("front_withdrawal", None)')
    body = replace_once(body, '            if staged:\n                left_hold = None\n                observed, rotation = observe_resting_cube(',
                        '''            if staged:
                if slot in (1, 2, 3):
                    unload_normal(support_side, 1, 1, "B2_stage_support_unload")
                    left_hold = None
                    push_faces.pop(support_side, None)
                    position, _ = tcp_fk(support_side, applied_targets[support_side])
                    cartesian_move(support_side, position - [0., .065, 0.], 4., "B2_stage_rgb_clearance")
                left_hold = None
                observed, rotation = observe_resting_cube(''')
    body = replace_once(body, '                rear = observed.copy()\n',
                        '''                if slot in (1, 2, 3):
                    if abs(math.atan2(rotation[1, 0], rotation[0, 0])) > math.radians(1.5):
                        raise RuntimeError("Staged rear insertion: Cube yaw exceeds 1.5 degrees")
                    cartesian_move(support_side, observed + [0., -half - .007, front_height],
                                   3., "B2_stage_rear_pretouch")
                    allowed_contacts.add(support_side)
                    touch_face(support_side, np.array([0., 1., 0.]), .016, "B2_stage_rear_touch")
                    observed, rotation = sensed_cube()
                rear = observed.copy()
''')
    body = replace_once(body, '                                                  "target_uncertainty_m": .003, "samples": 30})',
                        '''                                                  "target_uncertainty_m": .003, "samples": 30})
                probe_rear_face(support_side)''')
    source = replace_once(source, 'if push_faces and receive_time - capture_time <= .20:',
                          'if push_faces and not current_task.get("front_withdrawal") and 0 <= receive_time - capture_time <= .20:')
    source = source.replace('current_task.get("contact_target") in ("Cube_1", "Cube_2")',
                            'current_task.get("contact_target") in ("Cube_1", "Cube_2", "Cube_12", "Cube_15")')
    source = replace_once(source, 'neighbor = contact_target in ("Cube_1", "Cube_2")',
                          'neighbor = contact_target in ("Cube_1", "Cube_2", "Cube_12", "Cube_15")')
    source = replace_once(source, '            if issues:\n                raise RuntimeError(f"Measured collision in {phase}: {issues[0]}")', """            if issues and "front_slot" in current_task and push_faces and issues[0]["kind"] == "cube_obstacle":
                # PhysX reports the signed surface distance even for speculative, positive-gap neighbors.
                # Check every pair so a tolerated optical error cannot hide a link collision.
                physical = contact_monitor.cube_contact_snapshot()
                expected = current_task.get("rear_contact_names", ())
                if physical["valid"] and issues[0]["b"] in expected:
                    unverified = []
                    for issue in checker.check(joints_now, **{**options, "first_only": False}):
                        separation = next((p["separation_m"] for p in physical["pairs"]
                            if p["current_step_observed"] and p["last_separation_physics_step"] == physical["physics_step_index"]
                            and {p["collider0"], p["collider1"]} == {active_cube_path, "/World/" + issue["b"]}), None)
                        if (issue["kind"] != "cube_obstacle" or issue["b"] not in expected
                                or issue["clearance_m"] < -.005 or separation is None or separation < -.00035):
                            unverified.append(issue)
                        else:
                            key = str(current_task["cube_id"]) + ":" + issue["b"]
                            audit = report.setdefault("front_contact_geometry", {}).setdefault(key,
                                {"verified_ticks": 0, "min_native_m": float(separation), "min_estimated_m": issue["clearance_m"]})
                            audit["verified_ticks"] += 1
                            audit["min_native_m"] = min(audit["min_native_m"], float(separation))
                            audit["min_estimated_m"] = min(audit["min_estimated_m"], issue["clearance_m"])
                    issues = unverified
            if issues:
                raise RuntimeError(f"Measured collision in {phase}: {issues[0]}")""")
    body = replace_once(body, '            for time_s in push_times:\n                planning_update("B2: smooth Cartesian force-controlled path")\n                t = time_s / push_times[-1]\n                blend = 10 * t**3 - 15 * t**4 + 6 * t**5',
                        """            from front_row_skills import seam_blends
            if staged:
                t = push_times / push_times[-1]
                blends = t**3 * (10 - 15*t + 6*t*t)
            else:
                push_times, blends = seam_blends(b2_start[0], distance, rear, half, push_sign, push_times)
            for blend in blends:
                planning_update("B2: planning lateral travel")""")
    body = replace_once(body, '            execute_joint_path(push_samples, "B2_push", check_step=check_b2_hold,',
                        """            vertical_offset = axial_offset = axial_average = tangential_offset = 0.
            rear_gap_ticks = 0
            compliant_rear = slot in (1, 2) and not staged
            if compliant_rear:
                support_settings = (left_hold.target_force_N, left_hold.gain_m_per_Ns, left_hold.max_speed_m_s)
                left_hold.gain_m_per_Ns = .00016
                left_hold.max_speed_m_s = .001
            load_control = report.setdefault("front_push_load_control", {}).setdefault(str(slot),
                {"max_raise_m": 0., "max_axial_yield_m": 0., "peak_axial_N": 0., "peak_downward_N": 0.,
                 "held_steps": 0, "contact_ticks": 0, "rear_loaded_ticks": 0, "max_rear_gap_ticks": 0,
                 "floor_loaded_ticks": 0, "max_tangential_yield_m": 0.,
                 "min_rear_force_target_N": left_hold.target_force_N if left_hold is not None else None})

            def level_push(index, commands):
                nonlocal vertical_offset, axial_offset, axial_average, tangential_offset
                wrench = np.asarray(last_force_sample[push_side]["tcp_gravity_corrected_world"])
                if not np.all(np.isfinite(wrench)) or np.linalg.norm(wrench[3:]) > 1.:
                    raise RuntimeError(f"Front B2 wrist wrench invalid: {wrench}")
                axial = float(-push_sign * wrench[0])
                axial_average += .25 * (axial - axial_average)
                axial_speed = float(np.clip(.001 * (6.5 - axial_average), -.003, .001))
                axial_offset = float(np.clip(axial_offset + axial_speed / 120, -.003, 0.))
                vertical = float(wrench[2])
                load_control["peak_downward_N"] = max(load_control["peak_downward_N"], vertical)
                load_control["max_axial_yield_m"] = max(load_control["max_axial_yield_m"], -axial_offset)
                vertical_error = vertical + (1.8 if slot in (1, 2) else 0.)
                deadband = math.copysign(max(abs(vertical_error) - .25, 0.), vertical_error)
                vertical_offset = float(np.clip(vertical_offset + np.clip(.0010 * deadband, -.003, .003) / 120,
                                                -.002, .006))
                load_control["max_raise_m"] = max(load_control["max_raise_m"], vertical_offset)
                if compliant_rear:
                    left_hold.target_force_N = float(np.clip(2. - .15 * max(0., axial_average - 3.), 1.45, 2.))
                    load_control["min_rear_force_target_N"] = min(load_control["min_rear_force_target_N"],
                                                                  left_hold.target_force_N)
                    tangent = math.copysign(max(abs(wrench[1]) - .25, 0.), wrench[1])
                    tangential_offset = float(np.clip(tangential_offset + np.clip(.0007 * tangent, -.001, .001) / 120,
                                                      -.002, .002))
                    load_control["max_tangential_yield_m"] = max(load_control["max_tangential_yield_m"], abs(tangential_offset))
                tip, attitude = tcp_fk(push_side, commands[push_side])
                correction = [push_sign * axial_offset, tangential_offset, vertical_offset]
                adjusted = cartesian_joint_target(push_side, commands[push_side], tip + correction,
                                                  attitude, retarget_guide=True)
                if np.max(np.abs(adjusted - applied_targets[push_side])) * 120 > .4:
                    raise RuntimeError("Front B2 load correction exceeded joint speed")
                follower_tip, follower_rotation = tcp_fk(support_side, commands[support_side])
                following = cartesian_joint_target(support_side, commands[support_side],
                    follower_tip + [push_sign * axial_offset, 0., 0.], follower_rotation)
                return {push_side: adjusted, support_side: following}

            def advance_under_load(count):
                for index in range(count):
                    yield index
                    for _ in range(240):
                        if axial_average < 7.2:
                            break
                        load_control["held_steps"] += 1
                        yield index
                    else:
                        raise RuntimeError(f"Front B2 cannot advance under sustained {axial_average:.2f} N load")

            rear_paths = {"/World/" + name for name in front_planes}

            def check_pressing_push(index, count):
                nonlocal rear_gap_ticks
                axial = abs(float(last_force_sample[push_side]["tcp_gravity_corrected_world"][0]))
                load_control["peak_axial_N"] = max(load_control["peak_axial_N"], axial)
                snap = contact_monitor.cube_contact_snapshot()
                pairs = [row for row in snap["pairs"] if row["current_step_observed"]
                         and active_cube_path in (row["collider0"], row["collider1"])]
                loaded = snap["valid"] and sum(120 * (row["normal_impulse_Ns"] or 0.)
                    for row in pairs if row["collider0"] in rear_paths or row["collider1"] in rear_paths) >= .10
                on_floor = snap["valid"] and any(row["label"] == "floor" and (row["normal_impulse_Ns"] or 0.) > 1e-5
                                                   for row in pairs)
                load_control["contact_ticks"] += 1
                load_control["rear_loaded_ticks"] += int(loaded)
                load_control["floor_loaded_ticks"] += int(on_floor)
                if staged and loaded:
                    raise RuntimeError("Staged lateral push made early contact with the rear row")
                if not staged:
                    rear_gap_ticks = 0 if loaded else rear_gap_ticks + 1
                    load_control["max_rear_gap_ticks"] = max(load_control["max_rear_gap_ticks"], rear_gap_ticks)
                    if rear_gap_ticks > 12:
                        raise RuntimeError("Front B2 lost rear bearing for more than 100 ms")
                check_b2_hold(index, count)

            execute_joint_path(push_samples, "B2_push", check_step=check_pressing_push,
                               adjust_targets=level_push, sample_order=advance_under_load,""")
    body = replace_once(body, 'cube_center_path=b2_cube_center_path, sampled_trajectory=(push_times, push_samples))\n            step(60)',
                          '''cube_center_path=b2_cube_center_path, sampled_trajectory=(push_times, push_samples))
            if compliant_rear:
                left_hold.target_force_N, left_hold.gain_m_per_Ns, left_hold.max_speed_m_s = support_settings
            step(60)''')
    source = replace_once(source, 'if side in velocities\n                                and 0 < index < len(times) - 1',
                          'if side in velocities\n                                and not (phase == "B2_push" and "front_slot" in current_task and sample_order is not None)\n                                and 0 < index < len(times) - 1')
    first = body.index('                outside_q = cartesian_joint_target(')
    end = body.index('                descent_times = ', first)
    body = body[:first] + """                candidates = sorted((np.asarray(q) for q in outside_plan["b2_entry_q"]),
                                    key=lambda q: np.linalg.norm(tcp_fk(push_side, q)[0] - outside)) if outside_plan else []
                if slot == 1:
                    # Interior of the feasible elbow branch at the front-row rail station.
                    candidates.insert(0, [.3599203, .4000311, -.6990516, -2.6418535, .9486691, 2.8280980, -2.7945712])
                candidates.extend([current_task.get("staged_precontact_q", case["precontact_q"][push_side]),
                                   role_seeds["descent"][-1], joint_state()[push_side]])
                for candidate in candidates:
                    try:
                        outside_q = cartesian_joint_target(push_side, candidate, outside, b2_rotation, retarget_guide=True)
                        break
                    except RuntimeError as error:
                        if "bounded Cartesian correction failed" not in str(error):
                            raise
                else:
                    seed = solve(push_side, outside, b2_rotation, seed_override=candidates[0])
                    outside_q = cartesian_joint_target(push_side, seed, outside, b2_rotation, retarget_guide=True)
""" + body[end:]
    header = '''        def run_front_b(slot, observed_neighbors):
            from front_row_skills import front_geometry, FRONT_IDS, OUTER_REAR_GAP_M
            nonlocal support_side, push_side, push_sign, left_hold, left_hold_z, left_hold_rotation
            nonlocal contact_plan_rotation
            assert current_task["cube_id"] == FRONT_IDS[slot-1]
            role = dict(CONFIG["cube_sequence"][slot - 1])
            for key in ("cube_id", "marker_id"):
                role.pop(key)
            current_task.update(role)
            if slot in (1, 2):
                current_task["staging_offset_from_rear_m"] = OUTER_REAR_GAP_M
            placement_target = front_geometry(slot, observed_neighbors, CONFIG["cube_size_m"], CONFIG["floor_top_m"],
                                    CONFIG["goal_cube_centers_m"][slot-1][0])
            current_task["goal_center_m"] = placement_target.tolist()
            rail_shift = placement_target[1] - CONFIG["goal_cube_centers_m"][slot-1][1]
            if slot in (3, 4):
                current_task["contact_target"] = f"Cube_{FRONT_IDS[slot-3]}"
            name = current_task["contact_target"]
            side_path = ("/World/" if slot in (3, 4) else "/World/Environment/") + name if name else None
            evidence = {}
            rear = [(name, np.asarray(p), np.asarray(r)) for name, p, r in observed_neighbors
                    if name in {f"Cube_{i}" for i in range(1, 6)}]
            if len(rear) != 5:
                raise RuntimeError("Front B needs all five RGB observations of the rear row")
            front_planes = {name: p[1] - half * abs(r[1]).sum() for name, p, r in rear}
            current_task["rear_contact_names"] = list(front_planes)

            def overlapping_rear(center, rotation):
                return [name for name, p, r in rear if abs(center[0] - p[0]) <
                        half * (abs(rotation[0]).sum() + abs(r[0]).sum()) - .002]

            def observe_b_cube(phase):
                try:
                    return observe_resting_cube(phase, vision_camera, vision_estimator, CONFIG["floor_top_m"])
                except RuntimeError as exc:
                    if (phase != "B2_approach_pose" or str(exc) != phase + ": no stable fresh ArUco observation"
                            or (not evidence.get("first_rear") and not staged) or left_hold is None):
                        raise
                # While the marker is covered, the bearing rear face fixes Y and the held front face fixes yaw.
                # The last visible X is valid only while the support TCP remains on the same front patch.
                center, _ = sensed_cube()
                tcp, wrist_rotation = tcp_fk(support_side, joint_state()[support_side])
                nominal = np.asarray(FRONT_ROTATION) if support_side == "left" else np.diag([-1., 1., -1.])
                frame = wrist_rotation @ nominal.T
                yaw = math.atan2(frame[1, 0], frame[0, 0])
                ref_yaw = math.atan2(rear_rotation[1, 0], rear_rotation[0, 0])
                force = -last_force_sample[support_side]["tcp_gravity_corrected_world"][1]
                if (abs(math.atan2(math.sin(yaw-ref_yaw), math.cos(yaw-ref_yaw))) > math.radians(3)
                        or abs(tcp[0] - b1_actual[0]) > .006
                        or abs(center[1] - (b1_actual[1] if staged else b1_target[1])) > .005
                        or not .4 <= force <= 9.):
                    raise RuntimeError("Front box occluded with insufficient RGB, wrist and support-face agreement")
                c, s = math.cos(yaw), math.sin(yaw)
                upright = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                center[0] = b1_actual[0]
                report["target_observations"].append({"phase": phase, "source": "last_visible_x_and_loaded_face_FK",
                    "position_m": center.tolist(), "yaw_deg": math.degrees(yaw), "support_force_N": float(force)})
                return center, upright

            def probe_rear_face(arm, verify_path=None):
                # Floor sliding friction can exceed the old 0.5 N wall-seating criterion by several newtons.
                baseline = []
                for _ in range(18):
                    step()
                    baseline.append(-last_force_sample[arm]["tcp_gravity_corrected_world"][1])
                # A confirmed loaded target pair separates rear bearing from sliding friction.
                threshold = .5 if verify_path is not None else min(7., max(5.5, float(np.median(baseline)) + 1.2))
                tip, orientation = tcp_fk(arm, applied_targets[arm])
                travel = 0.
                stable = 0
                for _ in range(2400):
                    wrench = np.asarray(last_force_sample[arm]["tcp_gravity_corrected_world"])
                    if not np.all(np.isfinite(wrench)) or np.linalg.norm(wrench[:3]) > 9. or np.linalg.norm(wrench[3:]) > 1.:
                        raise RuntimeError(f"Front rear seating exceeded wrist limit at {travel*1000:.2f} mm: {wrench}")
                    loaded = True
                    if verify_path is not None:
                        snapshot = contact_monitor.cube_contact_snapshot()
                        loaded = snapshot["valid"] and sum(
                            120 * (pair["normal_impulse_Ns"] or 0.) for pair in snapshot["pairs"]
                            if pair["current_step_observed"]
                            and {pair["collider0"], pair["collider1"]} == {active_cube_path, verify_path}) >= .10
                    if -wrench[1] >= threshold and loaded:
                        stable += 1
                        if stable >= 18:
                            record("front_rear_probe", "WRIST_LOAD_RISE", baseline_N=float(np.median(baseline)),
                                   threshold_N=threshold, measured_N=float(-wrench[1]), travel_m=travel)
                            return
                    else:
                        stable = 0
                        travel += .00045 / 120
                        if travel > .005:
                            break
                        q = cartesian_joint_target(arm, applied_targets[arm], tip + [0., travel, 0.], orientation)
                        command_positions(arm, q)
                    step()
                    check_current_collision("front_rear_probe")
                raise RuntimeError(f"Front rear contact unconfirmed after {travel*1000:.2f} mm tactile search")

            def verify_contact(label, paths, arm, axis, direction):
                positive = []
                floor_loads = []
                observed_paths = set()
                normals = []
                for _ in range(24):
                    step()
                    check_current_collision("front_B_" + label)
                    rows = contact_monitor.cube_contact_snapshot()["pairs"]
                    impulses = {}
                    for row in rows:
                        pair = {row["collider0"], row["collider1"]}
                        if active_cube_path not in pair or not row["current_step_observed"]:
                            continue
                        other = (pair - {active_cube_path}).pop()
                        impulses[other] = impulses.get(other, 0.) + 120 * (row["normal_impulse_Ns"] or 0.)
                    loaded = [path for path in paths if impulses.get(path, 0.) >= .10]
                    positive.append(bool(loaded))
                    observed_paths.update(loaded)
                    floor_loads.append(impulses.get("/World/Environment/floor", 0.))
                    wrench = np.asarray(last_force_sample[arm]["tcp_gravity_corrected_world"])
                    if not np.all(np.isfinite(wrench)) or np.linalg.norm(wrench[:3]) > 10. or np.linalg.norm(wrench[3:]) > 1.:
                        raise RuntimeError(f"Front {slot} {label} wrench outside limits: {wrench}")
                    normals.append(-direction * wrench[axis])
                evidence[label] = {"loaded_ticks": sum(positive), "ticks": 24, "contact_paths": sorted(observed_paths),
                                   "median_floor_N": float(np.median(floor_loads)),
                                   "median_wrist_normal_N": float(np.median(normals))}
                if sum(positive) < 18 or np.median(normals) < .25 or np.median(floor_loads) < .5:
                    raise RuntimeError(f"Front {slot} {label} lacked sustained floor/target contact: {evidence[label]}, expected {paths}")
                record("front_B_" + label, "NATIVE_CONTACT_CONFIRMED", **evidence[label])

            def center_gaps(center, rotation):
                tracked = {name: (p, r) for name, p, r in observed_neighbors}
                left, left_rotation = tracked[f"Cube_{FRONT_IDS[2]}"]
                right, right_rotation = tracked[f"Cube_{FRONT_IDS[3]}"]
                half_x = half * abs(rotation[0]).sum()
                return [float(center[0] - half_x - left[0] - half * abs(left_rotation[0]).sum()),
                        float(right[0] - half * abs(right_rotation[0]).sum() - center[0] - half_x)]

            def finish():
                position, rot = observe_b_cube("front_B_final")
                err = np.abs(position - placement_target)
                yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
                if max(err[:2]) > .005 or err[2] > .003 or abs(yaw) > 3.:
                    raise RuntimeError(f"Front slot {slot} final RGB pose out of tolerance: {err}, yaw={yaw}")
                if slot in (1, 2):
                    gap = oriented_wall_gap(position[0], rot[0], half, placement_target[0], push_sign)
                    evidence["final_side_geometry"] = {"rgb_signed_gap_m": float(gap), "limit_m": .004}
                    if abs(gap) > .004:
                        raise RuntimeError(f"Front slot {slot} side-wall placement changed after rear push: {gap*1000:.2f} mm")
                if slot in (3, 4):
                    gap = push_sign * (placement_target[0] - position[0]) + half * (1. - abs(rot[0]).sum())
                    evidence["final_side_geometry"] = {"rgb_signed_gap_m": float(gap), "limit_m": .004}
                    if abs(gap) > .004:
                        raise RuntimeError(f"Front slot {slot} neighbor gap after straight insertion is {gap*1000:.2f} mm")
                if slot == 5 and abs(position[0] - cube_start[0]) > .003:
                    raise RuntimeError("Front center box drifted over 3 mm during straight insertion")
                set_contact_drive_mode(False)
                return {"target_m": placement_target.tolist(), "observed_position_m": position.tolist(),
                        "observed_yaw_deg": yaw, "start_position_m": cube_start.tolist(), "contacts": evidence}

            if slot in (3, 4):
                from fr3_bilateral_b2 import b2_role_seeds

                support_side, push_side, push_sign = (current_task[key] for key in ("support_side", "push_side", "push_direction"))
                prefix = f"front_{slot}"
                set_contact_drive_mode(True)
                cube_start, attitude = observe_b_cube(f"{prefix}_landing_for_lateral")
                side_guide = b2_role_seeds(json.loads((ROOT / "validation" / "fr3_b2_collision_plan.json").read_text()),
                                            push_side)
                outward = -push_sign * attitude[:, 0]
                side_offset = np.array([-push_sign * half, 0., current_task["active_contact_height_m"]])
                side_tool = attitude @ side_guide["push_rotation_world"]
                base = rail_states[push_side].copy()
                base[1] = .80 + rail_shift
                record(f"{prefix}_{push_side}_rail", "RUNNING", arm=push_side, y_m=float(base[1]))
                move_robot_base(push_side, base)
                contact_plan_rotation = attitude.copy()
                target = cube_start + attitude @ side_offset + .007 * outward
                if slot == 4:
                    high = np.r_[target[:2], 1.05]
                    high_seed = np.array([-.974636, -.355642, 1.110620, -2.359528, .384786, 2.160466, 1.459065])
                    move({push_side: cartesian_joint_target(push_side, high_seed, high, side_tool, retarget_guide=True)})
                    cartesian_move(push_side, target, 5., f"{prefix}_lateral_descent")
                else:
                    seed = np.array([-.07088584, .27808819, -.10611819, -2.70179650, .17716509, 2.97585109, -1.91853907])
                    move({push_side: cartesian_joint_target(push_side, seed, target, side_tool, retarget_guide=True)})
                allowed_contacts.add(push_side)
                push_faces[push_side] = (0, outward, side_offset)
                touch_face(push_side, -outward, .016, f"{prefix}_{push_side}_touch")
                start, _ = sensed_cube()
                lateral = start.copy()
                # Slot 4's leading corner needs extra clearance past the seated neighboring box.
                clearance = .001 if slot == 3 else .0025
                lateral[0] = placement_target[0] - push_sign * (half * (abs(attitude[0]).sum() - 1.) + clearance)
                lateral_phase = f"{prefix}_{push_side}_lateral"
                record(lateral_phase, push_side.upper() + "_ARM_ONLY", target_x_m=float(lateral[0]))
                contact_path(push_side, start, lateral, lateral_phase, speed=.025, force_limit=10.)
                unload_normal(push_side, 0, push_sign, f"{prefix}_{push_side}_unload")
                push_faces.clear()
                tip, _ = tcp_fk(push_side, applied_targets[push_side])
                cartesian_move(push_side, tip - [push_sign * .014, 0., 0.], 2., f"{prefix}_{push_side}_retreat")
                allowed_contacts.clear()
                tip, _ = tcp_fk(push_side, applied_targets[push_side])
                cartesian_move(push_side, np.array([tip[0], tip[1], 1.05]), 5., f"{prefix}_{push_side}_lift")
                move({push_side: np.asarray(CONFIG["fr3"]["official_start_q_rad"])})
                aligned, attitude = observe_b_cube(f"{prefix}_after_lateral_rgb")
                lateral_gap = push_sign * (placement_target[0] - aligned[0]) - half * (abs(attitude[0]).sum() - 1.)
                yaw = math.degrees(math.atan2(attitude[1, 0], attitude[0, 0]))
                if not -.001 <= lateral_gap <= .004 or abs(yaw) > 1.5:
                    raise RuntimeError(f"Front {slot} lateral alignment outside the straight-entry range: gap={lateral_gap:.4f}, yaw={yaw:.2f}")
                evidence["lateral_alignment"] = {"rgb_side_gap_m": float(lateral_gap), "yaw_deg": yaw,
                                                  "target_x_m": float(lateral[0])}
                base = rail_states[support_side].copy()
                base[1] = (.60 if slot == 3 else .40) + rail_shift
                record(f"{prefix}_{support_side}_rail", "ALIGNED_AND_" + support_side.upper() + "_ARM_ONLY",
                       arm=support_side, y_m=float(base[1]))
                move_robot_base(support_side, base)
                front_rotation = np.asarray(FRONT_ROTATION) if support_side == "left" else np.diag([-1., 1., -1.])
                front_offset = np.array([0., -half, .004])
                target = aligned + front_offset - [0., .007, 0.]
                if slot == 4:
                    high = np.r_[target[:2], 1.05]
                    seed = np.array([-2.145056, .483576, 1.728614, -2.236692, -.554548, 2.082276, -.102048])
                    move({support_side: cartesian_joint_target(support_side, seed, high, front_rotation, retarget_guide=True)})
                    cartesian_move(support_side, target, 5., f"{prefix}_straight_descent")
                else:
                    seed = np.array([1.01259696, .59909831, -.74589782, -2.43871928, 1.05331681, 2.68562402, -.64496107])
                    move({support_side: cartesian_joint_target(support_side, seed, target, front_rotation, retarget_guide=True)})
                contact_plan_rotation = attitude.copy()
                allowed_contacts.add(support_side)
                push_faces[support_side] = (1, np.array([0., -1., 0.]), front_offset)
                touch_face(support_side, np.array([0., 1., 0.]), .016, f"{prefix}_{support_side}_touch")
                start, attitude = sensed_cube()
                straight = start.copy()
                straight[1] = placement_target[1] - half * (abs(attitude[1]).sum() - 1.) - .0015
                straight_phase = f"{prefix}_{support_side}_straight_insert"
                record(straight_phase, support_side.upper() + "_ARM_ONLY", x_m=float(start[0]))
                contact_path(support_side, start, straight, straight_phase, speed=.025, force_limit=10.)
                rear_path = f"/World/Cube_{slot}"
                probe_rear_face(support_side, verify_path=rear_path)
                verify_contact("rear_seat", [rear_path], support_side, 1, 1)
                unload_normal(support_side, 1, 1, f"{prefix}_{support_side}_unload")
                push_faces.clear()
                tip, _ = tcp_fk(support_side, applied_targets[support_side])
                cartesian_move(support_side, tip - [0., .035, 0.], 3., f"{prefix}_{support_side}_retreat")
                allowed_contacts.clear()
                park_for_final_rgb()
                return finish()

            if slot == 5:
                center, rotation = observe_b_cube("front_center_alignment_check")
                gaps = center_gaps(center, rotation)
                if min(gaps) < .001:
                    from fr3_bilateral_b2 import b2_role_seeds

                    correction = .5 * (gaps[1] - gaps[0])
                    if not -.003 <= correction < 0. or np.mean(gaps) < .001:
                        raise RuntimeError(f"Center slot cannot use the checked right-arm alignment: {gaps}")
                    set_contact_drive_mode(True)
                    original_base = rail_states["right"].copy()
                    base = original_base.copy()
                    base[1] = .50
                    record("front_center_align", "RIGHT_ARM_LATERAL_ONLY", gap_before_m=gaps, correction_m=float(correction))
                    move_robot_base("right", base)
                    guide = b2_role_seeds(json.loads((ROOT / "validation" / "fr3_b2_collision_plan.json").read_text()), "right")
                    offset = np.array([half, 0., .020])
                    outward = rotation[:, 0]
                    tool_rotation = rotation @ guide["push_rotation_world"]
                    target = center + rotation @ offset + .007 * outward
                    high = np.r_[target[:2], 1.05]
                    seed = np.array([-.365763, -.300486, -.207569, -2.516135, -.076668, 2.220312, -2.088649])
                    move({"right": cartesian_joint_target("right", seed, high, tool_rotation, retarget_guide=True)})
                    cartesian_move("right", target, 5., "front_center_align_descent")
                    contact_plan_rotation = rotation.copy()
                    allowed_contacts.add("right")
                    push_faces["right"] = (0, outward, offset)
                    touch_face("right", -outward, .016, "front_center_align_touch")
                    start, _ = sensed_cube()
                    contact_path("right", start, start + [correction, 0., 0.], "front_center_align_lateral",
                                 speed=.005, force_limit=10.)
                    unload_normal("right", 0, -1, "front_center_align_unload")
                    push_faces.clear()
                    tip, _ = tcp_fk("right", applied_targets["right"])
                    cartesian_move("right", tip + [.014, 0., 0.], 2., "front_center_align_retreat")
                    allowed_contacts.clear()
                    tip, _ = tcp_fk("right", applied_targets["right"])
                    cartesian_move("right", np.r_[tip[:2], 1.05], 5., "front_center_align_lift")
                    move({"right": np.asarray(CONFIG["fr3"]["official_start_q_rad"])})
                    move_robot_base("right", original_base)
                    center, rotation = observe_b_cube("front_center_after_align")
                    gaps = center_gaps(center, rotation)
                    if min(gaps) < .001:
                        raise RuntimeError(f"Center slot clearance remains below 1 mm: {gaps}")
                    record("front_center_align", "RGB_CLEARANCE_CONFIRMED", gap_after_m=gaps)
'''
    first = header.index('                cube_start, attitude = observe_b_cube(f"{prefix}_landing_for_lateral")')
    end = header.index('                rear_path = f"/World/Cube_{slot}"', first)
    header = header[:first] + '''                direct_rear = CHECKPOINT.get("front_direct_rear") if slot == START_FRONT else None
                if direct_rear:
                    if direct_rear["cube_id"] != current_task["cube_id"] or direct_rear["slot"] != slot:
                        raise RuntimeError("Mismatched direct-push endpoint checkpoint")
                    cube_start = np.asarray(direct_rear["cube_start_rgb"])
                    evidence["lateral_alignment"] = direct_rear["lateral_alignment"]
                    contact_plan_rotation = control_rotation.copy()
                    allowed_contacts.add(support_side)
                    push_faces[support_side] = (1, np.array([0., -1., 0.]), np.array([0., -half, .004]))
                    record(f"{prefix}_rear_resume", "RESTORED_STRAIGHT_ENDPOINT", source_step=direct_rear["source_step"])
                    touch_face(support_side, np.array([0., 1., 0.]), .005, f"{prefix}_{support_side}_resume_touch")
                else:
''' + indent(header[first:end], '    ') + header[end:]
    # half belongs to the surrounding runner. The floor B path reads it before the copied body.
    body = replace_once(body, '            half = CONFIG["cube_size_m"] / 2\n', '')
    anchor = '        def execute_upper_push('
    return replace_once(source, anchor, park + header + body + '            return finish()\n\n' + anchor)
