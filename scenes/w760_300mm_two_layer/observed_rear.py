"""Bring the rear row into the same markerless runtime as the front and upper rows."""

import numpy as np

from observed_transport import wall_plane
from source_hooks import replace_once


SIDE_TARGETS = ("wall_left", "wall_right", "Cube_1", "Cube_2", None)
# Relative clearance lanes from the high-wall floor acceptance, in slot order.
LANDING_LATERAL_FROM_SEAT_M = (.305, -.260, .12585, -.04900345, 0.)
LANDING_TO_REAR_WALL_M = (.415, .415, .415, .41623417, .41593692)


def center_slot_edges(poses, half):
    left, left_rotation = poses["Cube_3"]
    right, right_rotation = poses["Cube_4"]
    return (left[0] + half * np.abs(left_rotation[0]).sum(),
            right[0] - half * np.abs(right_rotation[0]).sum())


def rear_side_geometry(config, slot, poses):
    """Return the calibrated wall or observed neighbor plane and the direction into it."""
    half = config["cube_size_m"] / 2
    name = SIDE_TARGETS[slot - 1]
    sign = -1 if slot in (1, 3) else 1
    if slot in (1, 2):
        plane = wall_plane(config, name, 0, sign)
    elif name is not None:
        center, rotation = poses[name]
        plane = center[0] - sign * half * np.abs(rotation[0]).sum()
    else:
        plane = sum(center_slot_edges(poses, half)) / 2
    final_x = plane - sign * half if name is not None else plane
    return name, plane, final_x


def rear_landing(config, slot, poses):
    _, _, final_x = rear_side_geometry(config, slot, poses)
    return np.array([final_x + LANDING_LATERAL_FROM_SEAT_M[slot - 1],
                     wall_plane(config, "wall_back", 1, 1) - LANDING_TO_REAR_WALL_M[slot - 1],
                     config["floor_top_m"] + config["cube_size_m"] / 2])


def install(source):
    hook = '        def save_front_checkpoint('
    source = replace_once(source, hook, REAR_BODY + '\n' + hook)
    source = replace_once(source, '            if phase == "A_pickup_pose":\n',
                          '''            if phase == "rear_landing_resume":
                return np.asarray(CHECKPOINT["rear_landed"]["position_m"]), None
            if phase == "A_pickup_pose":
''')
    source = replace_once(source, '            snapshot = contact_monitor.cube_contact_snapshot()\n'
                          '            return sum((row.get("normal_impulse_Ns") or 0.) for row in snapshot["pairs"]\n'
                          '                       if row["current_step_observed"]\n'
                          '                       and active_cube_path in (row["collider0"], row["collider1"])\n'
                          '                       and ({row["collider0"], row["collider1"]} - {active_cube_path}) <= LOWER_PATHS)',
                          '''            snapshot = contact_monitor.cube_contact_snapshot()
            support_paths = {"/World/Environment/floor"} if current_task.get("rear_slot") else LOWER_PATHS
            return sum((row.get("normal_impulse_Ns") or 0.) for row in snapshot["pairs"]
                       if row["current_step_observed"]
                       and active_cube_path in (row["collider0"], row["collider1"])
                       and ({row["collider0"], row["collider1"]} - {active_cube_path}) <= support_paths)''')
    source = replace_once(source, '                                        or current_task.get("front_slot") in (2, 4))',
                          '                                        or current_task.get("front_slot") in (2, 4)\n'
                          '                                        or current_task.get("rear_slot") in (2, 4))')
    source = replace_once(source, '            cruise = cruise and current_task.get("front_slot") not in (2, 4)',
                          '            cruise = cruise and current_task.get("front_slot") not in (2, 4)\n'
                          '            cruise = cruise and current_task.get("rear_slot") not in (2, 4)')
    source = replace_once(source, '            guides_start = {side: tcp_fk(side, guide_q[side][0])[0] for side in robots}',
                          '''            paired_offsets = {}
            if carrying and (current_task.get("rear_slot") == 3 or current_task.get("front_slot") == 3
                             or current_task.get("slot") in (2, 3, 5)):
                from scipy.spatial.transform import Rotation

                # Keep both commanded tool poses consistent with the same grasped box pose.
                # Reseeding each segment's wrist reference independently can accumulate internal twist.
                for side, (offset, relative_rotation) in carry_offsets.items():
                    measured_tcp, measured_rotation = actual_tool[side]
                    attitude = payload_rotation @ relative_rotation.T
                    position = start_center - attitude @ offset
                    paired_offsets[side] = (position - measured_tcp,
                        Rotation.from_matrix(attitude @ measured_rotation.T).as_rotvec())
            guides_start = {side: tcp_fk(side, guide_q[side][0])[0] for side in robots}''')
    source = replace_once(source, '                                target = measured_tcp + payload_path[index] - start_center',
                          '''                                target = measured_tcp + payload_path[index] - start_center
                                if paired_offsets:
                                    translation, angular = paired_offsets[side]
                                    progress = min(1., index / 30.)
                                    blend = progress**3 * (10 - 15 * progress + 6 * progress**2)
                                    target += blend * translation
                                    rotation = Rotation.from_rotvec(blend * angular).as_matrix() @ rotation''')
    source = replace_once(source, '                "duration_sim_s": frames / 120.,',
                          '''                "duration_sim_s": frames / 120.,
                "shared_grasp_reference": bool(paired_offsets),''')
    source = replace_once(source, '                            speed = .00010 if force < 1.6 else .000020',
                          '''                            slow_load = max(1.6, probe.get("sliding_baseline_N", 0.) + .5)
                            speed = .00010 if force < slow_load else .000020''')
    return replace_once(source, '            if front_slot != START_FRONT:\n',
                        '            if front_slot != START_FRONT or START_REAR < 6:\n')


REAR_BODY = '''
        def run_observed_rear(slot, neighbors):
            from collections import deque
            from fr3_bilateral_b2 import b2_role_seeds
            from observed_transport import wall_plane

            nonlocal support_side, push_side, push_sign, left_hold, left_hold_z, left_hold_rotation
            nonlocal contact_plan_rotation
            # A restored box is resting on the floor, with no holding wrist.
            allowed_contacts.clear()
            rear_arm = "left" if slot in (1, 3, 5) else "right"
            lateral = "right" if slot in (1, 3) else "left"
            sign = -1 if lateral == "right" else 1
            support_side, push_side, push_sign = rear_arm, lateral, sign
            rear_y = wall_plane(CONFIG, "wall_back", 1, 1)
            final_forward_resume = CHECKPOINT.get("rear_forward_resume") if slot == START_REAR and slot == 4 else None
            lateral_resume = CHECKPOINT.get("rear_lateral_resume") if slot == START_REAR and slot == 2 else None
            postures = json.loads((ROOT / "scenes/w760_300mm/b_postures.json").read_text())
            outer = postures["outer"].get(str(slot))
            third = postures["third"] if slot == 3 else None
            from observed_rear import center_slot_edges, rear_side_geometry
            tracked = {name: (center, rotation) for name, center, rotation in neighbors}
            side_name, side_x, target_x = rear_side_geometry(CONFIG, slot, tracked)
            target = np.array([target_x, rear_y - half, SUPPORT_TOP + half])

            def confirm_wall_bearing(arm, axis, direction, boundary, target_name, follower=None):
                phase = f"rear_{slot}_B2_{axis}_bearing"
                record(phase, "RUNNING", target=target_name)
                q = applied_targets[arm].copy()
                tip, attitude = tcp_fk(arm, q)
                start = tip[axis]
                if follower is not None:
                    follow_q = applied_targets[follower].copy()
                    follow_tip, follow_attitude = tcp_fk(follower, follow_q)
                # Inner slots' rear contact rises on the same timescale as the lateral wall contacts.
                short_window = axis == 0 or slot >= 3
                window = deque(maxlen=19 if short_window else 61)
                average_count = 4 if short_window else 8
                min_advance = .000012 if short_window else .00004
                if slot == 4 and axis == 1:
                    # The wrench envelope limits advance while the neighboring box is already bearing.
                    min_advance = .000004
                max_tracking_ratio = .45 if short_window else .35
                extended_terminal = axis == 0 or slot != 1
                travel_limit = .012 if slot == 5 else .009 if extended_terminal else .004
                gap_limit = (B2_CONTROLLER_PARAMETERS["seating_gap_tolerance_m"]
                             if slot == 2 and axis == 1 else .00045)
                for _ in range(5400 if slot == 5 else 3600):
                    center, rotation = sensed_cube()
                    measured, _ = tcp_fk(arm, joint_state()[arm])
                    wrench = last_force_sample[arm]["tcp_gravity_corrected_world"]
                    total = np.linalg.norm(wrench[:3])
                    normal = -direction * float(wrench[axis])
                    gap = direction * (boundary - center[axis]) - half * np.abs(rotation[axis]).sum()
                    if total > 9. or np.linalg.norm(wrench[3:]) > 1. or gap < -.002:
                        raise RuntimeError(f"{phase}: exceeded force or geometry envelope")
                    window.append((direction * tip[axis], direction * measured[axis], normal))
                    if len(window) == window.maxlen:
                        values = np.asarray(window)
                        commanded, response, rise = (values[-average_count:].mean(axis=0)
                                                     - values[:average_count].mean(axis=0))
                        if (-.002 <= gap <= gap_limit and commanded >= min_advance
                                and abs(response) <= max_tracking_ratio * commanded
                                and rise >= .30 and normal >= 5.5):
                            record(phase, "FORCE_CONFIRMED", contact_target=target_name, normal_force_N=normal,
                                   estimated_gap_m=float(gap), command_advance_m=float(commanded),
                                   encoder_advance_m=float(response), force_rise_N=float(rise))
                            return
                    increment = 0.
                    if total < 7.7:
                        speed = .0005 if extended_terminal and gap > .001 else .00012
                        increment = direction * speed / 120
                        tip[axis] += increment
                    if abs(tip[axis] - start) > travel_limit:
                        break
                    candidate = cartesian_joint_target(arm, q, tip, attitude)
                    if np.max(np.abs(candidate - q)) * 120 > .4:
                        raise RuntimeError(f"{phase}: exceeded joint speed limit")
                    q = candidate
                    command_positions(arm, q)
                    if follower is not None and increment:
                        follow_tip[axis] += increment
                        follow_q = cartesian_joint_target(follower, follow_q, follow_tip, follow_attitude)
                        command_positions(follower, follow_q)
                    step()
                    check_current_collision(phase)
                raise RuntimeError(f"{phase}: no verified stiffness response (gap {gap*1000:.2f} mm, "
                                   f"force {normal:.2f} N)")

            def approach(arm, axis, direction):
                from observed_b1 import FACE_STANDOFF_M

                center, rotation = observe_resting_cube(
                    f"rear_{slot}_{arm}_face_pose", vision_camera, rows, SUPPORT_TOP)
                inward = direction * rotation[:, axis]
                centered_rear2 = slot == 2 and axis == 1
                height = .0 if centered_rear2 else .004 if axis == 1 else .020
                offset = -direction * half * np.eye(3)[axis] + [0., 0., height]
                if axis == 1:
                    tool = np.asarray(FRONT_ROTATION) if arm == "left" else np.diag([-1., 1., -1.])
                    hint = (outer["b1_q"] if outer is not None
                            else third["precontact_q_rad"] if third is not None else None)
                    if final_forward_resume:
                        hint = final_forward_resume["right_wrist_seed_rad"]
                else:
                    role = b2_role_seeds(json.loads((ROOT / "validation/fr3_b2_collision_plan.json").read_text()), arm)
                    tool = role["push_rotation_world"]
                    hint = role["rail_hold_push_q"]
                rear2_lateral_entry = slot == 2 and axis == 0
                distance = FACE_STANDOFF_M if slot == 2 else .010
                destination = center + rotation @ offset - distance * inward
                if rear2_lateral_entry:
                    hint = min(outer["b2_entry_q"],
                               key=lambda q: np.linalg.norm(tcp_fk(arm, np.asarray(q))[0] - destination))
                neighbor_entry = axis == 0 and slot in (3, 4)
                if neighbor_entry:
                    guide = json.loads((ROOT / "validation/fr3_row_b2_final_feasibility.json").read_text())
                    case = next(case for case in guide["cases"] if case["cube_id"] == slot)
                    hint = case["precontact_q"][arm]
                if lateral_resume:
                    hint = lateral_resume["wrist_seeds_rad"][arm]
                descend_to_side = slot in (2, 3) and axis == 0
                margin = .003 if descend_to_side else .002
                move_to_observed_precontact(arm, destination, rotation @ tool, f"rear_{slot}_B1_approach", hint,
                                            prefer_hint=rear2_lateral_entry or neighbor_entry
                                            or (axis == 1 and hint is not None), planning_margin_m=margin,
                                            approach_lift_m=.243 if descend_to_side else 0.)
                if not allowed_contacts:
                    set_contact_drive_mode(True)
                allowed_contacts.add(arm)
                push_faces[arm] = (axis, -inward, offset)
                touch_face(arm, inward, .018, f"rear_{slot}_B1_touch")

            def advance(arm, axis, direction, boundary, target_name, follower=None):
                start, rotation = sensed_cube()
                measured_slide = {}
                goal = start.copy()
                standoff = .005 if slot != 1 and target_name == "wall_back" else .001
                if slot == 5 and target_name == "wall_back":
                    standoff = .009
                goal[axis] = boundary - direction * (half * np.abs(rotation[axis]).sum() + standoff)
                if target_name is None:
                    goal[axis] = boundary
                if direction * (goal[axis] - start[axis]) > .0001:
                    if follower is None:
                        phase = f"rear_{slot}_B2_{axis}_travel"
                        if target_name is None:
                            # Ease into the observed lateral staging lane before turning across it.
                            if direction * (goal[axis] - start[axis]) > .005:
                                lead = goal.copy()
                                lead[axis] -= direction * .005
                                contact_path(arm, start, lead, phase, speed=.018, force_limit=9.)
                                start, _ = sensed_cube()
                            contact_path(arm, start, goal, phase + "_stage", speed=.004, force_limit=9.)
                        else:
                            contact_path(arm, start, goal, phase, speed=.018, force_limit=9.)
                    else:
                        # Keep the command-to-encoder preload established by contact synchronization.
                        phase = f"rear_{slot}_B2_side_travel"
                        stop = goal.copy()
                        terminal_gap = .001 if slot == 4 else .004
                        stop[axis] -= direction * min(terminal_gap, abs(goal[axis] - start[axis]) / 2)
                        seeds = {active: applied_targets[active].copy() for active in (arm, follower)}
                        tips = {active: tcp_fk(active, q) for active, q in seeds.items()}
                        count = int(round(B2_CONTROLLER_PARAMETERS["push_duration_s"] * 120)) + 1
                        times = np.arange(count) / 120
                        fraction = times / times[-1]
                        translation = ((10 - 15*fraction + 6*fraction**2)*fraction**3)[:, None] * (stop - start)
                        paths = {active: [] for active in seeds}
                        record(phase, "RUNNING", arm=arm, follower=follower)
                        for delta in translation:
                            planning_update(f"{phase}: coordinated contact path")
                            for active, (tip, tool) in tips.items():
                                q = cartesian_joint_target(active, seeds[active], tip + delta, tool, retarget_guide=True)
                                seeds[active] = q
                                paths[active].append(q)
                        samples = {active: np.asarray(path) for active, path in paths.items()}
                        if max(float(np.max(np.abs(np.diff(path, axis=0)))) for path in samples.values()) * 120 > .4:
                            raise RuntimeError(f"{phase}: exceeded 0.4 rad/s while planning the slide")

                        axial = 0.
                        relief = np.zeros(3)
                        tangential_limit = .004 if slot == 2 else .002
                        nominal_axial = direction * (tips[arm][0][axis] + translation[:, axis])
                        allowed_lead = .001

                        def measured_lead(index):
                            measured = last_force_sample[arm]["tcp_position_m"][axis]
                            return nominal_axial[index] + relief[0] - direction * measured

                        def shifted_lateral_commands(commands):
                            adjusted = {}
                            for active in (arm, follower):
                                tip, tool = tcp_fk(active, commands[active])
                                offset = np.array([direction * relief[0], 0., 0.])
                                if active == arm:
                                    offset[1:] = relief[1:]
                                adjusted[active] = cartesian_joint_target(active, commands[active], tip + offset, tool)
                            return adjusted

                        def level_push(index, commands):
                            nonlocal axial, relief
                            wrench = last_force_sample[arm]["tcp_gravity_corrected_world"]
                            axial += .25 * (-direction * wrench[0] - axial)
                            vertical = wrench[2] + 1.8
                            deadband = math.copysign(max(abs(vertical) - .25, 0.), vertical)
                            speeds = [np.clip(.001 * (6.5 - axial), -.003, .001),
                                      np.clip(.0007 * wrench[1], -.001, .001),
                                      np.clip(.001 * deadband, -.003, .003)]
                            relief = np.clip(relief + np.asarray(speeds) / 120,
                                             [-.003, -tangential_limit, -.002], [0., tangential_limit, .006])
                            relief[0] -= max(0., measured_lead(index) - allowed_lead)
                            center, attitude = sensed_cube()
                            options = dict(cube_center=center, cube_rotation=attitude, cup_contacts=allowed_contacts,
                                           margin_m=0., first_only=True)
                            adjusted = shifted_lateral_commands(commands)
                            issues = checker.check(adjusted, **options)
                            if issues and issues[0]["kind"] == "tool_obstacle" and issues[0]["b"] == "Cube" and (
                                    issues[0]["a"].startswith(f"{arm}/cup_") and arm in allowed_contacts):
                                # Give back any axial lead that would exceed the permitted cup contact depth.
                                relief[0] += issues[0]["clearance_m"] + .00290
                                adjusted = shifted_lateral_commands(commands)
                                issues = checker.check(adjusted, **options)
                            if issues:
                                raise RuntimeError(f"{phase}: contact correction blocked by {issues[0]}")
                            if any(np.max(np.abs(q - applied_targets[a])) * 120 > .4 for a, q in adjusted.items()):
                                raise RuntimeError(f"{phase}: contact correction exceeded joint speed")
                            return adjusted

                        def advance_under_load(count):
                            for index in range(count):
                                yield index
                                for _ in range(240):
                                    force = last_force_sample[arm]["tcp_gravity_corrected_world"][:3]
                                    follow_index = min(index + 1, count - 1)
                                    if (axial < 7.2 and np.linalg.norm(force) < 7.8
                                            and measured_lead(follow_index) <= allowed_lead):
                                        break
                                    yield index
                                else:
                                    raise RuntimeError(f"{phase}: lateral travel remained blocked under load")

                        def check_load(index, count):
                            if slot == 4 and .2 * count <= index <= .7 * count and index not in measured_slide:
                                measured_slide[index] = -direction * last_force_sample[arm]["tcp_gravity_corrected_world"][0]
                            for active in robots:
                                force = np.asarray(last_force_sample[active]["tcp_gravity_corrected_world"][:3])
                                if np.linalg.norm(force) > 9.:
                                    record(phase, "WRIST_LIMIT", arm=active, force_N=force.tolist(),
                                           lateral_relief_m=relief.tolist())
                                    raise RuntimeError(f"Rear slot {slot} coordinated push exceeded wrist load")

                        execute_joint_path(samples, phase, check_step=check_load, adjust_targets=level_push,
                                           sample_order=advance_under_load,
                                           cube_center_path=lambda _samples, _times: start + translation,
                                           sampled_trajectory=(times, samples))
                if target_name in ("wall_back", "wall_left", "wall_right"):
                    confirm_wall_bearing(arm, axis, direction, boundary, target_name, follower)
                elif target_name is not None:
                    probe = None
                    if slot == 4:
                        baseline = float(np.percentile(list(measured_slide.values()), 90)) if measured_slide else 0.
                        probe = {"sliding_baseline_N": baseline, "entry_force_N": max(2., baseline + .5),
                                 "entry_samples": 3, "samples": 30, "minimum_force_N": 1.5,
                                 "mean_force_N": max(1.8, baseline + .4), "max_axial_travel_m": .0005,
                                 "target_uncertainty_m": .003}
                    seat_against_wall(arm, axis, direction, boundary - direction * half,
                                      f"rear_{slot}_B2_{axis}_bearing", contact_target=target_name,
                                      follow_side=follower, probe_override=probe)

            def clear(arm, direction):
                nonlocal left_hold
                if arm == rear_arm:
                    left_hold = None
                dynamic_withdraw({arm: np.asarray(direction)}, f"rear_{slot}_{arm}_withdraw")
                push_faces.pop(arm, None)
                allowed_contacts.discard(arm)
                move({arm: np.asarray(CONFIG["fr3"]["official_start_q_rad"])})

            def finish_placement():
                nonlocal left_hold
                phase = f"rear_{slot}_B2_settle"
                record(phase, "WAITING_FOR_STABLE_SEATING")
                history = deque(maxlen=30)
                for _ in range(240):
                    center, attitude = sensed_cube()
                    history.append(np.r_[center, math.atan2(attitude[1, 0], attitude[0, 0])])
                    spread = np.ptp(np.asarray(history), axis=0)
                    if (len(history) == history.maxlen and np.linalg.norm(center - target) <= .005
                            and np.linalg.norm(spread[:3]) <= .0003 and spread[3] <= math.radians(.3)):
                        record(phase, "POSE_STABLE", estimated_center_m=center.tolist(),
                               position_spread_m=float(np.linalg.norm(spread[:3])))
                        break
                    step()
                    check_current_collision(phase)
                else:
                    raise RuntimeError(f"Rear slot {slot} did not stabilize at the seating target")
                left_hold = None
                outward = {rear_arm: np.array([0., -1., 0.])}
                if side_name is not None and lateral in allowed_contacts:
                    outward[lateral] = np.array([-sign, 0., 0.])
                dynamic_withdraw(outward, f"rear_{slot}_final_withdraw", require_support=True)
                push_faces.clear()
                allowed_contacts.clear()
                set_contact_drive_mode(False)
                record(f"rear_{slot}_repark", "RRTCONNECT_FROM_SEPARATION", arms=list(robots))
                move({arm: np.asarray(CONFIG["fr3"]["official_start_q_rad"]) for arm in robots})

            stations = {rear_arm: .4}
            if side_name is not None:
                stations[lateral] = .8
            if outer is not None:
                stations = {rear_arm: outer["support_y"], lateral: outer["push_y"]}
            elif third is not None:
                # Reuse the support branch checked through the full lateral slide.
                stations[rear_arm] = third["support_rail_y_m"]
            nominal_wall = CONFIG["goal_cube_centers_m"][0][1] + half
            for arm, station in stations.items():
                base = rail_states[arm].copy()
                base[1] = station + rear_y - nominal_wall
                if not np.isclose(rail_states[arm][1], base[1]):
                    move_robot_base(arm, base)
            if slot == 5:
                edges = center_slot_edges(tracked, half)
                center, rotation = observe_resting_cube("rear_center_entry", vision_camera, rows, SUPPORT_TOP)
                extent = half * np.abs(rotation[0]).sum()
                if min(center[0] - extent - edges[0], edges[1] - center[0] - extent) < .001:
                    raise RuntimeError("Rear center insertion has insufficient visually verified slot clearance")
            contact_plan_rotation = control_rotation.copy()
            approach(rear_arm, 1, 1)
            if slot == 4 and not final_forward_resume:
                neighbor_center, neighbor_attitude = tracked[side_name]
                near_face = neighbor_center[1] - half * np.abs(neighbor_attitude[1]).sum()
                _, attitude = sensed_cube()
                leading_extent = half * np.abs(attitude[1]).sum()
                side_push_overlap = .044
                advance(rear_arm, 1, 1, near_face + side_push_overlap - leading_extent, None)
            else:
                advance(rear_arm, 1, 1, rear_y, "wall_back")
            if side_name and not final_forward_resume:
                position, left_hold_rotation = tcp_fk(rear_arm, joint_state()[rear_arm])
                left_hold_z = float(position[2])
                support_load = .65 if slot == 4 else B2_CONTROLLER_PARAMETERS["left_target_force_N"]
                left_hold = NormalForceController(position[1], target_force_N=support_load, gain_m_per_Ns=.0001,
                                                   max_offset_m=.0035, max_speed_m_s=.0004)
                approach(lateral, 0, sign)
                settle_b2_contacts(lateral, sign, f"rear_{slot}_B2_contact_sync", support_target=support_load)
                advance(lateral, 0, sign, side_x, side_name, follower=rear_arm)
                if slot == 4:
                    clear(lateral, [-sign, 0., 0.])
                    left_hold = None
                    touch_face(rear_arm, contact_plan_rotation[:, 1], .006, f"rear_{slot}_right_retouch")
                    advance(rear_arm, 1, 1, rear_y, "wall_back")
            finish_placement()
            final, attitude = observe_resting_cube("rear_final_pose", vision_camera, rows, SUPPORT_TOP)
            stable = support_window()
            if np.linalg.norm(final - target) > .010 or stable["positive_steps"] < 48:
                raise RuntimeError(f"Rear slot {slot} failed placement: observed {final}, target {target}")
            rows.placed[current_task["cube_id"]] = (final.copy(), attitude.copy())
            result = {"cube_id": current_task["cube_id"], "slot": slot,
                      "target_m": target.tolist(), "observed_position_m": final.tolist(), "support": stable}
            if slot == 5:
                sides = observe_known_lower({"Cube_3", "Cube_4"})
                left_edge, right_edge = center_slot_edges(
                    {name: (center, rotation) for name, center, rotation in sides}, half)
                extent = half * np.abs(attitude[0]).sum()
                result["observed_slot_gaps_m"] = {"left": float(final[0] - extent - left_edge),
                                                 "right": float(right_edge - final[0] - extent)}
            return result

        if START_REAR < 6:
            from front_row_a import floor_guide
            from observed_rear import rear_landing

            rear_results = list(CHECKPOINT.get("rear_completed", []))
            report["rear_row"] = {"completed": rear_results}

            def save_rear_checkpoint(next_slot, landed=None):
                bodies = upper_pose_records(lower_bodies)
                data = {"scene": SCENE, "next_rear": next_slot, "next_front": 1, "next_slot": 1,
                    "initial_boxes": bodies, "rear_completed": rear_results, "rear_initial": bodies,
                    "rear_landed": landed, "front_completed": [], "completed_slots": [],
                    "markerless_placements": {str(key): [p.tolist(), r.tolist()] for key, (p, r) in rows.placed.items()}}
                path = output_directory / ("rear_landing_checkpoint.json" if landed else "rear_row_checkpoint.json")
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\\n")
                tmp.replace(path)

            for rear_slot in range(START_REAR, 6):
                task_id = rear_slot
                active_cube_path = f"/World/Cube_{task_id}"
                current_task = {"cube_id": task_id, "marker_id": task_id + 22, "rear_slot": rear_slot}
                if rear_slot != START_REAR:
                    next_cube = spawn_supply_cube(stage, world, CONFIG, task_id - 1, contact_monitor)
                    supply_transition = True
                    step()
                    next_cube.initialize()
                    cube = next_cube
                    supply_transition = False
                truth_history.clear()
                control_center = control_rotation = carry_offsets = contact_plan_rotation = None
                supported_pose_time = None
                supported_pose_filter = PlanarPoseMedian()
                step(120)
                neighbors = observe_known_lower() if lower_bodies else []
                checker.set_other_cubes(neighbors)
                GUIDE = floor_guide(rear_slot)
                FIRST_LOADED = next(i for i, row in enumerate(GUIDE["phases"]) if row["carrying"])
                tracked = {name: (center, rotation) for name, center, rotation in neighbors}
                current_task["live_landing_m"] = rear_landing(CONFIG, rear_slot, tracked)
                report["upper_row"] = {}
                if rear_slot == START_REAR and CHECKPOINT.get("rear_landed"):
                    observe_resting_cube("rear_landing_resume", vision_camera, rows, SUPPORT_TOP)
                    record("rear_landing_resume", "SENSOR_POSE_CONFIRMED")
                else:
                    run_cube()
                    save_rear_checkpoint(rear_slot, upper_pose_records({str(task_id): cube})[0])
                rear_results.append(run_observed_rear(rear_slot, neighbors))
                lower_bodies[str(task_id)] = cube
                record("rear_slot_complete", "SEATED_STABLE", slot=rear_slot)
                save_rear_checkpoint(rear_slot + 1)
                if rear_slot == STOP_AFTER_REAR:
                    report["status"] = "PASS_REAR_SLOT"
                    report["ab_status"] = "SEATED_STABLE"
                    return 0
            rear_initial = upper_pose_records(lower_bodies)
            rear_reference = {str(b["cube_id"]): np.asarray(b["position_m"]) for b in rear_initial}
            report["front_row"]["rear_initial"] = rear_initial
'''
