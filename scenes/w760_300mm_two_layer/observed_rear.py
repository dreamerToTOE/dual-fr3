"""Bring the rear row into the same markerless runtime as the front and upper rows."""

from source_hooks import replace_once


def install(source):
    hook = '        def save_front_checkpoint(next_front, landed=None):'
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
    return replace_once(source, '            if front_slot != START_FRONT:\n',
                        '            if front_slot != START_FRONT or START_REAR < 6:\n')


REAR_BODY = '''
        def run_observed_rear(slot, neighbors):
            from collections import deque
            from fr3_bilateral_b2 import b2_role_seeds
            from observed_transport import wall_plane

            nonlocal support_side, push_side, push_sign, left_hold, left_hold_z, left_hold_rotation
            nonlocal contact_plan_rotation
            rear_arm = "left" if slot in (1, 3, 5) else "right"
            lateral = "right" if slot in (1, 3) else "left"
            sign = -1 if lateral == "right" else 1
            support_side, push_side, push_sign = rear_arm, lateral, sign
            rear_y = wall_plane(CONFIG, "wall_back", 1, 1)
            side_name = {1: "wall_left", 2: "wall_right", 3: "Cube_1", 4: "Cube_2"}.get(slot)
            tracked = {name: (center, rotation) for name, center, rotation in neighbors}
            if slot in (1, 2):
                side_x = wall_plane(CONFIG, side_name, 0, sign)
            elif slot in (3, 4):
                p, r = tracked[side_name]
                side_x = p[0] - sign * half * np.abs(r[0]).sum()
            else:
                a, r = tracked["Cube_3"]
                b, s = tracked["Cube_4"]
                edges = [a[0] + half * np.abs(r[0]).sum(), b[0] - half * np.abs(s[0]).sum()]
                side_x = sum(edges) / 2
            target = np.array([side_x - sign * half if side_name else side_x, rear_y - half, SUPPORT_TOP + half])

            def confirm_rear_bearing(arm):
                phase = f"rear_{slot}_B2_1_bearing"
                record(phase, "RUNNING", target="wall_back")
                q = applied_targets[arm].copy()
                tip, attitude = tcp_fk(arm, q)
                start_y = tip[1]
                window = deque(maxlen=61)
                for _ in range(3600):
                    center, rotation = sensed_cube()
                    measured, _ = tcp_fk(arm, joint_state()[arm])
                    wrench = last_force_sample[arm]["tcp_gravity_corrected_world"]
                    total = np.linalg.norm(wrench[:3])
                    normal = -float(wrench[1])
                    gap = rear_y - center[1] - half * np.abs(rotation[1]).sum()
                    if total > 9. or np.linalg.norm(wrench[3:]) > 1. or gap < -.002:
                        raise RuntimeError(f"{phase}: exceeded force or geometry envelope")
                    window.append((tip[1], measured[1], normal))
                    if len(window) == window.maxlen:
                        values = np.asarray(window)
                        commanded, response, rise = values[-8:].mean(axis=0) - values[:8].mean(axis=0)
                        if (-.002 <= gap <= .00045 and commanded >= .00004
                                and abs(response) <= .35 * commanded
                                and rise >= .30 and normal >= 5.5):
                            record(phase, "FORCE_CONFIRMED", contact_target="wall_back", normal_force_N=normal,
                                   estimated_gap_m=float(gap), command_advance_m=float(commanded),
                                   encoder_advance_m=float(response), force_rise_N=float(rise))
                            return
                    if total < 7.7:
                        tip[1] += .00012 / 120
                    if tip[1] - start_y > .004:
                        break
                    candidate = cartesian_joint_target(arm, q, tip, attitude)
                    if np.max(np.abs(candidate - q)) * 120 > .4:
                        raise RuntimeError(f"{phase}: exceeded joint speed limit")
                    q = candidate
                    command_positions(arm, q)
                    step()
                    check_current_collision(phase)
                raise RuntimeError(f"{phase}: no verified stiffness response (gap {gap*1000:.2f} mm, "
                                   f"force {normal:.2f} N)")

            def approach(arm, axis, direction):
                center, rotation = observe_resting_cube(
                    f"rear_{slot}_{arm}_face_pose", vision_camera, rows, SUPPORT_TOP)
                inward = direction * rotation[:, axis]
                offset = -direction * half * np.eye(3)[axis] + [0., 0., .004 if axis == 1 else .020]
                if axis == 1:
                    tool = np.asarray(FRONT_ROTATION) if arm == "left" else np.diag([-1., 1., -1.])
                    hint = None
                else:
                    role = b2_role_seeds(json.loads((ROOT / "validation/fr3_b2_collision_plan.json").read_text()), arm)
                    tool = role["push_rotation_world"]
                    hint = role["rail_hold_push_q"]
                destination = center + rotation @ offset - .010 * inward
                move_to_observed_precontact(arm, destination, rotation @ tool, f"rear_{slot}_B1_approach", hint)
                if not allowed_contacts:
                    set_contact_drive_mode(True)
                allowed_contacts.add(arm)
                push_faces[arm] = (axis, -inward, offset)
                touch_face(arm, inward, .018, f"rear_{slot}_B1_touch")

            def advance(arm, axis, direction, boundary, target_name, follower=None):
                start, rotation = sensed_cube()
                goal = start.copy()
                goal[axis] = boundary - direction * (half * np.abs(rotation[axis]).sum() + .001)
                if target_name is None:
                    goal[axis] = boundary
                if direction * (goal[axis] - start[axis]) > .0001:
                    if follower is None:
                        contact_path(arm, start, goal, f"rear_{slot}_B2_{axis}_travel", speed=.018, force_limit=9.)
                    else:
                        # Increment both TCPs together while the front wrist holds rear pressure.
                        position, rotation = tcp_fk(arm, joint_state()[arm])
                        other, other_rotation = tcp_fk(follower, joint_state()[follower])
                        height = NormalForceController(0., target_force_N=0., gain_m_per_Ns=.00055,
                                                       max_offset_m=.0022, max_speed_m_s=.001, max_force_N=9.)
                        direction_vector = np.eye(3)[axis]
                        distance = goal[axis] - start[axis]
                        samples = max(120, math.ceil(abs(distance)*1.875/.010*120))
                        record(f"rear_{slot}_B2_side_travel", "RUNNING", arm=arm, follower=follower)
                        for fraction in np.linspace(0., 1., samples + 1):
                            delta = (10 - 15*fraction + 6*fraction**2)*fraction**3 * distance * direction_vector
                            vertical = float(last_force_sample[arm]["tcp_gravity_corrected_world"][2])
                            # Positive Z reaction means the cup is adding downward load to the box.
                            vertical_error = math.copysign(max(0., abs(vertical) - .20), vertical)
                            correction = np.array([0., 0., height.update(-vertical_error)])
                            commands = joint_state()
                            for active, tip, tool in ((arm, position + correction, rotation),
                                                      (follower, other, other_rotation)):
                                q = cartesian_joint_target(active, applied_targets[active], tip + delta, tool)
                                commands[active] = q
                            estimated_center, estimated_rotation = sensed_cube()
                            issues = checker.check(commands, cube_center=estimated_center,
                                                   cube_rotation=estimated_rotation, cup_contacts=allowed_contacts,
                                                   margin_m=0., first_only=True)
                            if issues:
                                raise RuntimeError(f"Rear slot {slot} lateral segment blocked: {issues[0]}")
                            for active, q in commands.items():
                                command_positions(active, q)
                            step()
                            check_current_collision(contact_monitor.phase)
                            for a in robots:
                                force = np.asarray(last_force_sample[a]["tcp_gravity_corrected_world"][:3])
                                if np.linalg.norm(force) > 9.:
                                    record(contact_monitor.phase, "WRIST_LIMIT", arm=a, force_N=force.tolist(),
                                           vertical_correction_m=height.offset_m)
                                    raise RuntimeError(f"Rear slot {slot} coordinated push exceeded wrist load")
                if target_name == "wall_back":
                    confirm_rear_bearing(arm)
                elif target_name is not None:
                    probe = dict(B2_CONTROLLER_PARAMETERS["neighbor_seating_probe" if target_name.startswith("Cube_")
                                                            else "wall_seating_probe"])
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
                if lateral in allowed_contacts:
                    outward[lateral] = np.array([-sign, 0., 0.])
                dynamic_withdraw(outward, f"rear_{slot}_final_withdraw", require_support=True)
                push_faces.clear()
                allowed_contacts.clear()
                set_contact_drive_mode(False)
                record(f"rear_{slot}_repark", "RRTCONNECT_FROM_SEPARATION", arms=list(robots))
                move({arm: np.asarray(CONFIG["fr3"]["official_start_q_rad"]) for arm in robots})

            for arm in (rear_arm,) if slot == 5 else (rear_arm, lateral):
                base = rail_states[arm].copy()
                # Use current task geometry for the carriage approach station.
                base[1] = rear_y - .351
                if not np.isclose(rail_states[arm][1], base[1]):
                    move_robot_base(arm, base)
            if slot == 5:
                center, rotation = observe_resting_cube("rear_center_entry", vision_camera, rows, SUPPORT_TOP)
                extent = half * np.abs(rotation[0]).sum()
                if min(center[0] - extent - edges[0], edges[1] - center[0] - extent) < .001:
                    raise RuntimeError("Rear center insertion has insufficient visually verified slot clearance")
            contact_plan_rotation = control_rotation.copy()
            approach(rear_arm, 1, 1)
            if slot == 4:
                advance(rear_arm, 1, 1, target[1] - .106, None)
            else:
                advance(rear_arm, 1, 1, rear_y, "wall_back")
            if side_name:
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
            return {"cube_id": current_task["cube_id"], "slot": slot,
                    "target_m": target.tolist(), "observed_position_m": final.tolist(), "support": stable}

        if START_REAR < 6:
            from front_row_a import floor_guide
            from observed_transport import wall_plane

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
                landing = np.asarray(GUIDE["landing_m"], dtype=float)
                wall_y = wall_plane(CONFIG, "wall_back", 1, 1)
                landing[1] += wall_y - (CONFIG["goal_cube_centers_m"][0][1] + half)
                landing[2] = SUPPORT_TOP + half
                if rear_slot == 5:
                    tracked = {name: (p, rot) for name, p, rot in neighbors}
                    a, ra = tracked["Cube_3"]
                    b, rb = tracked["Cube_4"]
                    landing[0] = (a[0] + b[0] + half * (np.abs(ra[0]).sum() - np.abs(rb[0]).sum())) / 2
                current_task["live_landing_m"] = landing
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
