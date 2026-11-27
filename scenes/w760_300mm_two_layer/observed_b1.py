"""Solve approach targets against live perception, then plan from actual joints."""

from source_hooks import replace_once


def install(source):
    source = replace_once(source, '        def move(targets, duration=1.0):',
                          '        def move(targets, duration=1.0, *, planning_margin_m=.002):')
    source = replace_once(source, '''                return checker.is_clear(states, cube_center=center, cube_rotation=rotation,
                                        cup_contacts=allowed_contacts)''',
                          '''                return checker.is_clear(states, cube_center=center, cube_rotation=rotation,
                                        cup_contacts=allowed_contacts, margin_m=planning_margin_m)''')
    source = replace_once(source,
                          '        def cartesian_joint_target(side, seed, target, target_rotation, retarget_guide=False):',
                          '''        def cartesian_joint_target(side, seed, target, target_rotation, retarget_guide=False,
                                   *, fit_tolerance=None):''')
    source = replace_once(source, '''            if (np.linalg.norm(error[:3]) < position_tolerance
                    and np.linalg.norm(error[3:]) / .2 < angle_tolerance):
                return fit.x''', '''            position_limit, angle_limit = fit_tolerance or (position_tolerance, angle_tolerance)
            if (np.linalg.norm(error[:3]) < position_limit
                    and np.linalg.norm(error[3:]) / .2 < angle_limit):
                return fit.x''')
    anchor = '        def solve(side, tcp, rotation, continuous=False, seed_override=None):'
    source = replace_once(source, anchor, '''        def move_to_observed_precontact(arm, destination, attitude, phase, ik_hint=None,
                                        *, prefer_hint=False, planning_margin_m=.002, approach_lift_m=0.):
            measured = joint_state()
            tool, tool_rotation = tcp_fk(arm, measured[arm])
            center, rotation = sensed_cube()
            record(phase, "RRTCONNECT_FROM_MEASUREMENT", arm=arm,
                   measured_tcp_m=tool.tolist(), measured_tcp_rotation=tool_rotation.tolist(),
                   measured_joints_rad=measured[arm].tolist(), measured_rail_y_m=float(rail_states[arm][1]),
                   observed_cube_m=center.tolist(), precontact_tcp_m=destination.tolist(),
                   planning_margin_m=planning_margin_m, approach_lift_m=approach_lift_m)
            guesses = [("measured_joints", measured[arm])]
            if ik_hint is not None:
                guesses.insert(0 if prefer_hint else 1, ("posture_hint", np.asarray(ik_hint, dtype=float)))
            failures = []
            for seed_source, guess in guesses:
                try:
                    # Keep the fit tighter than the measured arrival gate (1 mm, about 14 mrad).
                    q = cartesian_joint_target(arm, guess, destination, attitude, retarget_guide=True,
                                               fit_tolerance=(.0005, .007))
                except RuntimeError as error:
                    failures.append(str(error))
                    continue
                trial = {**measured, arm: q}
                issues = checker.check(trial, cube_center=center, cube_rotation=rotation,
                                       cup_contacts=allowed_contacts, margin_m=planning_margin_m, first_only=True)
                if not issues:
                    try:
                        if approach_lift_m:
                            raised = [q]
                            for lift in np.linspace(0., approach_lift_m, 41)[1:]:
                                above = destination + np.array([0., 0., lift])
                                raised.append(cartesian_joint_target(arm, raised[-1], above, attitude,
                                                                     retarget_guide=True))
                            move({arm: raised[-1]}, planning_margin_m=planning_margin_m)
                            execute_joint_path({arm: np.asarray(raised[::-1])}, phase + "_descent")
                        else:
                            move({arm: q}, planning_margin_m=planning_margin_m)
                    except RuntimeError as error:
                        if not str(error).startswith(("No collision-free joint path found",
                                                      "Motion endpoint is in collision")):
                            raise
                        failures.append(str(error))
                    else:
                        # A timed path ending does not imply the measured TCP has arrived.
                        for _ in range(240):
                            position, actual_attitude = tcp_fk(arm, joint_state()[arm])
                            error = float(np.linalg.norm(position - destination))
                            frame_error = float(np.linalg.norm(actual_attitude - attitude))
                            if error <= .001 and frame_error <= .02:
                                record(phase, "PRECONTACT_REACHED", arm=arm, measured_tcp_m=position.tolist(),
                                       position_error_m=error, orientation_matrix_error=frame_error,
                                       ik_seed_source=seed_source)
                                return
                            step()
                            check_current_collision(phase)
                        raise RuntimeError(f"{phase}: measured precontact error is {error*1000:.2f} mm "
                                           f"and frame error {frame_error:.3f}")
                else:
                    failures.append(str(issues[0]))
            raise RuntimeError(f"{phase}: observed precontact is unreachable: {failures}")

''' + anchor)
    start = source.index('                    b1_seed = None\n', source.index('        def run_front_b('))
    end = source.index('                    allowed_contacts.add(support_side)', start)
    source = source[:start] + '''                    hint = (outside_plan["b1_q"] if outside_plan is not None
                            else third_box_seed if slot == 3 else None)
                    move_to_observed_precontact(support_side, contact, front_tool_rotation,
                                                "front_B1_observed_approach", hint, prefer_hint=slot in (1, 2))
''' + source[end:]
    for arm, point, attitude in (("push_side", "high", "side_tool"),
                                 ("push_side", "target", "side_tool"),
                                 ("support_side", "high", "front_rotation"),
                                 ("support_side", "target", "front_rotation")):
        for seed in ("seed", "high_seed"):
            old = f'move({{{arm}: cartesian_joint_target({arm}, {seed}, {point}, {attitude}, retarget_guide=True)}})'
            # Slot 3 and 4 can share the same expression but use different IK hints.
            new = (f'move_to_observed_precontact({arm}, {point}, {attitude}, '
                   f'f"{{prefix}}_{{{arm}}}_{"approach_high" if point == "high" else "approach"}", {seed})')
            source = source.replace(old, new)
    source = replace_once(source, '''                q = cartesian_joint_target(support_side, seed, destination, rotation @ nominal_rotation,
                                           retarget_guide=True)
                record("upper_B_front_approach", "RUNNING", arm=support_side)
                move({support_side: q})''', '''                move_to_observed_precontact(support_side, destination, rotation @ nominal_rotation,
                                            "upper_B_front_approach", seed)''')
    # Both staging heights are relative to the currently observed box and wall tops.
    source = replace_once(source, '''                high_point = np.asarray(reference["tcp_m"][push_side][0], dtype=float)
                stand = abs(float(reference["tcp_m"][push_side][-1][0]
                                  - reference["cube_centers_m"][-1][0])) - half''', '''                reference_tips = np.asarray(reference["tcp_m"][push_side], dtype=float)
                reference_centers = np.asarray(reference["cube_centers_m"], dtype=float)
                stand = abs(float(reference_tips[-1, 0] - reference_centers[-1, 0])) - half
                high_z = center[2] + reference_tips[0, 2] - reference_centers[0, 2]''')
    source = replace_once(source, '''                high_point[:2] = destination[:2]
                q = cartesian_joint_target(push_side, high_seed, high_point, side_rotation, retarget_guide=True)
                record("upper_B_side_high", "RUNNING", arm=push_side)
                move({push_side: q})''', '''                high_point = destination.copy()
                high_point[2] = high_z
                move_to_observed_precontact(push_side, high_point, side_rotation, "upper_B_side_high",
                                            high_seed, prefer_hint=True)''')
    source = replace_once(source, '''                    reapproach_q = applied_targets[support_side].copy()
                    record("B2_stage_camera_clear", "RUNNING", arm=support_side)
                    move({support_side: np.asarray(CONFIG["fr3"]["official_start_q_rad"])})''',
                          '''                    # The just-cleared wrist provides a local IK seed for this face.
                    reapproach_seed = (side_resume or {}).get("cleared_wrist_q_rad", joint_state()[support_side])
                    if not side_resume:
                        center, attitude = sensed_cube()
                        save_front_checkpoint(slot, upper_pose_records({str(current_task["cube_id"]): cube})[0],
                            side_seat={"cube_id": current_task["cube_id"], "source_step": tick,
                                       "cube_start_rgb": cube_start.tolist(), "lateral_progress_m": progress,
                                       "side_seat": evidence["side_seat"], "estimated_center_m": center.tolist(),
                                       "estimated_rotation_world": attitude.tolist(),
                                       "cleared_wrist_q_rad": np.asarray(reapproach_seed).tolist()})
                    record("B2_stage_camera_clear", "RUNNING", arms=list(robots))
                    move({arm: np.asarray(CONFIG["fr3"]["official_start_q_rad"]) for arm in robots})''')
    source = replace_once(source, '''                if slot in (1, 2, 3):
                    move({support_side: reapproach_q})
                if abs(observed[0] - b2_target[0]) > .004:''',
                          '''                if slot in (1, 2, 3):
                    destination = observed + rotation @ [0., -half - .007, front_height]
                    move_to_observed_precontact(support_side, destination, rotation @ front_rotation,
                                                "B2_staged_observed_reapproach", reapproach_seed)
                if abs(observed[0] - b2_target[0]) > .004:''')
    source = replace_once(source, '''                    cartesian_move(support_side, observed + [0., -half - .007, front_height],
                                   3., "B2_stage_rear_pretouch")
                    allowed_contacts.add(support_side)''',
                          '''                    allowed_contacts.add(support_side)''')
    source = replace_once(source, '        def save_front_checkpoint(next_front, landed=None):',
                          '        def save_front_checkpoint(next_front, landed=None, *, side_seat=None):')
    source = replace_once(source, '''            name = "front_landing_checkpoint.json" if landed else "front_row_checkpoint.json"''',
                          '''            name = "front_landing_checkpoint.json" if landed else "front_row_checkpoint.json"
            if side_seat is not None:
                data["front_side"] = side_seat
                name = "front_side_checkpoint.json"''')
    return source
