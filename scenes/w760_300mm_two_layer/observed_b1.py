"""Solve approach targets against live perception, then plan from actual joints."""

from source_hooks import replace_once


FACE_STANDOFF_M = .007


DYNAMIC_PAIR = '''        def move_independent_pair(targets, phase, *, routes=None, planning_margin_m=.002):
            from fr3_dynamic_pair import arm_route, run_independent_pair

            if left_hold is not None or allowed_contacts or carry_offsets is not None:
                raise RuntimeError(f"{phase}: independent paths require unloaded, separated tools")
            arms = tuple(targets)
            limits = {arm: tuple(np.asarray(robots[arm].dof_properties[key][joints[arm]])
                                 for key in ("lower", "upper")) for arm in arms}
            center, rotation = sensed_cube()
            obstacles = checker.other_cubes
            static_cache = {arm: {} for arm in arms}

            def clear(states):
                options = dict(cube_center=center, cube_rotation=rotation, margin_m=planning_margin_m)
                geometry = {}
                bounds = []
                for arm, q in states.items():
                    key = q.tobytes()
                    if key not in static_cache[arm]:
                        body = checker.geometry(arm, q)
                        safe = checker.is_clear({arm: q}, geometry={arm: body}, **options)
                        static_cache[arm][key] = safe, body, checker.arm_bounds(body)
                    safe, geometry[arm], extent = static_cache[arm][key]
                    if not safe:
                        return False
                    bounds.append(extent)
                if len(bounds) < 2:
                    return True
                (a_min, a_max), (b_min, b_max) = bounds
                if np.any(np.maximum(a_min - b_max, b_min - a_max) >= planning_margin_m):
                    return True
                return not checker.check_interarm(geometry, margin_m=planning_margin_m, first_only=True)

            def observe():
                nonlocal center, rotation, obstacles
                new_center, new_rotation = sensed_cube()
                if (obstacles is not checker.other_cubes or not np.array_equal(center, new_center)
                        or not np.array_equal(rotation, new_rotation)):
                    for cache in static_cache.values():
                        cache.clear()
                center, rotation = new_center, new_rotation
                obstacles = checker.other_cubes
                for cache in static_cache.values():
                    if len(cache) > 64:
                        cache.clear()
                return joint_state(), {arm: np.asarray(robots[arm].get_joint_velocities(joint_indices=joints[arm]))
                                       for arm in arms}

            def plan_one(arm, start, goal, held):
                def arm_clear(q):
                    planning_update(f"{phase}: {arm} RRTConnect")
                    return clear({**held, arm: q})
                return arm_route(start, goal, limits[arm], arm_clear)

            def plan_remainder(actual, goals):
                failures = []
                for first, second in (arms, arms[::-1]):
                    try:
                        first_route = plan_one(first, actual[first], goals[first], {second: actual[second]})
                        second_route = plan_one(second, actual[second], goals[second], {first: goals[first]})
                    except RuntimeError as error:
                        failures.append(str(error))
                        continue
                    # Preserve the verified passing order only for the conflicting remainder.
                    return {first: first_route,
                            second: np.vstack((np.tile(actual[second], (len(first_route) - 1, 1)), second_route))}
                raise RuntimeError(f"{phase}: no remaining independent arm paths: {failures}")

            def advance():
                step()
                check_current_collision(phase)

            record(phase, "DYNAMIC_PAIR_RUNNING", arms=list(arms))
            if routes is None:
                actual, _ = observe()
                routes = {arm: plan_one(arm, actual[arm], targets[arm], {}) for arm in arms}
            for arm, path in routes.items():
                if not np.all(np.isfinite(path) & (path >= limits[arm][0]) & (path <= limits[arm][1])):
                    raise RuntimeError(f"{phase}: {arm} path exceeds joint limits")
            record(phase, "INDEPENDENT_PATHS_READY", samples={arm: len(path) for arm, path in routes.items()})
            result = run_independent_pair(routes, observe, clear, command_positions, advance, plan_remainder,
                                          lambda status, **details: record(phase, status, **details))
            record(phase, "DYNAMIC_PAIR_COMPLETE", **result)

'''


def install(source):
    source = replace_once(source, '        def move(targets, duration=1.0):',
                          '        def move(targets, duration=1.0, *, planning_margin_m=.002, continuation=()):')
    source = replace_once(source, '            cached = take_cached_phase(phase, current)\n',
                          '''            if len(sides) == 2:
                move_independent_pair(targets, phase, planning_margin_m=planning_margin_m)
                return
            cached = None if continuation else take_cached_phase(phase, current)
''')
    source = replace_once(source, '        def move(targets, duration=1.0, *, planning_margin_m=.002, continuation=()):',
                          DYNAMIC_PAIR + '        def move(targets, duration=1.0, *, planning_margin_m=.002, continuation=()):')
    source = replace_once(source, '            execute_joint_path(samples, phase, sampled_trajectory=(times, samples))\n\n'
                                 '        def seat_against_wall(',
                          '            move_independent_pair({arm: path[-1] for arm, path in samples.items()}, phase, '
                          'routes=samples)\n\n        def seat_against_wall(')
    source = replace_once(source,
                          "                                prefetched_cache=None, sampled_trajectory=None, sample_order=None, adjust_targets=None):",
                          "                                prefetched_cache=None, sampled_trajectory=None, sample_order=None, adjust_targets=None, *, use_cache=True):")
    source = replace_once(source, "            cached_entry = (None if sampled_trajectory is not None else",
                          "            cached_entry = (None if sampled_trajectory is not None or not use_cache else")
    source = replace_once(source,
                          "            execute_joint_path(paths, phase, max_joint_speed=0.4)\n",
                          '''            if len(continuation):
                side, = sides
                paths[side] = np.vstack((paths[side], continuation))
            execute_joint_path(paths, phase, max_joint_speed=0.4, use_cache=not continuation)
''')
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
    source = replace_once(source, '''            approach_targets = {
                side: supply + supply_rotation @ (np.asarray(GUIDE["grasp_tcp_offsets_world"][side]) + .007 * normal)
                for side, normal in face_normals.items()}
            paired_cartesian_move(approach_targets, 2., "A_contact_approach")
            allowed_contacts.update(robots)''',
                          '''            from observed_b1 import FACE_STANDOFF_M

            approach_targets = {side: face_centers[side] + FACE_STANDOFF_M * (supply_rotation @ normal)
                                for side, normal in face_normals.items()}
            paired_cartesian_move(approach_targets, 2., "A_contact_approach")
            for side, target in approach_targets.items():
                position, error, frame_error = wait_for_precontact(
                    side, target, desired_rotation, "A_contact_approach")
                record("A_contact_approach", "PRECONTACT_REACHED", arm=side,
                       target_tcp_m=target.tolist(), target_face_point_m=face_centers[side].tolist(),
                       measured_tcp_m=position.tolist(), position_error_m=error,
                       orientation_matrix_error=frame_error, standoff_m=FACE_STANDOFF_M)
            allowed_contacts.update(robots)''')
    anchor = '        def solve(side, tcp, rotation, continuous=False, seed_override=None):'
    source = replace_once(source, anchor, '''        def wait_for_precontact(arm, destination, attitude, phase):
            # An ended trajectory may still have tracking lag. Verify before allowing face contact.
            for _ in range(240):
                position, actual_attitude = tcp_fk(arm, joint_state()[arm])
                error = float(np.linalg.norm(position - destination))
                frame_error = float(np.linalg.norm(actual_attitude - attitude))
                if error <= .001 and frame_error <= .02:
                    return position, error, frame_error
                step()
                check_current_collision(phase)
            raise RuntimeError(f"{phase}: measured precontact error is {error*1000:.2f} mm "
                               f"and frame error {frame_error:.3f}")

        def move_to_observed_precontact(arm, destination, attitude, phase, ik_hint=None,
                                        *, prefer_hint=False, planning_margin_m=.002, approach_lift_m=0., outward=None):
            measured = joint_state()
            tool, tool_rotation = tcp_fk(arm, measured[arm])
            center, rotation = sensed_cube()
            record(phase, "RRTCONNECT_FROM_MEASUREMENT", arm=arm,
                   measured_tcp_m=tool.tolist(), measured_tcp_rotation=tool_rotation.tolist(),
                   measured_joints_rad=measured[arm].tolist(), measured_rail_y_m=float(rail_states[arm][1]),
                   observed_cube_m=center.tolist(), precontact_tcp_m=destination.tolist(),
                   planning_margin_m=planning_margin_m, approach_lift_m=approach_lift_m)
            guesses = [("measured_joints", measured[arm], None)]
            if outward is not None and not approach_lift_m:
                # Continue near a freshly solved branch while retaining the same final precontact target.
                guesses.append(("online_normal_continuation", measured[arm], .003 * outward))
            if ik_hint is not None:
                guesses.insert(0 if prefer_hint else len(guesses),
                               ("posture_hint", np.asarray(ik_hint, dtype=float), None))
            failures = []
            for seed_source, guess, outer_delta in guesses:
                continuation = []
                outer = destination if outer_delta is None else destination + outer_delta
                try:
                    # Keep the fit tighter than the measured arrival gate (1 mm, about 14 mrad).
                    q = cartesian_joint_target(arm, guess, outer, attitude, retarget_guide=True,
                                               fit_tolerance=(.0005, .007))
                    if outer_delta is not None:
                        next_q = q
                        for fraction in np.linspace(0., 1., 7)[1:]:
                            point = outer - fraction * outer_delta
                            next_q = cartesian_joint_target(arm, next_q, point, attitude)
                            continuation.append(next_q)
                except RuntimeError as error:
                    failures.append(str(error))
                    continue
                if continuation:
                    chain = np.vstack((q, continuation))
                    samples = {side: chain if side == arm else np.tile(state, (len(chain), 1))
                               for side, state in measured.items()}
                    validation = checker.validate_path(
                        samples, cube_centers=center, cube_rotations=rotation, cup_contacts=allowed_contacts,
                        margin_m=planning_margin_m, max_joint_step_rad=.005)
                    record(phase, "K0_IK_CONTINUATION", arm=arm, seed_source="measured_joints",
                           outer_target_tcp_m=outer.tolist(), final_target_tcp_m=destination.tolist(),
                           outer_q_rad=q.tolist(), final_q_rad=continuation[-1].tolist(),
                           continuation_checked_samples=validation["samples_checked"],
                           continuation_issues=validation["issues"], prior_candidate_failures=failures.copy(),
                           planning_margin_m=planning_margin_m,
                           allowed_contacts=sorted(allowed_contacts))
                    if validation["status"] != "CLEAR":
                        failures.extend(str(issue) for issue in validation["issues"])
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
                            move({arm: q}, planning_margin_m=planning_margin_m, continuation=continuation)
                    except RuntimeError as error:
                        if not str(error).startswith(("No collision-free joint path found",
                                                      "Motion endpoint is in collision")):
                            raise
                        failures.append(str(error))
                    else:
                        position, error, frame_error = wait_for_precontact(arm, destination, attitude, phase)
                        record(phase, "PRECONTACT_REACHED", arm=arm, measured_tcp_m=position.tolist(),
                               position_error_m=error, orientation_matrix_error=frame_error,
                               ik_seed_source=seed_source)
                        return
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
    source = replace_once(source, '''            def approach_front():
''', '''            from observed_b1 import FACE_STANDOFF_M

            def approach_front():
''')
    source = replace_once(source, '                destination = center + rotation @ front_offset - .008 * front_normal',
                          '                destination = center + rotation @ front_offset - FACE_STANDOFF_M * front_normal')
    source = replace_once(source, '''                q = cartesian_joint_target(support_side, seed, destination, rotation @ nominal_rotation,
                                           retarget_guide=True)
                record("upper_B_front_approach", "RUNNING", arm=support_side)
                move({support_side: q})''', '''                move_to_observed_precontact(support_side, destination, rotation @ nominal_rotation,
                                            "upper_B_front_approach", seed)''')
    source = replace_once(source, '''                allowed_contacts.add(push_side)
                cartesian_move(push_side, b2_center + outward * (half + .007) + [0, 0, active_height],
                               2.0, "B2_suction_plane_approach")''',
                          '''                final_precontact = b2_center + outward * (half + .007) + [0, 0, active_height]
                cartesian_move(push_side, final_precontact, 2.0, "B2_suction_plane_approach")
                wait_for_precontact(push_side, final_precontact, b2_rotation, "B2_suction_plane_approach")
                allowed_contacts.add(push_side)''')
    source = replace_once(source, '''                        cartesian_move(support_side, target, 5., f"{prefix}_straight_descent")''',
                          '''                        cartesian_move(support_side, target, 5., f"{prefix}_straight_descent")
                        wait_for_precontact(support_side, target, front_rotation, f"{prefix}_straight_descent")''')
    # Keep the high waypoint relative to the fitted box and end the descent at the common standoff.
    source = replace_once(source, '''                high_point = np.asarray(reference["tcp_m"][push_side][0], dtype=float)
                stand = abs(float(reference["tcp_m"][push_side][-1][0]
                                  - reference["cube_centers_m"][-1][0])) - half
                if slot == 3:
                    stand = .005''', '''                reference_tips = np.asarray(reference["tcp_m"][push_side], dtype=float)
                reference_centers = np.asarray(reference["cube_centers_m"], dtype=float)
                stand = FACE_STANDOFF_M
                high_z = center[2] + reference_tips[0, 2] - reference_centers[0, 2]''')
    source = replace_once(source, '''                high_point[:2] = destination[:2]
                q = cartesian_joint_target(push_side, high_seed, high_point, side_rotation, retarget_guide=True)
                record("upper_B_side_high", "RUNNING", arm=push_side)
                move({push_side: q})''', '''                high_point = destination.copy()
                high_point[2] = high_z
                move_to_observed_precontact(push_side, high_point, side_rotation, "upper_B_side_high",
                                            high_seed, prefer_hint=True)''')
    source = replace_once(source, '''                    align_contact_plane(push_side, aligned, pitch_phase)
                    record(pitch_phase, "ALIGNED_BEFORE_CONTACT"''', '''                    align_contact_plane(push_side, aligned, pitch_phase)
                    side_rotation = aligned
                    record(pitch_phase, "ALIGNED_BEFORE_CONTACT"''')
    source = replace_once(source, '''                if stand > .007:
                    cartesian_move(push_side, face + .007 * normal, 2., "upper_B_side_standoff")''',
                          '''                position, error, frame_error = wait_for_precontact(
                    push_side, destination, side_rotation, "upper_B_side_descent")
                record("upper_B_side_descent", "PRECONTACT_REACHED", arm=push_side,
                       precontact_tcp_m=destination.tolist(), measured_tcp_m=position.tolist(),
                       position_error_m=error, orientation_matrix_error=frame_error, standoff_m=stand)''')
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
