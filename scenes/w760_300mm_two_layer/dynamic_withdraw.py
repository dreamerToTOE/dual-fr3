"""Finish a checked retreat on observed cup clearance instead of a stored endpoint."""

from source_hooks import replace_once


WITHDRAW = '''
        def dynamic_withdraw(directions, phase, *, require_support=False, limit_m=.012):
            from fr3_collision import obb_signed_clearance

            record(phase, "RUNNING", arms=list(directions), travel_limit_m=limit_m)
            start = joint_state()
            origins = {arm: tcp_fk(arm, start[arm]) for arm in directions}
            outward = {arm: vector / np.linalg.norm(vector) for arm, vector in directions.items()}
            seeds = start.copy()
            height = np.full(3, half)
            travel = 0.
            quiet = 0
            checked_arms = list(robots if require_support else directions)
            # The legacy wrist and geometry separation inputs remain independent of truth.
            while travel <= limit_m + 1e-9:
                center, rotation = sensed_cube()
                measured = joint_state()
                gaps = {}
                for arm in checked_arms:
                    _, tool_parts = checker.geometry(arm, measured[arm])
                    gaps[arm] = min(obb_signed_clearance(p, h, r, center, height, rotation)
                                    for name, p, h, r in tool_parts if name.startswith("cup_"))
                forces = {arm: float(np.linalg.norm(last_force_sample[arm]["tcp_gravity_corrected_world"][:3]))
                          for arm in checked_arms}
                supported = abs(center[2] - (SUPPORT_TOP + half)) <= .004
                if all(gaps[a] >= .003 and forces[a] < .30 for a in checked_arms) and (
                        not require_support or supported):
                    quiet += 1
                    if quiet >= 6:
                        actual_travel = {arm: float((tcp_fk(arm, measured[arm])[0] - origins[arm][0])
                                                    @ outward[arm]) for arm in directions}
                        record(phase, "S0_CONFIRMED" if require_support else "TOOL_CLEAR",
                               travel_m=travel, cup_gap_m=gaps, wrist_N=forces,
                               measured_travel_m=actual_travel, box_estimate_m=center.tolist(),
                               measured_joints_rad={arm: q.tolist() for arm, q in measured.items()})
                        return
                    step()
                    check_current_collision(phase)
                    continue
                quiet = 0
                if travel >= limit_m - 1e-9:
                    break
                travel = min(limit_m, travel + .003 / 120)
                for arm, (tip, attitude) in origins.items():
                    seeds[arm] = cartesian_joint_target(arm, seeds[arm], tip + travel * outward[arm], attitude)
                proposed = {**measured, **{arm: seeds[arm] for arm in directions}}
                issues = checker.check(proposed, cube_center=center, cube_rotation=rotation,
                    cup_contacts=allowed_contacts, margin_m=0., first_only=True)
                if issues:
                    record(phase, "SAFETY_WINDOW_BLOCKED", travel_m=travel, collision=issues[0])
                    raise RuntimeError(f"{phase}: observed withdrawal range blocked by {issues[0]}")
                for arm in directions:
                    if np.max(np.abs(seeds[arm] - applied_targets[arm])) * 120 > .4:
                        raise RuntimeError(f"{phase}: dynamic withdrawal reached joint speed limit")
                    command_positions(arm, seeds[arm])
                step()
                check_current_collision(phase)
            record(phase, "WINDOW_WITHOUT_CONFIRMATION", travel_m=travel, cup_gap_m=gaps, wrist_N=forces)
            raise RuntimeError(f"{phase}: no observed separation in the collision-checked retreat")

'''


def install(source):
    source = replace_once(source, '        def touch_face(side, inward, distance, phase):',
                          WITHDRAW + '        def touch_face(side, inward, distance, phase):')
    source = replace_once(source, '''            park = next(p["q_rad"] for p in GUIDE["unloaded_transits"]
                        if p["phase"] == "return_home_after_withdrawal")
            record("A_empty_repark", "RUNNING")
            execute_joint_path({side: np.vstack((q, park[side])) for side, q in joint_state().items()},
                               "A_empty_repark")''', '''            record("A_empty_repark", "RUNNING")
            move({arm: np.asarray(CONFIG["fr3"]["official_start_q_rad"]) for arm in robots})''')
    source = replace_once(source, '''            for phase in GUIDE["phases"][FIRST_LOADED:]:
                if not phase["carrying"]:
                    follow_side_guide(phase)
            allowed_contacts.clear()
            park_after_side_transfer()''', '''            if allowed_contacts:
                center, attitude = sensed_cube()
                directions = {arm: attitude @ (np.asarray(face) / half)
                              for arm, face in GUIDE["grasp_faces_world"].items()}
                dynamic_withdraw(directions, "A_dynamic_withdraw", require_support=True)
            allowed_contacts.clear()
            park_after_side_transfer()''')
    start = source.index('                position, _ = tcp_fk(side, applied_targets[side])\n',
                         source.index('            def withdraw(side, axis, sign):'))
    end = source.index('\n            allowed_contacts.clear()', start)
    source = source[:start] + '''                outward = np.eye(3)[axis] * -sign
                dynamic_withdraw({side: outward}, phase + "_dynamic_withdraw",
                                 require_support=side == support_side)
                push_faces.pop(side, None)
                allowed_contacts.discard(side)
''' + source[end:]
    source = replace_once(source, '''                withdraw(push_side, 0, push_sign)
                record("upper_B_park", "RUNNING", arm=push_side)
                move({push_side: home})
            if side_first:''', '''                if side_first:
                    withdraw(push_side, 0, push_sign)
                    record("upper_B_park", "RUNNING", arm=push_side)
                    move({push_side: home})
            if side_first:''')
    source = replace_once(source, '''            withdraw(support_side, 1, 1)
            allowed_contacts.clear()
            push_faces.clear()
            set_contact_drive_mode(False)
            record("upper_B_park", "RUNNING", arm=support_side)
            move({support_side: home})''', '''            if push_side in allowed_contacts:
                left_hold = None
                dynamic_withdraw({support_side: np.array([0., -1., 0.]),
                                  push_side: np.array([-push_sign, 0., 0.])},
                                 "upper_B_dual_dynamic_withdraw", require_support=True)
                parked_arms = list(robots)
            else:
                withdraw(support_side, 1, 1)
                parked_arms = [support_side]
            allowed_contacts.clear()
            push_faces.clear()
            set_contact_drive_mode(False)
            record("upper_B_park", "RUNNING", arms=parked_arms)
            move({arm: home for arm in parked_arms})''')
    source = replace_once(source, '''                    unload_normal(push_side, 0, push_sign, f"{prefix}_{push_side}_unload")
                    push_faces.clear()
                    tip, _ = tcp_fk(push_side, applied_targets[push_side])
                    cartesian_move(push_side, tip - [push_sign * .014, 0., 0.], 2., f"{prefix}_{push_side}_retreat")''',
                          '''                    dynamic_withdraw({push_side: np.array([-push_sign, 0., 0.])},
                                     f"{prefix}_{push_side}_dynamic_withdraw")
                    push_faces.clear()''')
    source = replace_once(source, '''                    unload_normal("right", 0, -1, "front_center_align_unload")
                    push_faces.clear()
                    tip, _ = tcp_fk("right", applied_targets["right"])
                    cartesian_move("right", tip + [.014, 0., 0.], 2., "front_center_align_retreat")''',
                          '''                    dynamic_withdraw({"right": np.array([1., 0., 0.])},
                                     "front_center_align_dynamic_withdraw")
                    push_faces.clear()''')
    source = replace_once(source, '''                unload_normal(support_side, 1, 1, f"{prefix}_{support_side}_unload")
                push_faces.clear()
                tip, _ = tcp_fk(support_side, applied_targets[support_side])
                cartesian_move(support_side, tip - [0., .035, 0.], 3., f"{prefix}_{support_side}_retreat")''',
                          '''                dynamic_withdraw({support_side: np.array([0., -1., 0.])},
                                 f"{prefix}_{support_side}_dynamic_withdraw", require_support=True)
                push_faces.clear()''')
    source = replace_once(source, '''                    unload_normal(support_side, 1, 1, "B2_support_unload")
                    left_hold = None
                    push_faces.clear()
                    position, _ = tcp_fk(support_side, applied_targets[support_side])
                    cartesian_move(support_side, position + [0, -.035, 0],
                                   B2_CONTROLLER_PARAMETERS["retreat_duration_s"], "B2_support_retreat")''',
                          '''                    left_hold = None
                    dynamic_withdraw({support_side: np.array([0., -1., 0.])},
                                     "B2_center_dynamic_withdraw", require_support=True)
                    push_faces.clear()''')
    start = source.index('                push_faces.pop(push_side, None)\n'
                         '                position, _ = tcp_fk(push_side, applied_targets[push_side])',
                         source.index('        def run_front_b('))
    end = source.index('                allowed_contacts.discard(push_side)', start)
    source = source[:start] + '''                dynamic_withdraw({push_side: np.array([-push_sign, 0., 0.])},
                                 "front_B2_lateral_dynamic_withdraw")
                push_faces.pop(push_side, None)
''' + source[end:]
    source = replace_once(source, '''                    unload_normal(support_side, 1, 1, "B2_stage_support_unload")
                    left_hold = None
                    push_faces.pop(support_side, None)
                    position, _ = tcp_fk(support_side, applied_targets[support_side])
                    cartesian_move(support_side, position - [0., .065, 0.], 4., "B2_stage_rgb_clearance")''',
                          '''                    left_hold = None
                    dynamic_withdraw({support_side: np.array([0., -1., 0.])},
                                     "B2_stage_support_dynamic_withdraw", require_support=True)
                    push_faces.pop(support_side, None)
                    allowed_contacts.discard(support_side)''')
    return replace_once(source, '''            unload_normal(support_side, 1, 1, "B2_support_unload")
            left_hold = None
            push_faces.clear()
            position, _ = tcp_fk(support_side, applied_targets[support_side])
            cartesian_move(support_side, position + [0, -.035, 0],
                           B2_CONTROLLER_PARAMETERS["retreat_duration_s"], "B2_support_retreat")''',
                          '''            left_hold = None
            dynamic_withdraw({support_side: np.array([0., -1., 0.])},
                             "B2_support_dynamic_withdraw", require_support=True)
            push_faces.clear()''')
