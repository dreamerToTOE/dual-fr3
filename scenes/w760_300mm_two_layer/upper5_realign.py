"""Try one physically supported yaw correction after the upper-5 camera gate refuses."""

from textwrap import indent

from source_hooks import replace_once


REALIGN = '''
def realign_upper5(neighbors):
    nonlocal slot4_sealed_pose, contact_plan_rotation
    phase = "upper_B_yaw_recovery"
    arm = support_side
    close_upper_side_seal(arm, phase)
    center, box_rotation = sensed_cube()
    q = joint_state()[arm]
    anchor, tool_rotation = tcp_fk(arm, q)
    offset = tool_rotation.T @ (center - anchor)
    relative = tool_rotation.T @ box_rotation
    yaw = math.atan2(box_rotation[1, 0], box_rotation[0, 0])

    def sealed_pose():
        position, rotation = tcp_fk(arm, joint_state()[arm])
        return position + rotation @ offset, rotation @ relative

    slot4_sealed_pose = sealed_pose
    push_faces.clear()
    contact_monitor.phase = phase
    record(phase, "ROTATING_ON_LOWER_ROW", arm=arm, observed_yaw_deg=math.degrees(yaw))
    try:
        for t in np.linspace(0., 1., 361):
            u = (10. - 15.*t + 6.*t*t)*t**3
            turn = Rotation.from_rotvec([0., 0., -yaw*u]).as_matrix()
            target = center + turn @ (anchor - center)
            command = cartesian_joint_target(arm, q, target, turn @ tool_rotation, retarget_guide=True)
            if np.max(np.abs(command - q))*120 > .4:
                raise RuntimeError(f"{phase}: yaw correction exceeds 0.4 rad/s")
            q = command
            command_positions(arm, q)
            step()
            check_current_collision(phase)
            force(arm, 1, 1)
            if not grippers.is_closed(arm) or support_impulse() <= 1e-5:
                raise RuntimeError(f"{phase}: lost the front seal or lower support")
    finally:
        grippers.open(arm)
        slot4_sealed_pose = None
    contact_plan_rotation = None
    withdraw(arm, 1, 1)
    position, rotation = observe_resting_cube(
        phase + "_pose", vision_camera, vision_estimator, stack_plane)
    clear, entry = fixed_x_entry_clearance(neighbors, position, rotation)
    record("upper_B_straight_entry_gate", "CLEAR" if clear else "BLOCKED",
           after_realign=True, **entry)
    event = "ENTRY_CLEAR" if clear else "ENTRY_BLOCKED_GEOMETRY"
    selected = select_skill(Skill.REALIGN, event)
    record("upper_B_skill_selection", "SELECTED", current_skill=int(Skill.REALIGN),
           next_skill=int(selected), observation=event)
    if selected != Skill.APPROACH:
        raise RuntimeError("Upper slot 5 still has insufficient clearance after yaw correction")
    set_contact_drive_mode(False)
    return approach_front(settle_after_transit=True)
'''


def inject(source):
    source = replace_once(
        source, "use_extra_u2=True):",
        "use_extra_u2=True, realign_neighbors=None):",
    )
    source = replace_once(
        source, "nonlocal latched_contact_axis, contact_plan_rotation, carry_offsets",
        "nonlocal latched_contact_axis, contact_plan_rotation, carry_offsets, slot4_sealed_pose",
    )
    source = replace_once(
        source, '            allowed_contacts.clear()\n            push_faces.clear()\n            left_hold = None\n            home =',
        indent(REALIGN.strip() + "\n\n", "            ") +
        '            allowed_contacts.clear()\n            push_faces.clear()\n            left_hold = None\n            home =',
    )
    source = replace_once(
        source, '            contact_plan_rotation = approach_front()',
        '            contact_plan_rotation = approach_front()\n'
        '            if realign_neighbors is not None:\n'
        '                contact_plan_rotation = realign_upper5(realign_neighbors)',
    )
    source = replace_once(source, '            def approach_front():',
                          '            def approach_front(*, settle_after_transit=False):')
    source = replace_once(
        source, '                move({support_side: q})\n                set_contact_drive_mode(True)',
        '                move({support_side: q})\n'
        '                if settle_after_transit:\n'
        '                    # Let the wrist reach the standoff before contact mode rebases its target.\n'
        '                    step(24)\n'
        '                set_contact_drive_mode(True)',
    )
    return replace_once(
        source, '''                if not clear:
                    raise RuntimeError("Upper slot 5 straight push has insufficient measured lateral clearance "
                                       f"(left {entry['left_gap_m']*1000:.2f} mm, right {entry['right_gap_m']*1000:.2f} mm)")
                checker.set_other_cubes(neighbors, preserve_unobserved=True)
            execute_upper_push(GUIDE, slot, target)''',
        '''                checker.set_other_cubes(neighbors, preserve_unobserved=True)
                from experiments.skill_selection import Skill, entry_event, select_skill
                event = entry_event(entry["left_gap_m"], entry["right_gap_m"],
                                    entry["projected_cube_width_m"], 2 * half, entry["minimum_gap_per_side_m"])
                selected = select_skill(Skill.APPROACH, event)
                record("upper_B_skill_selection", "SELECTED", current_skill=int(Skill.APPROACH),
                       next_skill=int(selected), observation=event)
                if not clear and selected != Skill.REALIGN:
                    raise RuntimeError("Upper slot 5 does not have sufficient side clearance after yaw compensation")
            execute_upper_push(GUIDE, slot, target,
                               realign_neighbors=neighbors if slot == 5 and selected == Skill.REALIGN else None)''',
    )
