"""Inactive U4-only candidate. Apply to a generated upper-row runner after the outer slots pass."""
from inspect import getsource
from pathlib import Path
from textwrap import indent

import numpy as np

from source_hooks import replace_once


GUIDE = Path(__file__).with_name("upper_row_runtime_slot4_grip40_push29_candidate.json")


def left_loaded_face_x(tcp, tool_rotation, center, cube_rotation, half, radius, length, span):
    """Locate the cube's -X face from the forward-most in-bounds cylindrical lip."""
    outward = -cube_rotation[:, 0]
    if abs(outward[0]) < .95:
        raise ValueError("U4 side contact normal is outside the checked left-face geometry")
    axial = tool_rotation[:, 1]
    alignment = float(outward @ axial)
    radial = outward - alignment * axial
    radial_norm = np.linalg.norm(radial)
    extremum = .5 * length * np.sign(alignment) * axial
    if radial_norm > 1e-8:
        extremum += radius * radial / radial_norm
    gaps = []
    for z in (-span, span):
        for x in (-span, span):
            # On the left tool the cylinder center is +length/2 from the TCP along local Y.
            lip = tcp + tool_rotation @ np.array([x, .5 * length, z]) - extremum
            local = cube_rotation.T @ (lip - center)
            if np.max(np.abs(local[1:])) <= half:
                gaps.append(float(outward @ (lip - center) - half))
    if not gaps:
        raise ValueError("U4 contact has no side suction lip inside the estimated cube face")
    return float(center[0] + min(gaps) / outward[0])


def build(source):
    source = replace_once(source, "        def sensed_cube():",
                          indent(getsource(left_loaded_face_x), "        ") + "\n        def sensed_cube():")
    source = replace_once(source, """                        tcp, _ = tcp_fk(side, measured[side])
                        measured_axis = (tcp - control_rotation @ local_offset)[axis]
                        control_center[axis] = measured_axis""",
                          """                        tcp, tool_rotation = tcp_fk(side, measured[side])
                        measured_axis = (tcp - control_rotation @ local_offset)[axis]
                        if current_task.get("slot") == 4 and side == "left" and axis == 0:
                            if (last_force_sample["physics_step"] != tick or supported_pose_time is None
                                    or force @ (-control_rotation[:, 0]) <= .1):
                                continue
                            tool = CONFIG["side_suction_tool"]
                            measured_axis = left_loaded_face_x(tcp, tool_rotation, control_center, control_rotation,
                                                               half, tool["cup_radius_m"], tool["cup_length_m"],
                                                               ATTACHMENT_PARAMETERS["cup_center_half_span_m"])
                        control_center[axis] = measured_axis""")
    source = replace_once(source, """                        if latched_contact_axis is not None:
                            contact_axes.add(latched_contact_axis)""",
                          """                        if latched_contact_axis is not None and not (
                                current_task.get("slot") == 4 and latched_contact_axis == 0):
                            contact_axes.add(latched_contact_axis)""")
    source = replace_once(source, '''                    for side in robots:
                        if not grippers.close(side):
''', '''                    for side in (("right", "left") if current_task.get("slot") == 4 else robots):
                        if current_task.get("slot") == 4:
                            try:
                                close_upper_side_seal(side, "A_stationary_seal")
                                if side == "left" and not grippers.is_closed("right"):
                                    raise RuntimeError("Upper 4 right seal broke during left docking")
                            except Exception:
                                for connected in robots:
                                    grippers.open(connected)
                                raise
                        elif not grippers.close(side):
''')
    source = replace_once(source, '-plan["staging"]["grip_front_offset_mm"] / 1000 if side_first else 0.',
                          '-b.get("side_push_front_offset_mm", plan["staging"]["grip_front_offset_mm"]) / 1000 if side_first else 0.')
    source = replace_once(source, '            def supported_travel(start, axis, sign, distance):',
                          '''            # At U4 the side stroke starts before rear-wall contact. A moving front face needs a following wrist.
            def slot4_side_rotation(base_rotation):
                nonlocal yaw_correction
                _, rotation = sensed_cube()
                relative = contact_plan_rotation.T @ rotation
                yaw = math.atan2(relative[1, 0], relative[0, 0])
                if abs(yaw) > math.radians(3):
                    raise RuntimeError(f"Upper 4 yaw exceeded 3 degrees ({math.degrees(yaw):.2f})")
                desired = float(np.clip(-yaw, -math.radians(1), math.radians(1)))
                yaw_correction += float(np.clip(desired - yaw_correction, -math.radians(1)/120, math.radians(1)/120))
                return Rotation.from_rotvec([0., 0., yaw_correction]).as_matrix() @ base_rotation

            yaw_correction = 0.
            def supported_travel(start, axis, sign, distance):''')
    source = replace_once(source, "            # At U4 the side stroke",
                          """                if slot == 4 and side_first:
                    left_hold.gain_m_per_Ns = .0018
                    left_hold.max_offset_m = .008
                    left_hold.max_speed_m_s = .0012

            # At U4 the side stroke""")
    source = replace_once(source,
                          '                arms = [side] if follow is None else [side, follow]',
                          '''                steer_front = slot == 4 and axis == 0 and follow is not None
                if steer_front:
                    _, steer_rotation = tcp_fk(side, applied_targets[side])
                    front_wait = 0

                def support_ready():
                    nonlocal front_wait
                    if force(follow, 1, 1) >= .15:
                        front_wait = 0
                        return True
                    front_wait += 1
                    if front_wait > 360:
                        raise RuntimeError("Upper 4 front wrist could not reestablish contact within 3 s")
                    return False

                slot4_boundary = slot == 4 and target in (side_target, rear_target)
                sliding_load = deque(maxlen=120)
                arms = [side] if follow is None else [side, follow]

                def check_slot4_command(commands):
                    velocity = float(np.max(np.abs(commands[side] - applied_targets[side]))) * 120
                    if not np.isfinite(velocity) or velocity > .4:
                        raise RuntimeError(f"{phase}: U4 command joint velocity {velocity:.6f} rad/s exceeds 0.400 rad/s")
                    center, orientation = sensed_cube()
                    issues = checker.check(commands, cube_center=center, cube_rotation=orientation,
                                           cup_contacts=allowed_contacts, margin_m=0., first_only=True)
                    if issues:
                        raise RuntimeError(f"{phase}: U4 command in collision: {issues[0]}")''')
    source = replace_once(source, 'y_offset = push_y.update(-y_error) if force_paced_lateral else 0.',
                          'y_offset = push_y.update(-y_error) if force_paced_lateral or (axis == 0 and slot == 4) else 0.')
    source = replace_once(source, '''                        actual, _ = sensed_cube()
                        if np.linalg.norm(actual - centers[index]) > .012:''',
                          '''                        actual, actual_rotation = sensed_cube()
                        if target in (rear_target, side_target) and .0012 < gap(actual, actual_rotation, axis, sign) < .015:
                            sliding_load.append(force(side, axis, sign))
                        if np.linalg.norm(actual - centers[index]) > .012:''')
    source = replace_once(source, '''                        if max(abs(offset), abs(y_offset)) <= 1e-6:
                            return commands
                        position, rotation = tcp_fk(side, commands[side])
                        position[1] += y_offset
                        position[2] += offset
                        commands[side] = cartesian_joint_target(side, commands[side], position, rotation)
                        center, cube_rotation = sensed_cube()''',
                          '''                        if max(abs(offset), abs(y_offset)) <= 1e-6 and not steer_front:
                            return commands
                        position, rotation = tcp_fk(side, commands[side])
                        position[1] += y_offset
                        position[2] += offset
                        if slot4_boundary:
                            if steer_front:
                                rotation = slot4_side_rotation(steer_rotation)
                            commands[side] = cartesian_joint_target(side, applied_targets[side], position, rotation)
                            check_slot4_command(commands)
                            return commands
                        commands[side] = cartesian_joint_target(side, commands[side], position, rotation)
                        center, cube_rotation = sensed_cube()''')
    source = replace_once(source, '''                    latched_contact_axis = axis
                    try:
                        execute_joint_path(samples, phase + "_travel"''', '''                    def follow_supported_samples(count):
                        previous = 0
                        for index in force_paced_samples(count):
                            while not support_ready():
                                yield previous
                            previous = index
                            yield index

                    latched_contact_axis = axis
                    try:
                        execute_joint_path(samples, phase + "_travel"''')
    source = replace_once(source, '''sample_order=force_paced_samples if force_paced_lateral else None,
                                           adjust_targets=relieve_downward_load if force_paced_lateral else None)''',
                          '''sample_order=follow_supported_samples if steer_front else
                                                        force_paced_samples if force_paced_lateral or slot4_boundary else None,
                                           adjust_targets=relieve_downward_load if force_paced_lateral or slot4_boundary else None)''')
    source = replace_once(source, '''                outer_wall_probe = target in (rear_target, side_target)
                rear_probe = slot in (1, 2, 3) and axis == 1 and target == rear_target
                sliding_load = deque(maxlen=120)
''', '''                witnessed_boundary = target in (rear_target, side_target)
                rear_probe = slot in (1, 2, 3) and axis == 1 and target == rear_target
''')
    source = replace_once(source, '                    if outer_wall_probe:', '                    if witnessed_boundary:')
    source = replace_once(source, '''                        if near > .0007:
                            sliding_load.append(measured_force)''',
                          '''                        if near > (.0012 if slot4_boundary else .0007):
                            sliding_load.append(measured_force)''')
    source = replace_once(source, '''                        contact_threshold = min(7.6, max(3., resistance + 1.0))
                        near_limit = .00045''',
                          '''                        contact_threshold = min(7.6, max(3., resistance + (.6 if slot4_boundary else 1.0)))
                        near_limit = .00075 if slot4_boundary else .00045''')
    source = replace_once(source, '''                    if measured_force >= contact_threshold and -.003 <= near <= near_limit:''',
                          '''                    front_engaged = not steer_front or support_ready()
                    negative_gap_limit = -.002 if slot4_boundary else -.003
                    if front_engaged and measured_force >= contact_threshold and negative_gap_limit <= near <= near_limit:''')
    source = replace_once(source, '''                        position[axis] += sign * increment
                    if len(forces) == settle_frame_count:''',
                          '''                        if front_engaged:
                            position[axis] += sign * increment
                    if len(forces) == settle_frame_count:''')
    source = replace_once(source, 'if outer_wall_probe and sum(wall_impulses) < 24:',
                          'if witnessed_boundary and sum(wall_impulses) < 24:')
    source = replace_once(source, 'wrist contact candidate lacks wall contact in the independent ',
                          'wrist contact candidate lacks target contact in the independent ')
    source = replace_once(source, '''                                   normal_force_samples_N=forces, estimated_gap_m=float(near))''',
                          '''                                   normal_force_samples_N=forces, estimated_gap_m=float(near),
                                   **({"witnessed_target_contact_ticks": sum(wall_impulses),
                                       "sliding_baseline_N": resistance,
                                       "force_threshold_N": contact_threshold} if witnessed_boundary else {}))''')
    return replace_once(source,
                        '''                    pusher = cartesian_joint_target(side, pusher, position, tool_rotation, retarget_guide=True)
                    command_positions(side, pusher)
                    if follow is not None:
                        force(follow, 1, 1)
                        following = follow_origin.copy()
                        following[axis] += position[axis] - origin[axis]
                        follower = cartesian_joint_target(follow, follower, following, follow_rot,
                                                         retarget_guide=True)
                        command_positions(follow, follower)''',
                        '''                    target_rotation = slot4_side_rotation(steer_rotation) if steer_front else tool_rotation
                    pusher = cartesian_joint_target(side, pusher, position, target_rotation,
                                                    retarget_guide=not slot4_boundary)
                    if follow is not None:
                        force(follow, 1, 1)
                        following = follow_origin.copy()
                        following[axis] += position[axis] - origin[axis]
                        follower = cartesian_joint_target(follow, follower, following, follow_rot,
                                                         retarget_guide=True)
                    if slot4_boundary:
                        commands = {**applied_targets, side: pusher}
                        if follow is not None:
                            commands[follow] = follower
                        check_slot4_command(commands)
                    command_positions(side, pusher)
                    if follow is not None:
                        command_positions(follow, follower)''')
