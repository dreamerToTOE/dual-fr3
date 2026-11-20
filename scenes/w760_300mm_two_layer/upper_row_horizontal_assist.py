"""Optional supported horizontal assistance for U1/U2. Call inject() on the built runner to opt in."""
from collections import deque

import numpy as np

from source_hooks import replace_once


class HorizontalLoadShare:
    def __init__(self, push_sign, weight_N):
        from fr3_contact_control import NormalForceController

        self.sign = push_sign
        self.weight_N = weight_N
        self.impulses = deque(maxlen=12)
        self.missing = 0
        self.lead = NormalForceController(0., gain_m_per_Ns=.00045, max_offset_m=.0008,
                                          max_speed_m_s=.0012, max_force_N=9.)

    def check(self, main, front, lower_impulse_Ns, sealed):
        wrenches = np.asarray((main, front), dtype=float)
        if (not sealed or not np.all(np.isfinite(wrenches))
                or np.linalg.norm(wrenches[:, :3], axis=1).max() > 9.
                or np.linalg.norm(wrenches[:, 3:], axis=1).max() > 1.):
            raise RuntimeError(f"Horizontal assistance lost the front seal or wrist envelope: {wrenches.tolist()}")
        if not np.isfinite(lower_impulse_Ns) or lower_impulse_Ns < 0:
            raise RuntimeError("Horizontal assistance requires fresh native support impulses")
        if abs(float(front[2])) > .20 * self.weight_N or -float(wrenches[:, 2].sum()) > .25 * self.weight_N:
            raise RuntimeError(f"Horizontal assistance applied vertical load: {wrenches[:, 2].tolist()}")
        bearing = 120 * lower_impulse_Ns >= .35 * self.weight_N
        self.missing = 0 if bearing else self.missing + 1
        self.impulses.append(lower_impulse_Ns)
        if self.missing >= 3 or (len(self.impulses) == self.impulses.maxlen
                                 and 120 * np.mean(self.impulses) < .70 * self.weight_N):
            raise RuntimeError("Lower support load interrupted during horizontal assistance")
        return bearing

    def command_x(self, *, measured, nominal, previous, main, front, yaw_error_rad, unloading=False):
        main_push = max(0., -self.sign * main[0])
        front_shear = -self.sign * front[0]
        yaw_scale = max(0., 1. - abs(yaw_error_rad) / np.deg2rad(2.))
        # Keep meaningful compression on the main contact and leave total force headroom.
        assist = 0. if unloading else yaw_scale * min(2., .40 * main_push, max(0., 8.4 - main_push))
        lead = self.lead.update(front_shear, target_force_N=assist)
        signed_goal = min(self.sign * measured + lead, self.sign * nominal + .0008)
        lo = self.sign * previous - .00045 / 120
        hi = self.sign * previous + (.0012 if unloading else .024) / 120
        return self.sign * float(np.clip(signed_goal, lo, hi))


def inject(source):
    """Patch a compiled-runner source string. This module does not change the default launcher."""
    source = replace_once(source, '        def close_upper_side_seal(side):',
                          '        def close_upper_side_seal(side, seal_phase="upper_B_seal"):')
    source = replace_once(source, '            record("upper_B_seal", "CLOSED_AT_LOW_PRELOAD",',
                          '            record(seal_phase, "CLOSED_AT_LOW_PRELOAD",')
    start = source.index('            def push_to_boundary(side, axis, sign, phase, target,')
    end = source.index('                if travel_limit is not None:\n                    return', start)
    body = source[start:end]
    body = replace_once(body, '                nonlocal latched_contact_axis\n',
                        '                nonlocal latched_contact_axis, left_hold\n'
                        '                from upper_row_horizontal_assist import HorizontalLoadShare\n')
    body = replace_once(body, '                    max_axial_lead = max_vertical_load = 0.\n',
                        '                    max_axial_lead = max_vertical_load = 0.\n'
                        '                    share = None\n'
                        '                    bearing = True\n')
    body = replace_once(body, '                        nonlocal max_vertical_load\n',
                        '                        nonlocal max_vertical_load, bearing\n'
                        '                        if share is not None:\n'
                        '                            main_wrench = last_force_sample[side]["tcp_gravity_corrected_world"]\n'
                        '                            front_wrench = last_force_sample[follow]["tcp_gravity_corrected_world"]\n'
                        '                            bearing = share.check(main_wrench, front_wrench, support_impulse(),\n'
                        '                                                  grippers.is_closed(follow))\n'
                        '                            bearing = bearing and .10 <= -front_wrench[1] <= 2.5\n')
    body = replace_once(body, '                            if measured > 7.35 or resultant > 7.8:\n',
                        '                            front_pulling_alone = (share is not None and measured < .10\n'
                        '                                and -sign * last_force_sample[follow]["tcp_gravity_corrected_world"][0] > .4)\n'
                        '                            if not bearing or front_pulling_alone or measured > 7.35 or resultant > 7.8:\n')
    body = replace_once(body, '                        vertical_error = vertical_force + 1.5\n',
                        '                        vertical_error = vertical_force if share is not None else vertical_force + 1.5\n')
    body = replace_once(body, '                        if abs(offset) <= 1e-6:\n', '''                        if share is not None and not bearing:
                            return {arm: applied_targets[arm].copy() for arm in commands}
                        if share is not None:
                            actual_front, _ = tcp_fk(follow, joint_state()[follow])
                            previous_front, _ = tcp_fk(follow, applied_targets[follow])
                            front, front_rotation = tcp_fk(follow, commands[follow])
                            front[1] = left_hold.position_m
                            front[2] = left_hold_z
                            _, cube_rotation = sensed_cube()
                            yaw_delta = Rotation.from_matrix(rotation.T @ cube_rotation).as_rotvec()[2]
                            front[0] = share.command_x(measured=actual_front[0], nominal=front[0],
                                                       previous=previous_front[0], yaw_error_rad=yaw_delta,
                                                       main=last_force_sample[side]["tcp_gravity_corrected_world"],
                                                       front=last_force_sample[follow]["tcp_gravity_corrected_world"])
                            commands[follow] = cartesian_joint_target(follow, commands[follow], front, front_rotation)
                            if np.max(np.abs(commands[follow] - applied_targets[follow])) * 120 > .401:
                                raise RuntimeError(f"{phase}: horizontal assistance exceeds joint speed limit")
                        if abs(offset) <= 1e-6 and share is None:
''')
    body = replace_once(body, '                    latched_contact_axis = axis\n                    try:\n', '''                    def unload_front_share():
                        record("upper_B_horizontal_assist", "UNLOADING_FRONT_SHEAR", arm=follow)
                        nominal, front_rotation = tcp_fk(follow, applied_targets[follow])
                        quiet = 0
                        for _ in range(480):
                            step()
                            main_wrench = last_force_sample[side]["tcp_gravity_corrected_world"]
                            front_wrench = last_force_sample[follow]["tcp_gravity_corrected_world"]
                            bearing = share.check(main_wrench, front_wrench, support_impulse(), grippers.is_closed(follow))
                            check_current_collision("upper_B_horizontal_assist_unload")
                            quiet = quiet + 1 if bearing and abs(front_wrench[0]) <= .25 else 0
                            if quiet >= 12:
                                return
                            measured, _ = tcp_fk(follow, joint_state()[follow])
                            point, _ = tcp_fk(follow, applied_targets[follow])
                            point[0] = share.command_x(measured=measured[0], nominal=nominal[0], previous=point[0],
                                                       main=main_wrench, front=front_wrench, yaw_error_rad=0., unloading=True)
                            q = cartesian_joint_target(follow, applied_targets[follow], point, front_rotation)
                            if np.max(np.abs(q - applied_targets[follow])) * 120 > .401:
                                raise RuntimeError(f"{phase}: assistance unload exceeds joint speed limit")
                            command_positions(follow, q)
                        raise RuntimeError(f"{phase}: front shear {front_wrench[0]:.3f} N did not unload")

                    latched_contact_axis = axis
                    try:
                        if outer_side_push and follow is not None:
                            left_hold = None
                            close_upper_side_seal(follow, "upper_B_horizontal_assist_dock")
                            weight = CONFIG["physics"]["cube_mass_kg"] * abs(CONFIG["physics"]["gravity_m_s2"][2])
                            share = HorizontalLoadShare(sign, weight)
                            hold_front(.65)
                            quiet = 0
                            for _ in range(480):
                                step()
                                main_wrench = last_force_sample[side]["tcp_gravity_corrected_world"]
                                front_wrench = last_force_sample[follow]["tcp_gravity_corrected_world"]
                                bearing = share.check(main_wrench, front_wrench, support_impulse(), grippers.is_closed(follow))
                                check_current_collision("upper_B_horizontal_assist_preload")
                                quiet = quiet + 1 if bearing and .25 <= -front_wrench[1] <= 1.5 else 0
                                if quiet >= 12:
                                    break
                            else:
                                raise RuntimeError(f"{phase}: front seal did not recover rear support pressure")
                            record("upper_B_horizontal_assist", "FRONT_SEAL_CLOSED", arm=follow,
                                   support_required=True, vertical_force_target_N=0.)
''')
    body = replace_once(body, '                    finally:\n                        latched_contact_axis = None\n', '''                        if share is not None:
                            unload_front_share()
                            grippers.open(follow)
                            record("upper_B_horizontal_assist", "FRONT_SEAL_OPENED_ON_SUPPORT", arm=follow)
                    finally:
                        if outer_side_push and follow is not None and grippers.is_closed(follow):
                            grippers.open(follow)
                        latched_contact_axis = None
''')
    return source[:start] + body + source[end:]
