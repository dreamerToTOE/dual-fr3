"""Supported U4 lateral push with one front-face orientation seal and scoped diagnostics."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
from statistics import median
import sys
from textwrap import indent

import numpy as np

from upper_row_physx import HERE, REPO, replace_once
from upper_slot4_candidate import GUIDE


class CompliantPushLead:
    """Use measured sliding resistance to build only the axial lead needed for contact."""

    def __init__(self, *, slide_lead_m=.0016, initial_lead_m=.00065):
        self.sliding_N = deque(maxlen=48)
        self.sticking = deque(maxlen=48)
        self.lead_m = initial_lead_m
        self.slide_lead_m = slide_lead_m
        self.target_N = 5.7
        self.breakaway_N = 5.7

    def command(self, *, measured_x, commanded_x, axial_N, force_N, moving, supported, target_contact,
                speed_m_s):
        if moving and not target_contact and axial_N > 0.:
            self.sliding_N.append(axial_N)
        if target_contact:
            self.sticking.clear()
            self.target_N, max_lead = 2.1, .00065
        else:
            # Native U4 breakaway occurred around 5.5–6 N. Rolling motion raises the target only
            # when measured resistance warrants it, leaving headroom below the 9 N hard guard.
            resistance = median(self.sliding_N) if self.sliding_N else 5.25
            self.target_N = min(6.6, max(self.breakaway_N, resistance + .45))
            self.sticking.append(supported and not moving and axial_N >= self.target_N - .15 and force_N < 7.65)
            if len(self.sticking) == self.sticking.maxlen and sum(self.sticking) >= 36:
                self.breakaway_N = self.target_N = min(6.6, self.target_N + .15)
                self.sticking.clear()
            max_lead = self.slide_lead_m
        if force_N > 8.1:
            self.lead_m = max(.0002, self.lead_m - .0011 / 120)
            return min(commanded_x, max(measured_x, commanded_x - .00045 / 120))
        if not supported or force_N >= 7.65 or (target_contact and axial_N >= 1.75):
            return commanded_x
        correction = .0010 * (self.target_N - axial_N)
        lead_step = max(-.0011, min(.0011, correction)) / 120
        self.lead_m = min(max_lead, max(.0002, self.lead_m + lead_step))
        desired_x = measured_x + self.lead_m
        delta = max(-.00045 / 120, min(speed_m_s / 120, desired_x - commanded_x))
        return commanded_x + delta


class SupportedPushHeight:
    """Damp coupled vertical contact and keep U4's weight on the lower boxes."""

    def __init__(self, main_z, front_z, weight_N):
        self.anchor = np.array([main_z, front_z])
        self.goal = self.anchor.copy()
        self.measured = self.anchor.copy()
        self.force = np.zeros(2)
        self.rate = np.zeros(2)
        self.weight_N = weight_N
        self.pitch = self.pitch_arm = 0.
        self.hold_ticks = self.lateral_holds = 0
        self.max_offset = np.zeros(2)

    def permit_lateral_step(self, main, front):
        if -(main[2] + front[2]) > .15 * self.weight_N or abs(front[2]) > .13 * self.weight_N:
            self.hold_ticks = 4
        if self.hold_ticks:
            self.hold_ticks -= 1
            self.lateral_holds += 1
            return False
        return True

    def update(self, main, front, measured_z, attitude):
        z_force = np.array([main[2], front[2]])
        self.force += .20 * (z_force - self.force)
        self.rate += .25 * (120 * (measured_z - self.measured) - self.rate)
        self.measured = measured_z
        # Positive sensor Z means a downward force on the box. Leave a small supported preload.
        error = self.force - [.35, .25]
        shared = min(0., float(self.force.sum()) - .60)
        speed = .00036 * error + .00022 * shared - .45 * self.rate
        limit = .0012 / 120
        desired = self.goal + np.clip(speed / 120, -limit, limit)
        # The old main height could run more than .45 mm above its measured position.
        # Limit stored vertical lead even if tangential sticking moves the surface patch.
        desired = np.clip(desired, measured_z - .00030, measured_z + [.00025, .00020])
        desired = np.clip(desired, self.anchor - .0012, self.anchor + .0012)
        self.goal += np.clip(desired - self.goal, -limit, limit)
        self.max_offset = np.maximum(self.max_offset, np.abs(self.goal - self.anchor))

        normal = -float(np.dot(main[:3], attitude[:, 0]))
        if normal >= 1.5:
            # Left arm pushes +X. Negative pitch moment means the upper cups are loaded.
            arm = np.clip(float(np.dot(main[3:], attitude[:, 1])) / normal, -.040, .040)
            self.pitch_arm += .15 * (arm - self.pitch_arm)
            error = math.copysign(max(0., abs(self.pitch_arm) - .006), self.pitch_arm)
            step = math.radians(.14) / 120 * float(np.clip(error / .024, -1., 1.))
            self.pitch = float(np.clip(self.pitch + step, -math.radians(.30), math.radians(.30)))
        return self.goal, self.pitch


class SupportedFrontYaw:
    """Correct absolute box yaw using fresh supported RGB and the native front restraint."""

    def __init__(self):
        self.samples = deque(maxlen=15)
        self.last_exposure = None
        self.target = math.radians(-.08)
        self.angle = 0.
        self.measured = None
        self.adjusted_ticks = 0
        self.peak_angle = 0.

    def update(self, observation, now, wrist):
        stamp = observation.get("supported_capture_time_s")
        matrix = observation.get("supported_rotation_world")
        if stamp is not None and stamp != self.last_exposure and 0. <= now - stamp <= .12 and matrix is not None:
            self.samples.append((stamp, math.atan2(matrix[1][0], matrix[0][0])))
            self.last_exposure = stamp
        while self.samples and now - self.samples[0][0] > .65:
            self.samples.popleft()
        if (len(self.samples) < 9 or now - self.samples[-1][0] > .12
                or np.linalg.norm(wrist[:3]) >= 7.65 or np.linalg.norm(wrist[3:]) >= .75):
            return
        self.measured = median(angle for _, angle in self.samples)
        error = self.target - self.measured
        error = math.copysign(max(0., abs(error) - math.radians(.035)), error)
        rate = math.radians(.10)
        self.angle = float(np.clip(self.angle + np.clip(.8 * error, -rate, rate) / 120,
                                   -math.radians(.55), math.radians(.55)))
        self.peak_angle = max(self.peak_angle, abs(self.angle))
        self.adjusted_ticks += 1


FRONT_PUSH = """
def push_u4_with_front_seal():
    from fr3_collision import obb_signed_clearance
    from upper_row_horizontal_assist import HorizontalLoadShare
    from upper_slot4_front_seal import CompliantPushLead, SupportedFrontYaw, SupportedPushHeight

    nonlocal left_hold, slot4_sealed_pose, latched_contact_axis
    main, front = push_side, support_side
    if (main, front, push_sign) != ("left", "right", 1) or not side_first:
        raise RuntimeError("U4 front seal requires the checked left-main side-first route")
    if any(grippers.is_closed(arm) for arm in robots):
        raise RuntimeError("U4 must enter its lateral push with both seals open")
    left_hold = None
    command_positions(main, joint_state()[main])
    before, before_rotation = sensed_cube()
    phase = "upper_B_side"
    if slot == 4:
        target_name = side_target.rsplit("/", 1)[-1]
        neighbor = next((box for box in checker.other_cubes if box[0].rsplit("/", 1)[-1] == target_name), None)
        if neighbor is None:
            raise RuntimeError("U4 requires an optically observed neighbor pose for side seating")
        _, neighbor_center, neighbor_half, neighbor_rotation = neighbor
        active_half = np.full(3, half)

    def neighbor_gap(center, attitude):
        if slot != 4:
            return gap(center, attitude, 0, 1)
        return obb_signed_clearance(center, active_half, attitude, neighbor_center, neighbor_half, neighbor_rotation)

    support_guard = HorizontalLoadShare(1, CONFIG["physics"]["cube_mass_kg"]
                                        * abs(CONFIG["physics"]["gravity_m_s2"][2]))
    try:
        close_upper_side_seal(front, "upper_B_u4_front_seal_dock")
        measured_joints = joint_state()
        anchor_main, main_rotation = tcp_fk(main, measured_joints[main])
        anchor_front, front_rotation = tcp_fk(front, measured_joints[front])
        relative_center = front_rotation.T @ (before - anchor_front)
        relative_rotation = front_rotation.T @ before_rotation

        def sealed_pose():
            tcp, rotation = tcp_fk(front, joint_state()[front])
            center = tcp + rotation @ relative_center
            attitude = rotation @ relative_rotation
            rgb = latest_camera_observation or {}
            stamp = rgb.get("supported_capture_time_s")
            if stamp is not None and 0. <= world.current_time - stamp <= .12:
                observed = np.asarray(rgb["supported_position_m"])
                correction = observed - center
                if np.linalg.norm(correction[:2]) <= .003:
                    center[:2] += .20 * correction[:2]
                delta = np.asarray(rgb["supported_rotation_world"]) @ attitude.T
                relative_yaw = math.atan2(delta[1, 0], delta[0, 0])
                if abs(relative_yaw) <= math.radians(2.):
                    attitude = Rotation.from_rotvec([0., 0., .20 * relative_yaw]).as_matrix() @ attitude
            return center, attitude

        slot4_sealed_pose = sealed_pose
        latched_contact_axis = 0
        point_main = anchor_main.copy()
        point_front = anchor_front.copy()
        grounded_height = SupportedPushHeight(anchor_main[2], anchor_front[2], support_guard.weight_N) if slot == 4 else None
        front_yaw = SupportedFrontYaw() if slot == 4 else None
        front_transverse = anchor_front.copy()
        x_relief = 0.
        main_vertical = 0.
        best_x = float(before[0])
        waiting = main_loaded = support_ticks = 0
        peak_force = peak_torque = peak_yaw = 0.
        confirms = deque(maxlen=30)
        record("upper_B_u4_front_seal", "FRONT_SEAL_CLOSED", arm=front,
               main_arm=main, target=active_cube_path, front_tcp_anchor_m=anchor_front.tolist(),
               native_bend_angle_deg=ATTACHMENT_PARAMETERS["bend_angle_deg"],
               max_front_vertical_N=.20 * support_guard.weight_N)
        record("upper_B_u4_main_seating", "RUNNING", arm=main)

        def sample():
            nonlocal support_ticks, peak_force, peak_torque, peak_yaw, main_loaded
            main_force = np.asarray(last_force_sample[main]["tcp_gravity_corrected_world"])
            front_force = np.asarray(last_force_sample[front]["tcp_gravity_corrected_world"])
            bearing = support_guard.check(main_force, front_force, support_impulse(),
                                          grippers.is_closed(front) and not grippers.is_closed(main))
            support_ticks += int(bearing)
            peak_force = max(peak_force, np.linalg.norm(main_force[:3]), np.linalg.norm(front_force[:3]))
            peak_torque = max(peak_torque, np.linalg.norm(main_force[3:]), np.linalg.norm(front_force[3:]))
            center, attitude = sensed_cube()
            delta = attitude @ before_rotation.T
            yaw = math.atan2(delta[1, 0], delta[0, 0])
            peak_yaw = max(peak_yaw, abs(yaw))
            if abs(yaw) > math.radians(2.):
                raise RuntimeError(f"U4 front seal could not restrain yaw ({math.degrees(yaw):.2f} deg)")
            compression = -float(main_force[0])
            main_loaded += int(compression >= .20)
            return center, attitude, main_force, front_force, compression, bearing

        def command(main_point, front_point, center, attitude, front_target_rotation):
            main_target_rotation = main_rotation
            if grounded_height is not None:
                wrists = [last_force_sample[arm] for arm in (main, front)]
                measured_z = np.array([wrist["tcp_position_m"][2] for wrist in wrists])
                wrench_main, wrench_front = [np.asarray(wrist["tcp_gravity_corrected_world"]) for wrist in wrists]
                heights, pitch = grounded_height.update(wrench_main, wrench_front, measured_z, attitude)
                main_point, front_point = main_point.copy(), front_point.copy()
                main_point[2], front_point[2] = heights
                main_target_rotation = Rotation.from_rotvec(pitch * attitude[:, 1]).as_matrix() @ main_rotation
            if front_yaw is not None and front_yaw.angle:
                # Rotate the sealed tool about the sensed box center, including the moment-arm translation.
                correction = Rotation.from_rotvec([0., 0., front_yaw.angle]).as_matrix()
                front_point = center + correction @ (front_point - center)
                front_target_rotation = correction @ front_target_rotation
            goals = {main: cartesian_joint_target(main, applied_targets[main], main_point, main_target_rotation),
                     front: cartesian_joint_target(front, applied_targets[front], front_point, front_target_rotation)}
            if any(np.max(np.abs(goals[arm] - applied_targets[arm])) * 120 > .4 for arm in goals):
                raise RuntimeError("U4 restrained push exceeds the 0.4 rad/s joint speed limit")
            issues = checker.check(goals, cube_center=center, cube_rotation=attitude,
                                   cup_contacts=allowed_contacts, margin_m=0., first_only=True)
            if issues:
                raise RuntimeError(f"U4 restrained push predicted a collision: {issues[0]}")
            for arm, target in goals.items():
                command_positions(arm, target)

        def left_native_contact():
            return any(pair["current_step_observed"] and (pair["normal_impulse_Ns"] or 0.) > 1e-5
                       and pair["label"] == "left_cup"
                       for pair in contact_monitor.cube_contact_snapshot()["pairs"])

        # Establish the left physical contact while the sealed front holds its measured docking pose.
        baseline = deque(maxlen=12)
        for _ in range(420):
            center, attitude, _, front_force, compression, bearing = sample()
            if bearing and compression >= .35 and left_native_contact():
                baseline.append(front_force[0])
                if len(baseline) == baseline.maxlen:
                    break
            else:
                baseline.clear()
                if bearing and compression < .35:
                    point_main[0] += .00055 / 120
            if point_main[0] - anchor_main[0] > .0014:
                raise RuntimeError("U4 left pusher did not seat before front X travel")
            command(point_main, point_front, center, attitude, front_rotation)
            step()
            check_current_collision("upper_B_u4_main_seating")
        else:
            raise RuntimeError("U4 left pusher did not maintain native contact before front travel")
        front_static_x_force = float(np.median(baseline))
        center, attitude = sensed_cube()
        left_seated, _ = tcp_fk(main, joint_state()[main])
        right_seated, _ = tcp_fk(front, joint_state()[front])
        start_x = float(center[0])
        record("upper_B_u4_front_seal", "MAIN_CONTACT_ESTABLISHED", arm=main, main_contact_ticks=12,
               main_compression_N=float(compression), stationary_front_x_force_N=front_static_x_force)
        record(phase, "RUNNING", arm=main, contact_target=side_target, front_suction_yaw_restraint=True)
        contact_monitor.phase = phase + "_travel"
        lead = CompliantPushLead()
        peak_lead_m = 0.
        last_supported_x = start_x
        previous_x = start_x
        advancing_ticks = advancing_left_ticks = advancing_native_left = front_drove = 0
        # Grounded travel may slow under load. The independent no-progress guard still checks every tick.
        travel_ticks = max(10000, math.ceil(120 * max(0., neighbor_gap(center, attitude)) / .00055) + 1200)
        for _ in range(travel_ticks):
            center, attitude, main_force, front_force, compression, bearing = sample()
            left_contact = left_native_contact()
            front_drive = max(0., front_static_x_force - float(front_force[0]))
            advancing = center[0] > previous_x + .000002
            if advancing:
                advancing_ticks += 1
                advancing_left_ticks += int(compression >= .20)
                advancing_native_left += int(left_contact)
                front_drove = front_drove + 1 if front_drive > max(.55, .8 * compression) else 0
                if front_drove >= 8:
                    raise RuntimeError("U4 front seal repeatedly drove ahead of the left pusher")
            previous_x = float(center[0])
            remaining = neighbor_gap(center, attitude)
            if remaining < -.002:
                raise RuntimeError(f"U4 front seal estimated more than 2 mm target overlap: {remaining:.4f} m")
            native_target = any(
                pair["current_step_observed"] and (pair["normal_impulse_Ns"] or 0.) > 1e-5
                and {pair["collider0"], pair["collider1"]} == {active_cube_path, side_target}
                for pair in contact_monitor.cube_contact_snapshot()["pairs"])
            confirms.append((native_target and compression >= 1.7, float(center[0]), compression))
            if (remaining <= .0015 and len(confirms) == confirms.maxlen
                    and sum(x[0] for x in confirms) >= 24
                    and max(x[1] for x in confirms) - min(x[1] for x in confirms) < .00045):
                if (advancing_ticks < 12 or advancing_left_ticks < .9 * advancing_ticks
                        or advancing_native_left < .8 * advancing_ticks):
                    raise RuntimeError("U4 lateral travel did not sustain its left physical pusher")
                record(phase, "FORCE_CONFIRMED", arm=main, contact_target=side_target,
                       held_samples=30, normal_force_N=float(np.mean([x[2] for x in confirms])),
                       normal_force_samples_N=[x[2] for x in confirms], estimated_gap_m=float(remaining),
                       witnessed_target_contact_ticks=sum(x[0] for x in confirms))
                break
            if center[0] > best_x + .00015:
                best_x = float(center[0])
                waiting = 0
            else:
                waiting += 1
            if waiting >= 1200:
                raise RuntimeError(f"U4 front-seal push stalled at {remaining*1000:.2f} mm from Cube 8 "
                                   f"(axial {compression:.2f} N, target {lead.target_N:.2f} N, "
                                   f"lead allowance {lead.lead_m*1000:.2f} mm)")
            actual_main, _ = tcp_fk(main, joint_state()[main])
            supported_to_advance = bearing and (grounded_height is None
                                                or grounded_height.permit_lateral_step(main_force, front_force))
            if front_yaw is not None and bearing and left_contact and not native_target and remaining <= .040:
                front_yaw.update(latest_camera_observation or {}, float(world.current_time), front_force)
                report["u4_front_yaw_control"] = {
                    "optical_target_deg": math.degrees(front_yaw.target),
                    "filtered_optical_yaw_deg": None if front_yaw.measured is None else math.degrees(front_yaw.measured),
                    "corrected_ticks": front_yaw.adjusted_ticks,
                    "commanded_correction_deg": math.degrees(front_yaw.angle),
                    "max_commanded_correction_deg": math.degrees(front_yaw.peak_angle),
                }
            point_main[0] = lead.command(
                measured_x=actual_main[0], commanded_x=point_main[0], axial_N=compression,
                force_N=float(np.linalg.norm(main_force[:3])), moving=advancing and left_contact,
                supported=supported_to_advance, target_contact=native_target,
                speed_m_s=.006 if remaining > .006 and not native_target else .00035)
            peak_lead_m = max(peak_lead_m, float(point_main[0] - actual_main[0]))
            if grounded_height is None:
                main_vertical = float(np.clip(main_vertical + .00020 * main_force[2] / 120, -.0008, .0008))
                point_main[2] = anchor_main[2] + main_vertical
            if bearing and compression >= .20 and left_contact:
                inferred = start_x + actual_main[0] - left_seated[0]
                fused_x = .65 * center[0] + .35 * np.clip(inferred, center[0] - .0015, center[0] + .0015)
                last_supported_x = max(last_supported_x, float(fused_x))
            x_relief = float(np.clip(x_relief + .0008 * (front_force[0] - front_static_x_force) / 120,
                                     -.001, .001))
            front_x = right_seated[0] + last_supported_x - start_x + x_relief
            front_step = .007 / 120
            point_front[0] += float(np.clip(front_x - point_front[0], -front_step, front_step))
            if grounded_height is None:
                front_transverse[1:] += np.clip(.00045 * front_force[1:3], -.0008, .0008) / 120
                front_transverse[1:] = np.clip(front_transverse[1:], anchor_front[1:] - .001,
                                               anchor_front[1:] + .001)
                point_front[1:] = front_transverse[1:]
            else:
                front_y = point_front[1] + float(np.clip(.00045 * front_force[1], -.0008, .0008)) / 120
                point_front[1] = float(np.clip(front_y, anchor_front[1] - .001, anchor_front[1] + .001))
            command(point_main, point_front, center, attitude, front_rotation)
            step()
            check_current_collision("upper_B_u4_front_seal")
        else:
            raise RuntimeError("U4 front-seal lateral push timed out")

        # Let the native restraint unload against the supported, seated box before opening it.
        quiet = 0
        for _ in range(480):
            center, attitude, _, front_force, _, bearing = sample()
            quiet = quiet + 1 if bearing and np.linalg.norm(front_force[:3]) < 1.2 else 0
            if quiet >= 12:
                break
            point_front += np.clip(.0004 * front_force[:3], -.0006, .0006) / 120
            point_front[1:] = np.clip(point_front[1:], anchor_front[1:] - .0012, anchor_front[1:] + .0012)
            command(point_main, point_front, center, attitude, front_rotation)
            step()
            check_current_collision("upper_B_u4_front_seal_unload")
        else:
            raise RuntimeError("U4 front seal did not unload before opening")
        record("upper_B_u4_front_seal", "FRONT_SEAL_OPENING", arm=front, main_arm=main)
        grippers.open(front)
        for _ in range(12):
            step()
            force(main, 0, 1)
            force(front, 1, 1)
            if support_impulse() <= 1e-5:
                raise RuntimeError("U4 lost lower support following front-seal release")
            check_current_collision("upper_B_u4_front_seal_release")
        if grounded_height is not None:
            report["u4_grounded_vertical"] = {
                "lateral_hold_ticks": grounded_height.lateral_holds,
                "peak_wrist_height_offsets_mm": (grounded_height.max_offset * 1000).tolist(),
                "final_pusher_pitch_correction_deg": math.degrees(grounded_height.pitch),
            }
        record("upper_B_u4_front_seal", "FRONT_SEAL_OPENED_ON_SUPPORT", arm=front, main_arm=main,
               support_loaded_ticks=support_ticks, main_loaded_ticks=main_loaded,
               advancing_ticks=advancing_ticks, left_force_ticks=advancing_left_ticks,
               left_physx_contact_ticks=advancing_native_left,
               peak_pusher_lead_mm=1000 * peak_lead_m,
               peak_force_N=float(peak_force), peak_torque_Nm=float(peak_torque),
               peak_yaw_deg=math.degrees(peak_yaw), supported_open_ticks=12)
    finally:
        grippers.open(front)
        slot4_sealed_pose = None
        latched_contact_axis = None
"""


REAR_PUSH = """
def push_u4_to_rear(side, phase, target):
    from upper_slot4_front_seal import CompliantPushLead

    nonlocal latched_contact_axis
    if side != "right" or any(grippers.is_closed(arm) for arm in robots):
        raise RuntimeError("U4 rear push requires the right front cups open and the left arm parked")
    center, attitude = sensed_cube()
    neighbor_name = side_target.rsplit("/", 1)[-1]
    neighbor = next((box for box in checker.other_cubes if box[0].rsplit("/", 1)[-1] == neighbor_name), None)
    if neighbor is None:
        raise RuntimeError("U4 rear push needs the observed side of the seated neighbor")
    side_normal = neighbor[3][:, 0]
    if side_normal[0] < .95 or abs(side_normal[1]) > .035:
        raise RuntimeError("U4 neighbor side lies outside the checked rear approach corridor")
    side_slope = -float(side_normal[1] / side_normal[0])

    q = applied_targets[side].copy()
    tcp, rotation = tcp_fk(side, q)
    anchor = tcp.copy()
    measured, _ = tcp_fk(side, joint_state()[side])
    center[1] = front_contact_center_y(side, center, attitude)
    initial = center.copy()
    start_gap = gap(center, attitude, 1, 1)
    if start_gap < -.002 or start_gap > .080:
        raise RuntimeError(f"{phase}: initial rear gap {start_gap*1000:.2f} mm is outside the checked corridor")
    lead = CompliantPushLead(slide_lead_m=.00245,
                             initial_lead_m=float(np.clip(tcp[1] - measured[1], .0002, .00245)))
    if start_gap > .00275:
        _, translation = supported_travel(initial, 1, 1, start_gap - .0025)
        speed_limit = 120 * np.diff(translation)
    else:
        translation, speed_limit = np.array([0., .00025]), np.array([.00035])
    record(phase, "RUNNING", arm=side, contact_target=target, estimated_start_gap_m=float(start_gap),
           control="rear_contact_admittance", neighbor_side_slope_x_per_y=side_slope)
    contact_monitor.phase = phase + "_travel"
    latched_contact_axis = 1
    horizontal_relief = vertical_relief = 0.
    filtered_tangent = np.zeros(2)
    best_y = float(initial[1])
    waiting = support_missing = pad_missing = 0
    measured_window = deque(maxlen=48)
    confirmations = deque(maxlen=settle_frame_count)
    peak_lead = peak_force = peak_torque = 0.
    summary = report["u4_rear_contact_admittance"] = {"max_lead_m": lead.slide_lead_m}

    try:
        travel_ticks = max(4500, math.ceil(120 * max(0., start_gap) / .0005) + 1800)
        for _ in range(travel_ticks):
            normal = force(side, 1, 1)
            wrench = np.asarray(last_force_sample[side]["tcp_gravity_corrected_world"])
            resultant = float(np.linalg.norm(wrench[:3]))
            torque = float(np.linalg.norm(wrench[3:]))
            peak_force, peak_torque = max(peak_force, resultant), max(peak_torque, torque)
            measured, _ = tcp_fk(side, joint_state()[side])
            center, attitude = sensed_cube()
            if normal >= .15:
                center[1] = front_contact_center_y(side, center, attitude)
            near = gap(center, attitude, 1, 1)
            if near < -.002 or abs(center[0] - initial[0]) > .003:
                raise RuntimeError(f"{phase}: encoder/RGB estimate left the checked rear corridor "
                                   f"(rear gap {near*1000:.2f} mm, lateral change "
                                   f"{(center[0] - initial[0])*1000:.2f} mm)")
            pairs = contact_monitor.cube_contact_snapshot()["pairs"]
            bearing = support_impulse() > 1e-5
            pad = any(pair["label"] == "right_cup" and pair["current_step_observed"]
                      and (pair["normal_impulse_Ns"] or 0.) > 1e-5 for pair in pairs)
            support_missing = 0 if bearing else support_missing + 1
            pad_missing = 0 if pad else pad_missing + 1
            if support_missing >= 4 or pad_missing >= 120:
                raise RuntimeError(f"{phase}: rear push lost the lower bearing or right front cup contact")

            measured_window.append(float(center[1]))
            moving = (len(measured_window) == measured_window.maxlen
                      and measured_window[-1] - measured_window[0] > .00005)
            native_wall = any(pair["current_step_observed"] and (pair["normal_impulse_Ns"] or 0.) > 1e-5
                              and {pair["collider0"], pair["collider1"]} == {active_cube_path, target}
                              for pair in pairs)
            resistance = float(np.median(lead.sliding_N)) if lead.sliding_N else 5.25
            threshold = min(7.2, max(5.2, resistance + .35))
            confirmations.append((float(center[1]), normal, native_wall, -.002 <= near <= .00075))
            if len(confirmations) == confirmations.maxlen:
                centers, loads, witness, geometry = np.asarray(confirmations).T
                if (np.ptp(centers) < .00025 and np.count_nonzero(loads >= threshold) >= 24
                        and np.count_nonzero(witness * geometry) >= 24):
                    record(phase, "FORCE_CONFIRMED", arm=side, contact_target=target,
                           held_samples=len(confirmations), normal_force_N=float(loads.mean()),
                           normal_force_samples_N=loads.tolist(), estimated_gap_m=float(near),
                           witnessed_target_contact_ticks=int(np.count_nonzero(witness)),
                           sliding_baseline_N=resistance, force_threshold_N=threshold)
                    return

            if center[1] >= best_y + .00016:
                best_y, waiting = float(center[1]), 0
            else:
                waiting += 1
            if waiting >= 1440:
                raise RuntimeError(f"{phase}: rear push stalled {near*1000:.2f} mm from the wall "
                                   f"(axial {normal:.2f} N, resultant {resultant:.2f} N, "
                                   f"physical lead {(tcp[1] - measured[1])*1000:.2f} mm, "
                                   f"native contact {sum(c[2] for c in confirmations)}/30)")

            # Follow the observed neighbor plane and unload the wrist's side/upward shear.
            filtered_tangent += .15 * (wrench[[0, 2]] - filtered_tangent)
            horizontal_relief = float(np.clip(horizontal_relief + np.clip(
                .00030 * (filtered_tangent[0] + .12), -.00065, .00065) / 120, -.0012, .0012))
            vertical_relief = float(np.clip(vertical_relief + np.clip(
                .00030 * (filtered_tangent[1] - .25), -.00065, .00065) / 120, -.0012, .0012))
            progress = max(0., float(center[1] - initial[1]))
            target_x = anchor[0] + side_slope * progress + horizontal_relief
            tcp[0] += float(np.clip(target_x - tcp[0], -.00075 / 120, .00075 / 120))
            tcp[2] = anchor[2] + vertical_relief
            index = min(len(speed_limit) - 1, np.searchsorted(translation, progress, side="right"))
            pace = max(.00035, float(speed_limit[index])) if near > .0025 else (.00012 if normal >= 4. else .00035)
            tcp[1] = lead.command(measured_x=measured[1], commanded_x=tcp[1], axial_N=normal,
                                  force_N=resultant, moving=moving and pad and near > .0025,
                                  supported=bearing and (pad or normal < .15), target_contact=False, speed_m_s=pace)
            peak_lead = max(peak_lead, float(tcp[1] - measured[1]))
            summary.update(travel_m=progress, peak_force_N=peak_force, peak_torque_Nm=peak_torque,
                           commanded_minus_measured_lead_m=float(tcp[1] - measured[1]), peak_lead_m=peak_lead,
                           side_relief_m=horizontal_relief, vertical_relief_m=vertical_relief,
                           force_target_N=lead.target_N, estimated_rear_gap_m=float(near))
            new_q = cartesian_joint_target(side, q, tcp, rotation)
            velocity = float(np.max(np.abs(new_q - q))) * 120
            if not np.isfinite(velocity) or velocity > .4:
                raise RuntimeError(f"{phase}: rear command joint velocity {velocity:.6f} rad/s exceeds 0.400 rad/s")
            issues = checker.check({**applied_targets, side: new_q}, cube_center=center, cube_rotation=attitude,
                                   cup_contacts=allowed_contacts, margin_m=0., first_only=True)
            if issues:
                raise RuntimeError(f"{phase}: rear command in collision: {issues[0]}")
            q = new_q
            command_positions(side, q)
            step()
            check_current_collision(phase)
        raise RuntimeError(f"{phase}: rear push reached its bounded travel time at gap {near*1000:.2f} mm")
    finally:
        latched_contact_axis = None
"""


def inject(source):
    """Apply only to the generated U4 candidate. Other slots keep their normal commands."""
    source = replace_once(source, "        control_center = control_rotation = carry_offsets = None\n",
                          "        control_center = control_rotation = carry_offsets = None\n"
                          "        slot4_sealed_pose = None\n")
    source = replace_once(source, "            if carry_offsets is not None:\n                measured = joint_state()",
                          "            if slot4_sealed_pose is not None:\n"
                          "                control_center, control_rotation = slot4_sealed_pose()\n"
                          "            elif carry_offsets is not None:\n                measured = joint_state()")
    source = replace_once(source, '"source": "encoder_grasp_geometry" if carry_offsets is not None else',
                          '"source": "front_side_seal_encoder_rgb" if slot4_sealed_pose is not None else\n'
                          '                          "encoder_grasp_geometry" if carry_offsets is not None else')
    source = replace_once(source, "            def push_to_boundary(side, axis, sign, phase, target, follow=None, travel_limit=None):\n",
                          indent(REAR_PUSH.strip() + "\n\n", "            ") +
                          "            def push_to_boundary(side, axis, sign, phase, target, follow=None, travel_limit=None):\n"
                          "                if slot == 4 and axis == 1 and phase == 'upper_B_rear' and target == rear_target:\n"
                          "                    return push_u4_to_rear(side, phase, target)\n")
    source = replace_once(source, "            def approach_front():\n",
                          indent(FRONT_PUSH.strip() + "\n\n", "            ") + "            def approach_front():\n")
    source = replace_once(source,
                          '                push_to_boundary(push_side, 0, push_sign, "upper_B_side", side_target, follow=support_side)',
                          '                if slot == 4:\n'
                          '                    push_u4_with_front_seal()\n'
                          '                else:\n'
                          '                    push_to_boundary(push_side, 0, push_sign, "upper_B_side", side_target, follow=support_side)')
    return replace_once(source, '            if side_first:\n                left_hold = None\n',
                        '            if side_first:\n                left_hold = None\n'
                        '                if slot == 4:\n'
                        '                    touch_face(support_side, contact_plan_rotation[:, 1], .006, '
                        '"upper_B_u4_front_retouch")\n')
