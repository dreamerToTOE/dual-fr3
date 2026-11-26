"""Test continuous left-side pressure during the markerless upper-4 rear push."""

import argparse
import json
from pathlib import Path
import sys


def clearance(checker, q, position, attitude, half):
    from fr3_collision import _sphere_box_clearances, obb_signed_clearance
    import numpy as np

    centers, tools = checker.geometry("left", q)
    link_gaps = _sphere_box_clearances(centers, checker.radii, position, half, attitude)
    link = int(np.argmin(link_gaps))
    distances = [(f"fr3_link{checker.link_indices[link]}", float(link_gaps[link]))]
    distances.extend(
        (name, obb_signed_clearance(center, extent, rotation, position, half, attitude))
        for name, center, extent, rotation in tools
    )
    return distances


FOLLOW_TRIAL = '''
            if slot == 4 and RELEASE_CHECKPOINT.get("side_follow_start"):
                cfg = RELEASE_CHECKPOINT["side_follow_start"]
                restored_tcp = {}
                for arm, saved_joints in cfg["joint_rad"].items():
                    q = np.asarray(saved_joints, dtype=float)
                    base = rail_states[arm].copy()
                    base[1] = cfg["rail_y_m"][arm]
                    set_robot_base_pose(arm, base)
                    tcp, attitude = tcp_fk(arm, q)
                    restored_tcp[arm] = (tcp, attitude)
                    # Native contact impulses are not in the checkpoint. Start
                    # clear of the two faces and reacquire them in PhysX.
                    delta = np.array([-.010, 0., 0.] if arm == "left" else [0., -.004, 0.])
                    q = cartesian_joint_target(arm, q, tcp + delta, attitude, retarget_guide=True)
                    robots[arm].set_joint_positions(q, joint_indices=joints[arm])
                    robots[arm].set_joint_velocities(np.zeros(7), joint_indices=joints[arm])
                    command_positions(arm, q)
                allowed_contacts.update(("left", "right"))
                set_contact_drive_mode(True)
                contact_plan_rotation = control_rotation.copy()
                center, cube_rotation = sensed_cube()
                side_tcp, side_rot = restored_tcp[push_side]
                metrics = report["u4_side_follow"] = {
                    "yaw_change_deg": U4_FOLLOW_YAW, "left_target_N": U4_LEFT_FORCE,
                    "source_tick": cfg["physics_step"], "by_phase": {}
                }

                def log_clearance():
                    measured = joint_state()
                    other = upper_bodies["9"]
                    pos, quat = other.get_world_pose()
                    reference = clearance(checker, measured["left"], np.asarray(pos),
                                          quat_to_rot_matrix(quat), np.full(3, half))
                    row = metrics["by_phase"].setdefault(contact_monitor.phase, {
                        "steps": 0, "left_load_ge_0p3N_steps": 0, "left_physx_cup_load_steps": 0,
                        "side_neighbor_load_steps": 0, "rear_load_steps": 0, "lower_support_steps": 0,
                        "left_force_sum_N": 0., "left_peak_force_N": 0., "right_peak_force_N": 0.,
                        "minimum_by_part": {}
                    })
                    row["steps"] += 1
                    left = last_force_sample["left"]["tcp_gravity_corrected_world"]
                    right = last_force_sample["right"]["tcp_gravity_corrected_world"]
                    row["left_force_sum_N"] -= float(left[0])
                    row["left_load_ge_0p3N_steps"] += int(left[0] <= -.3)
                    row["left_peak_force_N"] = max(row["left_peak_force_N"], float(np.linalg.norm(left[:3])))
                    row["right_peak_force_N"] = max(row["right_peak_force_N"], float(np.linalg.norm(right[:3])))
                    touching = [p for p in contact_monitor.cube_contact_snapshot()["pairs"]
                                if p["current_step_observed"] and (p["normal_impulse_Ns"] or 0.) > 1e-5]
                    row["left_physx_cup_load_steps"] += int(any(p["label"] == "left_cup" for p in touching))
                    row["side_neighbor_load_steps"] += int(any(side_target in (p["collider0"], p["collider1"])
                                                               for p in touching))
                    row["rear_load_steps"] += int(any(p["label"] == "wall_back" for p in touching))
                    # Match the stock upper-row acceptance check for lower support.
                    row["lower_support_steps"] += int(support_impulse() * 120 >= 6.)
                    for part, distance in reference:
                        minima = row["minimum_by_part"]
                        if part not in minima or distance < minima[part]["clearance_m"]:
                            minima[part] = {"clearance_m": distance, "step": tick}

                u4_clearance_tick = log_clearance
                record("upper_B_side_follow_restore", "CHECKPOINT_RESTORED", source_tick=cfg["physics_step"])
                for _ in range(36):
                    step()
                    check_current_collision(contact_monitor.phase, contact_uncertainty_m=.002)
                if U4_FOLLOW_YAW:
                    from scipy.spatial.transform import Rotation
                    attitude = Rotation.from_euler("z", U4_FOLLOW_YAW, degrees=True).as_matrix() @ side_rot
                    # The furthest corner of a rotated pad reaches the side before its TCP does.
                    extension = max(
                        float(attitude[0] @ [x, .005, z]) + float(np.abs(attitude[0]) @ [.010, .005, .010])
                        for x in (-.024, .024) for z in (-.024, .024)
                    )
                    target_tcp = side_tcp - [max(0., extension) + .005, 0., 0.]
                    q = cartesian_joint_target(push_side, applied_targets[push_side], target_tcp, attitude,
                                               retarget_guide=True)
                    record("upper_B_side_follow_reorient", "PLANNING", yaw_change_deg=U4_FOLLOW_YAW)
                    move({push_side: q})
                touch_face(push_side, np.array([1., 0., 0.]), .015, "upper_B_side_follow_touch")
                side_tcp, _ = tcp_fk(push_side, joint_state()[push_side])
                center, cube_rotation = sensed_cube()
                push_faces[push_side] = (0, -cube_rotation[:, 0], cube_rotation.T @ (side_tcp - center))
                push_faces[support_side] = (1, -cube_rotation[:, 1], front_offset)
                nominal, _ = tcp_fk(push_side, applied_targets[push_side])
                follower_y = float(nominal[1])
                front_origin_y = None
                side_hold = NormalForceController(nominal[0], target_force_N=U4_LEFT_FORCE, gain_m_per_Ns=.00045,
                                                 max_offset_m=.004, max_speed_m_s=.0015, max_force_N=9.)

                def follow_pressure():
                    nonlocal follower_y
                    pos, rotation = tcp_fk(push_side, nominal_targets[push_side])
                    pos[0] = side_hold.update(force(push_side, 0, 1))
                    center, attitude = sensed_cube()
                    if front_origin_y is not None:
                        # The loaded right cup gives the immediate front-face displacement.
                        if force(support_side, 1, 1) >= .15:
                            front_y = front_contact_center_y(support_side, center, attitude)
                            # Compensate for the contact drive's measured following lag.
                            right_command, _ = tcp_fk(support_side, applied_targets[support_side])
                            right_actual, _ = tcp_fk(support_side, joint_state()[support_side])
                            lead = float(np.clip(right_command[1] - right_actual[1], 0., .0015))
                            requested_y = nominal[1] + max(0., front_y - front_origin_y + .8 * lead)
                            follower_y += float(np.clip(requested_y - follower_y, -.024 / 120, .024 / 120))
                        pos[1] = follower_y
                    command = cartesian_joint_target(push_side, applied_targets[push_side], pos, rotation)
                    velocity = float(np.max(np.abs(command - applied_targets[push_side]))) * 120
                    if not np.isfinite(velocity) or velocity > .4:
                        raise RuntimeError(f"Left follow command exceeds 0.4 rad/s: {velocity:.6f} rad/s")
                    check = {**applied_targets, push_side: command}
                    issues = checker.check(check, cube_center=center, cube_rotation=attitude,
                                           cup_contacts=allowed_contacts, margin_m=0., first_only=True)
                    if issues:
                        raise RuntimeError(f"Left pressure correction in collision: {issues[0]}")
                    command_positions(push_side, command, nominal=False)

                u4_side_pressure = follow_pressure
                try:
                    record("upper_B_side_follow_preload", "HOLDING", target_force_N=U4_LEFT_FORCE)
                    for _ in range(180):
                        step()
                        check_current_collision(contact_monitor.phase)
                    left_hold = None
                    touch_face(support_side, contact_plan_rotation[:, 1], .006, "upper_B_u4_front_retouch")
                    center, attitude = sensed_cube()
                    front_origin_y = front_contact_center_y(support_side, center, attitude)
                    push_to_boundary(support_side, 1, 1, "upper_B_rear", rear_target)
                finally:
                    u4_side_pressure = None
                destinations = {}
                for arm, delta in ((push_side, [-.004, 0., 0.]), (support_side, [0., -.004, 0.])):
                    pos, _ = tcp_fk(arm, joint_state()[arm])
                    destinations[arm] = pos + delta
                paired_cartesian_move(destinations, 1.5, "upper_B_side_follow_unload")
                allowed_contacts.clear()
                push_faces.clear()
                set_contact_drive_mode(False)
                record("upper_B_side_follow_withdraw", "PLANNING", arms=["left", "right"])
                move({arm: home for arm in robots})
                u4_clearance_tick = None
                record("upper_B_release", "TOOLS_PARKED")
                return
'''


def install_trial(source, yaw, left_force, rear_speed, replace_once):
    source = replace_once(source, '        def step(count=1):',
                          '        from u4_side_follow import clearance\n'
                          f'        U4_FOLLOW_YAW = {float(yaw)!r}\n'
                          f'        U4_LEFT_FORCE = {float(left_force)!r}\n'
                          '        u4_side_pressure = u4_clearance_tick = None\n\n'
                          '        def step(count=1):')
    anchor = '                for side, robot in robots.items():\n                    gravity = robot._articulation_view'
    source = replace_once(source, anchor,
                          '                if u4_side_pressure is not None:\n'
                          '                    u4_side_pressure()\n' + anchor)
    anchor = '                last_force_sample = force_sample\n'
    source = replace_once(source, anchor, anchor +
                          '                if u4_clearance_tick is not None:\n'
                          '                    u4_clearance_tick()\n')
    anchor = '            b = plan["B"]'
    source = replace_once(source, anchor, '            nonlocal u4_side_pressure, u4_clearance_tick\n\n' + anchor)
    anchor = '            for side, y in b["rails_y_m"].items():'
    source = replace_once(source, anchor, FOLLOW_TRIAL + '\n' + anchor)
    if rear_speed is not None:
        anchor = 'supported=bearing and (pad or normal < .15), target_contact=False, speed_m_s=pace)'
        source = replace_once(source, anchor,
                              'supported=bearing and (pad or normal < .15), target_contact=False,\n'
                              f'                                          speed_m_s=min(pace, {rear_speed!r}) '
                              'if u4_side_pressure else pace)')
    anchor = "            save_upper_checkpoint(slot + 1)"
    return replace_once(source, anchor, anchor +
                        '\n            report["status"] = "CAPTURED_U4_SIDE_FOLLOW"'
                        '\n            return 0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_checkout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--yaw-deg", type=float, default=40.)
    parser.add_argument("--left-force-n", type=float, default=.65)
    parser.add_argument("--max-rear-speed-m-s", type=float)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not 0. < args.left_force_n < 9.:
        parser.error("The left-side pressure must be positive and below the existing 9 N wrist limit.")
    if args.max_rear_speed_m_s is not None and not 0. < args.max_rear_speed_m_s <= .024:
        parser.error("Specify a positive rear speed cap at or below the baseline 0.024 m/s.")
    seed = json.loads(args.resume.read_text())
    if seed["next_slot"] != 4 or not seed.get("landed", {}).get("side_follow_start"):
        parser.error("Use an upper-4 side-seating checkpoint with measured arm poses.")
    research = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(research))
    import fr3_contact_monitor

    base = args.base_checkout.resolve()
    scene = base / "scenes/w760_300mm_two_layer"
    sys.path[:0] = [str(scene), str(base)]
    import markerless_rows
    import markerless_tracking

    install = markerless_tracking.install

    def capture(source):
        return install_trial(install(source), args.yaw_deg, args.left_force_n, args.max_rear_speed_m_s,
                             markerless_rows.replace_once)

    markerless_tracking.install = capture
    sys.argv = [str(scene / "markerless_rows.py"), str(args.output.resolve()),
                "--resume", str(args.resume.resolve()), "--headless", "--close-on-finish"]
    if args.check_only:
        sys.argv.append("--check-only")
    return markerless_rows.main()


if __name__ == "__main__":
    raise SystemExit(main())
