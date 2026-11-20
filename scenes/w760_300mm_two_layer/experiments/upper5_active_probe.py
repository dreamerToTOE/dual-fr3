"""Apply the existing rear probe to Upper5 and record its causal inputs."""

from source_hooks import replace_once


def inject(source):
    source = replace_once(
        source, "            def probe_rear_after_seam(side, phase):",
        "            def probe_rear_after_seam(side, phase, probe_log=None):",
    )

    source = replace_once(
        source, '                    near = gap(center, rotation, 1, 1)\n                    if near < -.003:',
        '''                    near = gap(center, rotation, 1, 1)
                    if probe_log is not None:
                        camera = latest_camera_observation or {}
                        row = {
                            "physics_step": tick, "time_s": float(world.current_time), "arm": side,
                            "target_path": rear_target, "command_y_m": float(position[1]),
                            "measured_tcp_y_m": float(last_force_sample[side]["tcp_position_m"][1]),
                            "encoder_center_y_m": float(center[1]), "estimated_target_gap_m": float(near),
                            "axial_force_N": normal, "resultant_force_N": float(resultant),
                            "camera_frame_id": camera.get("frame_id"), "capture_time_s": camera.get("capture_time_s"),
                            "received_time_s": camera.get("received_time_s"),
                        }
                        probe_log.write(json.dumps(row, allow_nan=False) + "\\n")
                    if near < -.003:''',
    )
    source = replace_once(
        source, '                                    if sum(witness) < 24:\n'
        '                                        raise RuntimeError(f"{phase}: tactile stop lacks independent wall contact "',
        '                                    if probe_log is not None:\n'
        '                                        record(phase, "SENSOR_PROBE_CANDIDATE", arm=side, contact_target=rear_target,\n'
        '                                               estimated_gap_m=float(near), tactile_probe=stiffness)\n'
        '                                    if sum(witness) < 24:\n'
        '                                        raise RuntimeError(f"{phase}: tactile stop lacks independent wall contact "',
    )
    return replace_once(
        source, '''                if slot in (1, 2) and phase in ("upper_B_seam_rear_reseat", "upper_B_extra_seam_rear_reseat"):
                    probe_rear_after_seam(side, phase)''',
        '''                if slot == 5 and phase == "upper_B_rear" and target == rear_target:
                    contact_monitor.phase = "upper_B_rear_active_probe"
                    record(contact_monitor.phase, "PROBE_STARTED", arm=side, contact_target=target)
                    with (output_directory / "upper5_active_probe.jsonl").open("w") as probe_log:
                        probe_rear_after_seam(side, phase, probe_log=probe_log)
                    return
                if slot in (1, 2) and phase in ("upper_B_seam_rear_reseat", "upper_B_extra_seam_rear_reseat"):
                    probe_rear_after_seam(side, phase)''',
    )


def inject_terminal_resume(source):
    source = replace_once(source,
                          '            expected = RELEASE_CHECKPOINT["position_m"] if seam_resume else GUIDE["landing_m"]',
                          '            expected = RELEASE_CHECKPOINT["position_m"]')
    source = replace_once(
        source, '            nonlocal latched_contact_axis, contact_plan_rotation, carry_offsets',
        '            nonlocal latched_contact_axis, contact_plan_rotation, carry_offsets, control_center, control_rotation',
    )
    return replace_once(
        source, '            contact_plan_rotation = approach_front()',
        '''            terminal = RELEASE_CHECKPOINT["terminal_probe"]
            control_center, control_rotation = observe_resting_cube(
                "upper_B_terminal_pose", vision_camera, vision_estimator, stack_plane)
            contact_plan_rotation = control_rotation.copy()
            for arm in robots:
                q = np.asarray(terminal["q_rad"][arm])
                robots[arm].set_joint_positions(q, joint_indices=joints[arm])
                robots[arm].set_joint_velocities(np.zeros(7), joint_indices=joints[arm])
            set_contact_drive_mode(True)
            for arm in robots:
                command_positions(arm, np.asarray(terminal["q_target_rad"][arm]))
            allowed_contacts.add(support_side)
            push_faces[support_side] = (1, -control_rotation[:, 1], front_offset)
            contact_monitor.phase = "upper_B_terminal_resume"
            record(contact_monitor.phase, "ARCHIVED_TERMINAL_RESTORED", arm=support_side,
                   source_run=terminal["source_run"], source_step=terminal["source_step"])
            step(12)
            check_current_collision(contact_monitor.phase)''',
    )
