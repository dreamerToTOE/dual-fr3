"""Inject a post-rear-confirmation physical side-loss perturbation for U4."""

from textwrap import indent


PROBE = r'''
U4_SILENT_LOSS_PROBE_ACTIVE = False

def run_u4_silent_loss_probe():
    """Separate the active box from Cube 8 while preserving rear support."""
    import numpy as np
    global U4_SILENT_LOSS_PROBE_ACTIVE
    import u4_contact_observer

    probe_force_N = 0.0
    probe_force_steps = 0
    previous_phase = contact_monitor.phase
    info = report["u4_silent_loss_probe"] = {
        "start_step": tick,
        "arm": "left",
        "external_force_world_N": [probe_force_N, 0.0, 0.0],
        "duration_steps": 60,
        "force_steps": probe_force_steps,
        "right_arm_command_held": True,
        "physical_perturbation": "cube_pose_x_minus_6mm",
    }
    contact_monitor.phase = "upper_B_u4_silent_loss_probe"
    U4_SILENT_LOSS_PROBE_ACTIVE = True
    u4_contact_observer.U4_SILENT_LOSS_PROBE_ACTIVE = True
    left_q = applied_targets["left"].copy()
    left_tcp, left_rotation = tcp_fk("left", left_q)
    left_target = left_tcp.copy()
    left_target[0] -= .0040
    left_goal = cartesian_joint_target("left", left_q, left_target, left_rotation)
    try:
        record(contact_monitor.phase, "STARTED", cube_id=current_task["cube_id"], source="post_rear_force_confirmed")
        center, orientation = cube.get_world_pose()
        center = np.asarray(center, dtype=np.float32)
        center[0] -= .006
        cube.set_world_pose(center, orientation)
        cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
        cube.set_angular_velocity(np.zeros(3, dtype=np.float32))
        for index in range(60):
            alpha = (index + 1) / 60.
            smooth = alpha * alpha * (3. - 2. * alpha)
            left_command = left_q + smooth * (left_goal - left_q)
            command_positions("left", left_command)
            command_positions("right", applied_targets["right"])
            step()
            check_current_collision(contact_monitor.phase)
        cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
        center, _ = cube.get_world_pose()
        info.update({"end_step": tick, "cube_position_m": np.asarray(center).tolist()})
        record(contact_monitor.phase, "COMPLETED", **info)
        if STOP_AFTER_U4_SILENT_LOSS:
            raise RuntimeError("U4 silent-loss probe completed by request")
    finally:
        U4_SILENT_LOSS_PROBE_ACTIVE = False
        u4_contact_observer.U4_SILENT_LOSS_PROBE_ACTIVE = False
        contact_monitor.phase = previous_phase
'''


def inject(source):
    from source_hooks import replace_once

    # The probe is nested in execute_upper_push, while the observer is a
    # separate imported module. Define the latch in the generated runner and
    # mirror it into the observer module only during this experiment.
    source = source.replace(
        'from __future__ import annotations\n',
        'from __future__ import annotations\n\nU4_SILENT_LOSS_PROBE_ACTIVE = False\n',
        1,
    )

    source = replace_once(
        source,
        '            def push_u4_to_rear(side, phase, target):\n',
        indent(PROBE.strip() + "\n\n", "            ")
        + '            def push_u4_to_rear(side, phase, target):\n',
    )
    marker = '''                                record(phase, "FORCE_CONFIRMED", arm=side, contact_target=target,
                                       held_samples=len(confirmations), normal_force_N=float(loads.mean()),
                                       normal_force_samples_N=loads.tolist(), estimated_gap_m=float(near),
                                       witnessed_target_contact_ticks=int(np.count_nonzero(witness)),
                                       sliding_baseline_N=resistance, force_threshold_N=threshold)
                                return'''
    replacement = marker.replace(
        "                                return",
        "                                if U4_SILENT_LOSS_PROBE:\n"
        "                                    run_u4_silent_loss_probe()\n"
        "                                return",
    )
    return replace_once(source, marker, replacement)
