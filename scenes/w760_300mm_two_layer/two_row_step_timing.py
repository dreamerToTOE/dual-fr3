"""Keep the shared physics clock and reuse the wrist recorder's same-step encoder FK."""
from source_hooks import replace_once


def control_deadline(previous, now):
    dt = 1 / 120
    if now - previous > .10:
        return now
    return max(previous + dt, now - 3 * dt)


def inject(source):
    source = replace_once(source, '        last_control_wall = time.monotonic()\n        last_render_wall = last_control_wall', '''        from two_row_step_timing import control_deadline
        step_cost = report["execution_wall_s"] = {"pace_sleep": 0., "world_physics": 0., "all_render": 0.,
                                                  "step_render": 0., "step_total": 0., "wrist_sensor": 0.,
                                                  "recognition_and_reference": 0.}
        last_control_wall = time.monotonic()
        last_render_wall = last_control_wall''')
    source = replace_once(source, '''                if not args.headless:
                    time.sleep(max(0.0, 1 / 120 - (time.monotonic() - last_control_wall)))
                last_control_wall = time.monotonic()''', '''                if not args.headless:
                    now = time.monotonic()
                    last_control_wall = control_deadline(last_control_wall, now)
                    if last_control_wall > now:
                        time.sleep(last_control_wall - now)
                        step_cost["pace_sleep"] += time.monotonic() - now''')
    source = replace_once(source, '''            world.render()
            extra_steps''', '''            render_started = time.monotonic()
            world.render()
            step_cost["all_render"] += time.monotonic() - render_started
            extra_steps''')
    source = replace_once(source, '''                world.step(render=False)
                grippers.update()''', '''                phys_started = time.monotonic()
                world.step(render=False)
                step_cost["world_physics"] += time.monotonic() - phys_started
                grippers.update()''')
    source = replace_once(source, '        def step(count=1):\n            nonlocal tick,',
                          '        def step(count=1):\n            step_started = time.monotonic()\n'
                          '            render_before = step_cost["all_render"]\n            nonlocal tick,')
    source = replace_once(source, '        def capture_acceptance_view(event):',
                          '            step_cost["step_render"] += step_cost["all_render"] - render_before\n'
                          '            step_cost["step_total"] += time.monotonic() - step_started\n\n'
                          '        def capture_acceptance_view(event):')
    source = replace_once(source, '                force_sample = force_recorder.sample(',
                          '                wrist_started = time.monotonic()\n                force_sample = force_recorder.sample(')
    source = replace_once(source, '                last_force_sample = force_sample',
                          '                step_cost["wrist_sensor"] += time.monotonic() - wrist_started\n'
                          '                last_force_sample = force_sample')
    source = replace_once(source, '                recognized = contact_experiment.append(',
                          '                state_started = time.monotonic()\n                recognized = contact_experiment.append(')
    source = replace_once(source, '                if contact_monitor.phase.startswith("upper_B_"):',
                          '                step_cost["recognition_and_reference"] += time.monotonic() - state_started\n'
                          '                if contact_monitor.phase.startswith("upper_B_"):')
    source = replace_once(source, '''                measured = joint_state()
                sensor_poses = {side: tcp_fk(side, measured[side]) for side in robots}''', '''                # Wrist samples already carry these same encoder-based poses from this physics step.
                sensor_rotations = {side: quat_to_rot_matrix(np.asarray(force_sample[side]["sensor_orientation_wxyz"]))
                                    for side in robots}''')
    source = replace_once(source, '"tcp_positions_m": {side: pose[0].tolist() for side, pose in sensor_poses.items()},',
                          '"tcp_positions_m": {side: force_sample[side]["tcp_position_m"] for side in robots},')
    return replace_once(source, '''{side: (pose[1] @ np.array([
                            0., -1. if side == "left" else 1., 0.])).tolist()
                            for side, pose in sensor_poses.items()}''', '''{side: ((-1. if side == "left" else 1.) * rot[:, 1]).tolist()
                                              for side, rot in sensor_rotations.items()}''')
