"""Single transverse command cycle at the U4 arm exchange or early in rear push."""
from textwrap import indent


def near_rear(window, target_y, half, command_y, measured_y, load):
    """Original U4 half-second command/force signature plus calibrated normal distance."""
    if load < .5:
        window.clear()
        return False
    window.append((command_y, measured_y, load))
    if len(window) < 61 or not -.002 <= target_y - measured_y - half <= .001:
        return False
    rows = list(window)
    command, motion, rise = (sum(row[i] for row in rows[-8:]) / 8 - sum(row[i] for row in rows[:8]) / 8
                             for i in range(3))
    return .000040 <= command <= .000130 and abs(motion) <= .35 * command and rise >= .30


def cycle():
    start = 0.
    for name, end, ticks in (("pre", 0., 12), ("out_neg", -.00025, 36), ("hold_neg", -.00025, 12),
                             ("cross_pos", .00025, 60), ("hold_pos", .00025, 12),
                             ("return", 0., 36), ("post", 0., 24)):
        for i in range(1, ticks + 1):
            t = i / ticks
            yield name, start + (end - start) * (10 - 15*t + 6*t*t) * t**3
        start = end


def detached_ready(elapsed_ticks, retreat_m, force_x_N):
    """Wait for the left cup to unload and recede using encoder and wrist feedback."""
    return elapsed_ticks >= 96 and retreat_m >= .00015 and abs(force_x_N) <= .12


PROBE = """
def run_u4_side_probe(detach_gate=None):
    from experiments.u4_side_probe import cycle
    q = applied_targets["right"].copy()
    anchor, rotation = tcp_fk("right", q)
    previous_phase = contact_monitor.phase
    amplitude = 0. if U4_SIDE_PROBE == "sham" else .00025
    info = report["u4_side_probe"] = {"timing": U4_SIDE_PROBE, "arm": "right",
                                     "anchor_tcp_m": anchor.tolist(), "pre_step": tick,
                                     "amplitude_m": amplitude}
    if detach_gate is not None:
        info["left_detach_gate"] = detach_gate
    contact_monitor.phase = "upper_B_u4_side_probe"
    record(contact_monitor.phase, "STARTED", timing=U4_SIDE_PROBE)
    try:
        with (output_directory / "u4_side_probe.jsonl").open("x") as stream:
            for section, dx in cycle():
                if detach_gate is None or section != "pre":
                    target = anchor.copy()
                    target[0] += dx * (amplitude / .00025)
                    q = cartesian_joint_target("right", q, target, rotation)
                    center, attitude = sensed_cube()
                    issues = checker.check({**applied_targets, "right": q}, cube_center=center, cube_rotation=attitude,
                                           cup_contacts=allowed_contacts, margin_m=0., first_only=True)
                    if issues:
                        raise RuntimeError(f"U4 side probe command in collision: {issues[0]}")
                    if detach_gate is not None and "first_motion_step" not in info:
                        info["first_motion_step"] = tick + 1
                        record(contact_monitor.phase, "MOTION_COMMAND", command_step=tick + 1)
                    command_positions("right", q)
                step()
                arm = last_force_sample["right"]
                w = arm["tcp_gravity_corrected_world"]
                actual_command, _ = tcp_fk("right", applied_targets["right"])
                row = {"physics_step": tick, "time_s": last_force_sample["time_s"], "segment": section,
                       "command_dx_m": float(actual_command[0]-anchor[0]), "command_x_m": float(actual_command[0]),
                       "encoder_x_m": arm["tcp_position_m"][0], "right_wrench_world": w}
                stream.write(json.dumps(row, allow_nan=False) + "\\n")
                check_current_collision(contact_monitor.phase)
                if np.linalg.norm(w[:3]) > force_ceiling or np.linalg.norm(w[3:]) > .8:
                    raise RuntimeError(f"U4 side probe exceeded contact envelope at step {tick}: {w}")
        info["end_step"] = tick
        record(contact_monitor.phase, "RETURNED", timing=U4_SIDE_PROBE)
    finally:
        contact_monitor.phase = previous_phase
"""


DETACHED_UNLOAD = """
from experiments.u4_side_probe import detached_ready

unload_start = tick
before_x = last_force_sample["left"]["tcp_position_m"][0]

def after_unload_step():
    if "u4_side_probe" in report:
        return
    left = last_force_sample["left"]
    retreat_m = before_x - left["tcp_position_m"][0]
    force_x_N = left["tcp_gravity_corrected_world"][0]
    if detached_ready(tick - unload_start, retreat_m, force_x_N):
        run_u4_side_probe({"step": tick, "left_retreat_m": retreat_m, "left_force_x_N": force_x_N})

cartesian_move(push_side, side_clear, 1.0, "upper_B_side_unload", after_step=after_unload_step)
if "u4_side_probe" not in report:
    record("upper_B_u4_side_probe", "DETACH_NOT_CONFIRMED")
"""


def inject(source, *, detached=False):
    from source_hooks import replace_once

    marker = "            def withdraw_both():"
    source = replace_once(source, marker, indent(PROBE.strip(), "            ") + "\n\n" + marker)
    unload = '                    cartesian_move(push_side, side_clear, 1.0, "upper_B_side_unload")'
    if detached:
        source = replace_once(source, unload, indent(DETACHED_UNLOAD.strip(), "                    "))
        source = replace_once(source, "def cartesian_move(side, destination, duration, phase, *, retarget_guide=False):",
                              "def cartesian_move(side, destination, duration, phase, *, retarget_guide=False, after_step=None):")
        marker = "                check_current_collision(phase)\n\n        def align_contact_plane"
        return replace_once(source, marker, "                check_current_collision(phase)\n"
                            "                if after_step is not None:\n"
                            "                    after_step()\n\n        def align_contact_plane")
    source = replace_once(source, unload, unload +
                          '\n                    if U4_SIDE_PROBE in ("early", "sham"):\n'
                          '                        run_u4_side_probe()')
    marker = "                    for _ in range(travel_ticks):\n                        normal = force(side, 1, 1)"
    return replace_once(source, marker, '''                    from experiments.u4_side_probe import near_rear
                    recent_right = deque(maxlen=61)
                    for rear_tick in range(travel_ticks):
                        sensor_trigger = False
                        if U4_SIDE_PROBE == "near" and "u4_side_probe" not in report:
                            command_y = tcp_fk(side, applied_targets[side])[0][1]
                            right = last_force_sample[side]
                            sensor_trigger = near_rear(recent_right, final[1], half, command_y,
                                                       right["tcp_position_m"][1], -right["tcp_gravity_corrected_world"][1])
                        fixed_tick = {"late": 300, "weak": 720}.get(U4_SIDE_PROBE)
                        if rear_tick == fixed_tick or sensor_trigger:
                            run_u4_side_probe()
                            # Resume the admittance state, but rebuild windows that require consecutive motion ticks.
                            confirmations.clear()
                            measured_window.clear()
                        normal = force(side, 1, 1)''')
