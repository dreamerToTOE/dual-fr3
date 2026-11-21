"""Single transverse command cycle at the U4 arm exchange or early in rear push."""
from textwrap import indent


def cycle():
    start = 0.
    for name, end, ticks in (("pre", 0., 12), ("out_neg", -.00025, 36), ("hold_neg", -.00025, 12),
                             ("cross_pos", .00025, 60), ("hold_pos", .00025, 12),
                             ("return", 0., 36), ("post", 0., 24)):
        for i in range(1, ticks + 1):
            t = i / ticks
            yield name, start + (end - start) * (10 - 15*t + 6*t*t) * t**3
        start = end


PROBE = """
def run_u4_side_probe():
    from experiments.u4_side_probe import cycle
    q = applied_targets["right"].copy()
    anchor, rotation = tcp_fk("right", q)
    previous_phase = contact_monitor.phase
    info = report["u4_side_probe"] = {"timing": U4_SIDE_PROBE, "arm": "right",
                                     "anchor_tcp_m": anchor.tolist(), "pre_step": tick,
                                     "amplitude_m": .00025}
    contact_monitor.phase = "upper_B_u4_side_probe"
    record(contact_monitor.phase, "STARTED", timing=U4_SIDE_PROBE)
    try:
        with (output_directory / "u4_side_probe.jsonl").open("x") as stream:
            for section, dx in cycle():
                target = anchor.copy()
                target[0] += dx
                q = cartesian_joint_target("right", q, target, rotation)
                center, attitude = sensed_cube()
                issues = checker.check({**applied_targets, "right": q}, cube_center=center, cube_rotation=attitude,
                                       cup_contacts=allowed_contacts, margin_m=0., first_only=True)
                if issues:
                    raise RuntimeError(f"U4 side probe command in collision: {issues[0]}")
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
                if np.linalg.norm(w[:3]) > 10. or np.linalg.norm(w[3:]) > .8:
                    raise RuntimeError(f"U4 side probe exceeded contact envelope at step {tick}: {w}")
        info["end_step"] = tick
        record(contact_monitor.phase, "RETURNED", timing=U4_SIDE_PROBE)
    finally:
        contact_monitor.phase = previous_phase
"""


def inject(source):
    from source_hooks import replace_once

    marker = "            def withdraw(side, axis, sign):"
    source = replace_once(source, marker, indent(PROBE.strip(), "            ") + "\n\n" + marker)
    marker = '                cartesian_move(side, position + .003 * outward, 1.5, phase + "_unload")'
    source = replace_once(source, marker, marker + '\n                if slot == 4 and axis == 0 and U4_SIDE_PROBE == "early":\n                    run_u4_side_probe()')
    marker = "                    for _ in range(travel_ticks):\n                        normal = force(side, 1, 1)"
    return replace_once(source, marker, '                    for rear_tick in range(travel_ticks):\n'
                        '                        if U4_SIDE_PROBE == "late" and rear_tick == 300:\n'
                        '                            run_u4_side_probe()\n'
                        '                        normal = force(side, 1, 1)')
