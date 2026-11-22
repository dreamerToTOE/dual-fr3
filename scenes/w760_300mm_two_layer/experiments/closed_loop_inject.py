"""Optional closed-loop supervisor injection for the generated PhysX runner.

The production runner remains an observer-only pipeline unless
``--closed-loop-mode`` is passed to :mod:`two_row_physx`. In that mode the
existing sensor boundary is reused, one selected observer stream drives a
small hold supervisor, and every decision is recorded beside the independent
PhysX reference. The hook deliberately does not alter nominal trajectories.
"""

from __future__ import annotations

try:
    from source_hooks import replace_once
except ImportError:  # pragma: no cover - direct module use in unit tests
    from scenes.w760_300mm.source_hooks import replace_once


def inject(source: str, mode: str) -> str:
    """Add the optional closed-loop observer/supervisor to generated source."""
    if mode is None:
        return source
    if mode not in ("vision", "tactile", "fusion"):
        raise ValueError(f"Unknown closed-loop mode {mode!r}")

    setup_marker = '        contact_experiment = ContactStateExperiment(output_directory / "contact_states", CONFIG)\n'
    setup = setup_marker + '''        # Optional supervisory path. The estimator consumes the same
        # causal sensor row as the baseline observer and never receives the
        # PhysX reference. PhysX is logged only after the decision.
        from experiments.closed_loop_supervisor import ClosedLoopSupervisor
        closed_loop_supervisor = ClosedLoopSupervisor(CLOSED_LOOP_MODE)
        closed_loop_mode = closed_loop_supervisor.estimator_mode
        closed_loop_hold_steps = {side: 0 for side in robots}
        report["closed_loop"] = {
            "enabled": True, "mode": CLOSED_LOOP_MODE,
            "estimator_mode": closed_loop_mode,
            "control_policy": "hold current joint targets after three consecutive states outside the phase target",
            "samples": [],
        }
'''
    source = replace_once(source, setup_marker, setup)

    left_hold_marker = '''                if left_hold is not None:
                    normal = -last_force_sample[support_side]["tcp_gravity_corrected_world"][1]
'''
    left_hold_replacement = '''                if left_hold is not None and not any(closed_loop_hold_steps.values()):
                    normal = -last_force_sample[support_side]["tcp_gravity_corrected_world"][1]
'''
    source = replace_once(source, left_hold_marker, left_hold_replacement)

    step_marker = '''                for side, robot in robots.items():
                    gravity = robot._articulation_view.get_generalized_gravity_forces(joint_indices=joints[side])[0]
'''
    step_replacement = '''                # A hold freezes the last accepted joint target for this
                # physics tick. The next observer sample can release it.
                for side, remaining in closed_loop_hold_steps.items():
                    if remaining > 0:
                        q_hold = applied_targets[side].copy()
                        robots[side].get_articulation_controller().apply_action(
                            ArticulationAction(joint_positions=q_hold,
                                               joint_velocities=np.zeros(7),
                                               joint_indices=joints[side]))
                        closed_loop_hold_steps[side] = remaining - 1
                for side, robot in robots.items():
                    gravity = robot._articulation_view.get_generalized_gravity_forces(joint_indices=joints[side])[0]
'''
    source = replace_once(source, step_marker, step_replacement)

    command_marker = '''        def command_positions(side, q, nominal=True, velocity=None):
            q = np.asarray(q, dtype=float)
            if nominal:
'''
    command_replacement = '''        def command_positions(side, q, nominal=True, velocity=None):
            q = np.asarray(q, dtype=float)
            if closed_loop_hold_steps.get(side, 0) > 0:
                q = applied_targets[side].copy()
            if nominal:
'''
    source = replace_once(source, command_marker, command_replacement)

    recognized_marker = '                recognized = contact_experiment.append('
    if source.count(recognized_marker) != 1:
        raise RuntimeError("Expected one ContactStateExperiment append hook")
    append_end = source.index('                step_cost["recognition_and_reference"]',
                              source.index(recognized_marker))
    decision_block = '''                selected_prediction = contact_experiment.predictions[closed_loop_mode][-1]
                closed_loop_decision = closed_loop_supervisor.update(
                    selected_prediction["state"], contact_monitor.phase)
                if closed_loop_decision["action"] == "hold":
                    for hold_side in closed_loop_decision["hold_sides"]:
                        closed_loop_hold_steps[hold_side] = 1
                closed_loop_truth_state = contact_experiment.references[-1]
                report["closed_loop"]["samples"].append({
                    "physics_step": tick,
                    "time_s": float(world.current_time),
                    "phase": contact_monitor.phase,
                    "predicted_state": selected_prediction["state"],
                    "predicted_raw_state": selected_prediction.get("raw_state"),
                    "target_state": closed_loop_decision["target_states"],
                    "target_states": closed_loop_decision["target_states"],
                    "supervisor_action": closed_loop_decision["action"],
                    "reason": closed_loop_decision["reason"],
                    "physx_state": closed_loop_truth_state["state"],
                    "physx_tool_contact": closed_loop_truth_state["tool_contact"],
                    "physx_supported": closed_loop_truth_state["supported"],
                })
'''
    source = source[:append_end] + decision_block + source[append_end:]
    return source
