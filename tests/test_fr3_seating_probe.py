"""Exercise the production nested seat controller without starting Isaac Sim."""
import ast
from pathlib import Path

import numpy as np
import pytest

from fr3_contact_control import NormalForceController, oriented_wall_gap


def run_probe(forces, gaps, contact_target="Cube_1", *, follow_side="left", probe_override=None):
    tree = ast.parse((Path(__file__).resolve().parents[1] / "fr3_ab_visual_runner.py").read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "seat_against_wall")
    parameters = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "B2_CONTROLLER_PARAMETERS"
                              for target in node.targets))
    index, events, commands = 0, [], []
    env = {"np": np, "half": .075, "B2_CONTROLLER_PARAMETERS": parameters,
           "oriented_wall_gap": oriented_wall_gap,
           "applied_targets": {"left": np.zeros(3), "right": np.zeros(3)},
           "tcp_fk": lambda side, q: (np.zeros(3), np.eye(3)),
           "cartesian_joint_target": lambda side, seed, target, rotation: target.copy(),
           "command_positions": lambda side, q: commands.append((side, q.copy())),
           "record": lambda phase, status, **values: events.append((status, values)),
           "check_current_collision": lambda phase: None}

    def sample():
        env["last_force_sample"] = {"right": {"tcp_gravity_corrected_world": forces[index] + [0.] * 3}}

    def step():
        nonlocal index
        index += 1
        if index >= len(forces):
            raise RuntimeError("fixture exhausted before contact confirmation")
        sample()

    env["step"] = step
    env["sensed_cube"] = lambda: (np.array([gaps[index], 0., 0.]), np.eye(3))
    sample()
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<production seat controller>", "exec"), env)
    env["seat_against_wall"]("right", 0, -1, 0., "B2_push_seat",
                             contact_target=contact_target, follow_side=follow_side, probe_override=probe_override)
    return events, [q[0] for side, q in commands if side == "right"]


def test_original_cube3_false_neighbor_contact_is_rejected():
    # The old controller accepted these wrist samples with no Cube3–Cube1 impulse.
    forces = [[value, 0., 0.] for value in [6.399179, 2.169482, 4.403542]]
    with pytest.raises(RuntimeError, match="fixture exhausted"):
        run_probe(forces, [.0001] * len(forces))


def test_probe_requires_8_n_near_target_then_30_paused_ticks_at_or_above_6_n():
    paused_forces = [6., 7.9] * 15
    forces = [[value, 0., 0.] for value in [9., 7.999, 8., *paused_forces]]
    events, commands = run_probe(forces, [.006] + [.0001] * (len(forces) - 1))
    assert commands[1] < commands[0] < 0.
    assert commands[2:] == [commands[1]] * 30
    status, confirmation = events[-1]
    assert status == "FORCE_CONFIRMED"
    assert confirmation["contact_target"] == "Cube_1"
    assert confirmation["probe_entry_force_N"] == 8.
    assert confirmation["probe_minimum_force_N"] == 6.
    assert confirmation["probe_normal_samples_N"] == paused_forces


def test_single_subthreshold_tick_resumes_creep_and_resets_probe():
    paused_forces = [6.1] * 30
    forces = [[value, 0., 0.] for value in [8., 7.5, 5.999, 7.999, 8., *paused_forces]]
    events, commands = run_probe(forces, [.0001] * len(forces))
    assert commands[:2] == [0., 0.]
    assert commands[2] < commands[1]
    assert commands[3] < commands[2]
    assert commands[4:] == [commands[3]] * 30
    assert events[-1][1]["probe_normal_samples_N"] == paused_forces


def test_surveyed_wall_confirmed_at_lower_force_before_overload():
    # A fresh physical wall impulse preceded the 6.8424 N entry in the Cube1 trace.
    normals = [5.9423, 6.8424, 4.7470, 7.1585]
    forces = [[value, 0., 0.] for value in normals]
    events, commands = run_probe(forces, [.00014] * len(forces), contact_target="wall_left")
    assert commands == [commands[0]] * 3 and commands[0] < 0.
    status, confirmation = events[-1]
    assert status == "FORCE_CONFIRMED"
    assert confirmation["contact_target"] == "wall_left"
    assert confirmation["probe_entry_force_N"] == 6.
    assert confirmation["probe_mean_force_N"] == 4.
    assert confirmation["probe_normal_samples_N"] == normals[1:]


def test_surveyed_wall_resumes_advance_when_mean_is_below_4_n():
    normals = [6., 2.5, 3., 6., 3., 3.]
    forces = [[value, 0., 0.] for value in normals]
    events, commands = run_probe(forces, [.00014] * len(forces), contact_target="wall_right")
    assert commands[:2] == [0., 0.]
    assert commands[2] < 0.
    assert commands[3:] == [commands[2]] * 2
    assert events[-1][1]["probe_normal_samples_N"] == normals[3:]


@pytest.mark.parametrize("force", ([10.01, 0., 0.], [8., 7., 0.], [-10.01, 0., 0.]))
def test_raw_axis_and_resultant_overload_abort_before_confirmation(force):
    with pytest.raises(RuntimeError, match="wall acquisition overload"):
        run_probe([force], [.0001])


def test_reused_support_hold_switches_to_release_gain_before_unloading():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "fr3_ab_visual_runner.py").read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "unload_normal")
    parameters = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "B2_CONTROLLER_PARAMETERS"
                              for target in node.targets))
    controller = NormalForceController(1., gain_m_per_Ns=parameters["gain_m_per_Ns"],
                                       max_speed_m_s=.0002)
    force = {"left": {"tcp_gravity_corrected_world": [0., -2., 0., 0., 0., 0.]}}
    samples = []

    def step():
        command = controller.update(-force["left"]["tcp_gravity_corrected_world"][1])
        # The first 0.4 mm removes compliant servo command deflection, as in the
        # recorded low-stiffness unload; only later travel reduces contact load.
        outward = 1. - command
        compression = max(0., 2. - 20000 * max(0., outward - .0004))
        force["left"]["tcp_gravity_corrected_world"][1] = -compression
        samples.append(compression)

    env = {"np": np, "NormalForceController": NormalForceController,
           "B2_CONTROLLER_PARAMETERS": parameters, "left_hold": controller, "support_side": "left",
           "last_force_sample": force, "force_gain": parameters["gain_m_per_Ns"],
           "applied_targets": {"left": np.zeros(7)},
           "tcp_fk": lambda side, q: (np.array([0., 1., 0.]), np.eye(3)),
           "record": lambda *args, **kwargs: None, "step": step,
           "check_current_collision": lambda phase: None}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<production unload controller>", "exec"), env)
    env["unload_normal"]("left", 1, 1, "B2_support_unload")
    assert controller.gain_m_per_Ns == parameters["unload_gain_m_per_Ns"] == 1e-4
    assert parameters["gain_m_per_Ns"] == 2e-5
    assert len(samples) < 1800
    assert max(samples[-30:]) < .1


def test_cube4_final_wall_probe_rejects_sliding_force_and_holds_for_confirmation():
    # ~0.8 N of sliding resistance previously ended the push 1.26 mm short.
    probe = {"entry_force_N": 5., "mean_force_N": 3., "samples": 30,
             "target_uncertainty_m": .003}
    forces = [[f, 0., 0.] for f in [.8] * 10 + [5.] + [3.5] * 29]
    events, commands = run_probe(forces, [.0001] * len(forces), "wall_back",
                                 follow_side=None, probe_override=probe)
    assert commands[1] < commands[0]
    assert commands[-29:] == [commands[-1]] * 29
    status, confirmation = events[-1]
    assert status == "FORCE_CONFIRMED"
    assert confirmation["contact_target"] == "wall_back"
    assert confirmation["probe_normal_samples_N"] == [5.] + [3.5] * 29


def test_staged_side_requires_sustained_entry_frozen_load_and_small_measured_travel():
    # The sliding P90 is 1.15 N on the measured W760 route. Reject brief spikes
    # and verify the full paused window, including samples between 1.5 and 2 N.
    probe = {"entry_force_N": 2., "entry_samples": 3, "minimum_force_N": 1.5,
             "mean_force_N": 1.8, "max_axial_travel_m": .0005,
             "target_uncertainty_m": .003, "samples": 30}
    ripple = [1.6, 2.2] * 15
    normals = [2.1, 1., 2., 2., 2., *ripple]
    forces = [[f, 0., 0.] for f in normals]
    events, commands = run_probe(forces, [.0002] * len(forces), "Cube_2", probe_override=probe)
    assert commands[-30:] == [commands[-1]] * 30
    assert commands[1] < commands[0]
    assert events[-1][1]["probe_normal_samples_N"] == ripple
    assert events[-1][1]["measured_axial_travel_m"] == 0.
    for rejected, gaps in (
            ([2., 1.] * 20, [.0002] * 40),
            ([2.] * 3 + [1.6] * 30, [.0002] * 33),
            (normals, [.0002] * 6 + [.0009] * (len(forces) - 6))):
        with pytest.raises(RuntimeError, match="fixture exhausted"):
            run_probe([[f, 0., 0.] for f in rejected], gaps, "Cube_2", probe_override=probe)
