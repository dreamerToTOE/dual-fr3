"""Run the production touch-seeking helper with deterministic force feedback."""
import ast
import math
from pathlib import Path

import numpy as np
import pytest


def touch_trial(forces, distance=.000020, collision=False):
    tree = ast.parse((Path(__file__).resolve().parents[1] / "fr3_ab_visual_runner.py").read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "touch_face")
    parameters = next(ast.literal_eval(node.value) for node in tree.body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "B2_CONTROLLER_PARAMETERS"
                              for target in node.targets))
    positions, collisions, observations = [], [], []
    readings = iter(forces)
    env = {"np": np, "math": math, "B2_CONTROLLER_PARAMETERS": parameters,
           "applied_targets": {"right": np.zeros(3)},
           "tcp_fk": lambda side, q: (np.zeros(3), np.eye(3)),
           "cartesian_joint_target": lambda side, seed, target, rotation: target.copy(),
           "command_positions": lambda side, q: positions.append(float(q[0])),
           "record": lambda *args, **kwargs: None,
           "report": {"target_observations": observations}}

    def step():
        compression = next(readings)
        env["last_force_sample"] = {
            "right": {"tcp_gravity_corrected_world": [-compression, 0., 0., 0., 0., 0.]}}

    def check(phase, contact_uncertainty_m):
        collisions.append(contact_uncertainty_m)
        if collision:
            raise RuntimeError("physical collision")

    env.update(step=step, check_current_collision=check)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<production touch controller>", "exec"), env)
    return (lambda: env["touch_face"]("right", np.array([1., 0., 0.]), distance, "B2_touch"),
            positions, collisions, observations, parameters)


def test_force_detection_freezes_position_for_all_six_confirmation_samples():
    execute, positions, checks, observations, parameters = touch_trial(
        [0., .25, .30, .28, .31, .35, .29])
    execute()
    np.testing.assert_allclose(positions, [0.] + [5e-6] * 6, atol=1e-15)
    assert len(checks) == len(positions)
    assert all(value == .005 for value in checks)
    assert observations[-1]["search_m"] == pytest.approx(5e-6)
    assert parameters["touch_confirmation_frames"] == 6
    assert parameters["touch_contact_force_N"] == .20


def test_lost_candidate_resumes_at_next_increment_without_schedule_jump():
    execute, positions, _, observations, _ = touch_trial(
        [0., .25, .30, .15, .25, .30, .31, .32, .33, .34])
    execute()
    np.testing.assert_allclose(positions, [0., 5e-6, 5e-6, 5e-6] + [10e-6] * 6, atol=1e-15)
    assert observations[-1]["search_m"] == pytest.approx(10e-6)
    assert max(np.diff(positions)) <= 5e-6 + 1e-15


def test_failed_seek_is_bounded_and_never_exceeds_authorized_distance():
    execute, positions, checks, _, _ = touch_trial([0.] * 100)
    with pytest.raises(RuntimeError, match="did not confirm contact"):
        execute()
    np.testing.assert_allclose(positions, np.arange(5) * 5e-6, atol=1e-15)
    assert len(checks) == 5
    assert max(positions) <= 20e-6 + 1e-15


def test_confirmation_at_final_seek_point_can_finish_without_more_motion():
    execute, positions, _, _, _ = touch_trial([0.] * 4 + [.25] * 6)
    execute()
    np.testing.assert_allclose(positions, [0., 5e-6, 10e-6, 15e-6] + [20e-6] * 6, atol=1e-15)


def test_seek_includes_exact_endpoint_when_distance_is_not_a_step_multiple():
    execute, positions, _, _, _ = touch_trial([0.] * 100, distance=22e-6)
    with pytest.raises(RuntimeError, match="did not confirm contact"):
        execute()
    np.testing.assert_allclose(positions, [0., 5e-6, 10e-6, 15e-6, 20e-6, 22e-6], atol=1e-15)


@pytest.mark.parametrize("collision", [False, True])
def test_force_and_collision_protection_abort_the_search(collision):
    execute, positions, _, observations, _ = touch_trial([10.1], collision=collision)
    with pytest.raises(RuntimeError, match="physical collision" if collision else "unexpected contact force"):
        execute()
    assert positions == [0.]
    assert not observations
