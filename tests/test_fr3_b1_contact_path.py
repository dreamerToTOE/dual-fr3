"""Replay the B1 touch-to-push handoff without an Isaac Sim instance."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_b1_handoff_keeps_touch_command_and_rejects_rgb_normal_discontinuity():
    source = Path(__file__).resolve().parents[1] / "fr3_ab_visual_runner.py"
    tree = ast.parse(source.read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                 and node.name in ("sample_vision", "contact_path")]
    for function in functions:
        function.body = [ast.copy_location(ast.Global(names=node.names), node)
                         if isinstance(node, ast.Nonlocal) else node for node in function.body]

    anchor = np.array([.1475, .73425, .787])
    rgb = np.array([.14673, .73153, .787])
    touch_tcp = np.array([.14619, .65925, .787])
    rotation = np.eye(3)
    observation = {"status": "detected", "cube_position_m": rgb, "pixel_center": [0., 0.],
                   "cube_rotation_world": rotation, "reprojection_error_px": .18}
    frame = {"rendering_frame": 1, "rendering_time": 0., "rgba": np.zeros((1, 1, 4))}
    env = {
        "np": np, "math": math, "push_faces": {"right": (1, [0., -1., 0.], [0., -.075, 0.])},
        "latched_contact_axis": None, "control_center": anchor.copy(), "contact_plan_rotation": rotation,
        "world": SimpleNamespace(current_time=0.), "contact_monitor": SimpleNamespace(phase="B1"),
        "vision_camera": SimpleNamespace(get_current_frame=lambda **kwargs: frame),
        "vision_cfg": {"resolution": [1, 1]}, "seen_vision_frames": set(), "truth_history": [],
        "CONFIG": {"floor_top_m": .712}, "match_reference": lambda *args: None, "tick": 0,
        "vision_estimator": SimpleNamespace(estimate=lambda rgba: observation,
                                            resting_pose=lambda obs, height: (rgb, rotation)),
        "supported_pose_filter": SimpleNamespace(update=lambda *args, **kwargs: (rgb, rotation, 0.)),
        "last_force_sample": {"right": {"tcp_gravity_corrected_world": [0., -.001, 0., 0., 0., 0.]}},
        "vision_comparison": SimpleNamespace(append=lambda **kwargs: {
            "physics_step": None, "truth_position_m": None, "position_error_m": None}),
        "report": {"vision": dict(unique_frames=0, samples=0, detections=0, detection_log=[])},
        "joint_state": lambda: {"right": touch_tcp.copy()}, "applied_targets": {"right": touch_tcp},
        "tcp_fk": lambda side, q: (q.copy(), rotation),
        "cartesian_joint_target": lambda side, seed, target, orientation: target,
        "planning_update": lambda phase: None,
    }

    def execute(samples, phase, *, check_step, cube_center_path, sampled_trajectory):
        np.testing.assert_array_equal(samples["right"][0], touch_tcp)
        np.testing.assert_allclose(samples["right"][-1], touch_tcp + [0., .34, 0.], atol=1e-12)
        env["sample_vision"](render=False)
        np.testing.assert_array_equal(env["control_center"], [rgb[0], anchor[1], rgb[2]])

    env["execute_joint_path"] = execute
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), env)
    env["contact_path"]("right", anchor, anchor + [0., .34, 0.], "B1")
    frame["rendering_frame"] += 1
    env["sample_vision"](render=False)
    np.testing.assert_array_equal(env["control_center"], rgb)

    # A retreat image implying a 50-degree tilt cannot update the supported pose.
    tilt, yaw = np.radians([50., 4.])
    c, s = np.cos(tilt), np.sin(tilt)
    tilted = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    c, s = np.cos(yaw), np.sin(yaw)
    turned = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    env["supported_pose_filter"].update = lambda *args, **kwargs: (rgb, turned, 0.)
    observation["cube_rotation_world"] = tilted
    frame["rendering_frame"] += 1
    env["sample_vision"](render=False)
    camera = env["latest_camera_observation"]
    assert camera["control_pose_rejection"] == "marker_plane_tilt"
    np.testing.assert_array_equal(camera["cube_rotation_world"], tilted)
    np.testing.assert_array_equal(env["control_rotation"], rotation)

    observation["cube_rotation_world"] = turned
    frame["rendering_frame"] += 1
    with pytest.raises(RuntimeError, match="more than 3 degrees"):
        env["sample_vision"](render=False)
