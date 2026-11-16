"""Check fixed-base A/B and a proposed rail-assisted B2 plan with FR3 Lula.

This does not execute trajectories or check collisions, suction, or contact.
Fixed-base failures remain in the report even if the alternate B2 plan passes.
Exit 2 means no complete kinematic plan passed; python.sh maps this to 1.
The complete report is retained before Kit shuts down.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "validation" / "fr3_ab_ik_preflight.json"
POSITION_TOLERANCE_M = 0.001
ORIENTATION_TOLERANCE_RAD = 0.01
SIDE_ROTATION = [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
FRONT_ROTATION = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
B2_RAIL_ROTATION = SIDE_ROTATION


def waypoints(cfg):
    half = cfg["cube_size_m"] / 2
    task = cfg["task"]
    pickup_path = [
        ("A_supply", cfg["supply_cube_center_m"]),
        ("A_lift", task["phase_A"]["lift_center_m"]),
        ("A_over_front", task["phase_A"]["over_front_center_m"]),
        ("A_landing", task["phase_A"]["landing_center_m"]),
    ]
    for side, sign in (("left", -1), ("right", 1)):
        for name, center in pickup_path:
            yield side, name, [center[0] + sign * half, center[1], center[2]], SIDE_ROTATION
    for name, center in (
        ("B1_start", task["phase_A"]["landing_center_m"]),
        ("B1_back_wall", task["phase_B1"]["target_center_m"]),
    ):
        yield "left", name, [center[0], center[1] - half, center[2]], FRONT_ROTATION
    for name, center in (
        ("B2_start", task["phase_B1"]["target_center_m"]),
        ("B2_left_wall", task["phase_B2"]["target_center_m"]),
    ):
        yield "right", name, [center[0] + half, center[1], center[2]], SIDE_ROTATION


def check_waypoints(cfg):
    import numpy as np

    points = list(waypoints(cfg))
    assert len(points) == 12
    half = cfg["cube_size_m"] / 2
    assert np.allclose(np.array(points[4][2]) - points[0][2], [2 * half, 0, 0])
    for side, _, _, rotation in points:
        rotation = np.array(rotation)
        assert np.allclose(rotation.T @ rotation, np.eye(3))
        assert np.isclose(np.linalg.det(rotation), 1.0)
        # Each cup axis must point toward the contacted cube face.
        local_axis = np.array([0, -1 if side == "left" else 1, 0])
        expected = [0, 1, 0] if np.array_equal(rotation, FRONT_ROTATION) else [1 if side == "left" else -1, 0, 0]
        assert np.allclose(rotation @ local_axis, expected)
    return points


def solve_waypoint(cfg, solver, side, name, tcp, rotation, seed):
    import numpy as np
    from isaacsim.core.utils.rotations import rot_matrix_to_quat

    rotation = np.array(rotation)
    tool = cfg["side_suction_tool"]
    offset = np.array([0, (-1 if side == "left" else 1) * tool["tcp_side_offset_m"], tool["vertical_drop_m"]])
    flange = np.array(tcp) - rotation @ offset
    orientation = rot_matrix_to_quat(rotation)
    q, solver_success = solver.compute_inverse_kinematics(
        tool["mount_link"], flange, orientation, warm_start=seed,
        # Flange pose errors also move the offset TCP; leave room for that error.
        position_tolerance=POSITION_TOLERANCE_M / 10, orientation_tolerance=ORIENTATION_TOLERANCE_RAD / 10,
    )
    actual_flange, actual_rotation = solver.compute_forward_kinematics(tool["mount_link"], q)
    actual_tcp = np.asarray(actual_flange).reshape(3) + actual_rotation @ offset
    tcp_error = float(np.linalg.norm(actual_tcp - tcp))
    angle_error = float(np.arccos(np.clip((np.trace(rotation.T @ actual_rotation) - 1) / 2, -1, 1)))
    success = bool(solver_success and tcp_error <= POSITION_TOLERANCE_M and angle_error <= ORIENTATION_TOLERANCE_RAD)
    return {
        "arm": side, "waypoint": name, "success": success, "solver_success": bool(solver_success),
        "tcp_target_m": tcp, "flange_target_m": flange.tolist(), "flange_orientation_wxyz": orientation.tolist(),
        "joint_positions_rad": q.tolist(), "tcp_actual_m": actual_tcp.tolist(),
        "tcp_error_m": tcp_error, "orientation_error_rad": angle_error,
    }


def run(cfg):
    import numpy as np
    import omni.usd
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, interface_config_loader
    from pxr import UsdGeom

    UsdGeom.SetStageMetersPerUnit(omni.usd.get_context().get_stage(), cfg["meters_per_unit"])
    solvers = {}
    seeds = {}
    for side, base in cfg["fr3"]["bases"].items():
        solver = LulaKinematicsSolver(**interface_config_loader.load_supported_lula_kinematics_solver_config("FR3"))
        solver.sampling_seed = 0
        solver.set_robot_base_pose(np.array(base[:3]), np.array([math.cos(base[3] / 2), 0, 0, math.sin(base[3] / 2)]))
        assert solver.get_joint_names() == [f"fr3_joint{i}" for i in range(1, 8)]
        assert cfg["side_suction_tool"]["mount_link"] in solver.get_all_frame_names()
        solvers[side] = solver
        seeds[side] = np.array(cfg["fr3"]["official_start_q_rad"])

    results = []
    for side, name, tcp, rotation in check_waypoints(cfg):
        result = solve_waypoint(cfg, solvers[side], side, name, tcp, rotation, seeds[side])
        if result["success"]:
            seeds[side] = np.array(result["joint_positions_rad"])
        results.append(result)
    blocked = [f'{r["arm"]}:{r["waypoint"]}' for r in results if not r["success"]]

    rail_cfg = cfg["task"]["phase_B2"]["rail_assist"]
    rail_base = list(cfg["fr3"]["bases"]["right"])
    rail_base[1] = float(rail_cfg["base_y_m"])
    assert cfg["rails"]["axis"] == "Y"
    low, high = cfg["rails"]["travel_limits_m"]
    assert low <= rail_base[1] <= high, "B2 rail base Y exceeds travel limits"
    rotation = np.array(B2_RAIL_ROTATION)
    assert np.allclose(rotation.T @ rotation, np.eye(3)) and np.isclose(np.linalg.det(rotation), 1)
    assert np.allclose(rotation @ [0, 1, 0], [-1, 0, 0]), "right cup must face -X"
    rail_solver = LulaKinematicsSolver(**interface_config_loader.load_supported_lula_kinematics_solver_config("FR3"))
    rail_solver.set_robot_base_pose(
        np.array(rail_base[:3]), np.array([math.cos(rail_base[3] / 2), 0, 0, math.sin(rail_base[3] / 2)])
    )
    rail_results = []
    seed = np.array(cfg["fr3"]["official_start_q_rad"])
    start, end = [r["tcp_target_m"] for r in results[-2:]]
    expected_quat = np.asarray(rail_cfg["tcp_orientation_wxyz"], dtype=float)
    from isaacsim.core.utils.rotations import rot_matrix_to_quat
    actual_quat = rot_matrix_to_quat(rotation)
    assert np.allclose(actual_quat, expected_quat, atol=1e-9) or np.allclose(actual_quat, -expected_quat, atol=1e-9)
    for index, tcp in enumerate(np.linspace(start, end, int(rail_cfg["path_samples"]))):
        result = solve_waypoint(cfg, rail_solver, "right", f"B2_rail_{index}", tcp.tolist(), rotation, seed)
        rail_results.append(result)
        if result["success"]:
            seed = np.array(result["joint_positions_rad"])
    rail_pass = all(r["success"] for r in rail_results)
    joint_steps = np.diff([r["joint_positions_rad"] for r in rail_results], axis=0)
    complete_plan = all(r["success"] for r in results[:-2]) and rail_pass
    return {
        "scene_id": cfg["scene_id"], "scope": "fixed_base_and_proposed_B2_rail_waypoint_ik_only",
        "status": "PASS_KINEMATIC_RAIL_PLAN" if complete_plan else "BLOCKED", "ab_status": "BLOCKED_DYNAMIC",
        "claim": "IK success does not establish collision-free trajectories or A/B dynamics.",
        "base_poses_xyz_yaw": cfg["fr3"]["bases"], "rail_motion_executed": False,
        "frame": cfg["side_suction_tool"]["mount_link"], "joint_names": solvers["left"].get_joint_names(),
        "position_tolerance_m": POSITION_TOLERANCE_M, "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
        "solver_position_tolerance_m": POSITION_TOLERANCE_M / 10, "solver_orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD / 10,
        "fixed_base_status": "BLOCKED" if blocked else "PASS", "blocked_waypoints": blocked, "waypoints": results,
        "rail_plan": {
            "status": "PASS" if rail_pass else "BLOCKED", "arm": "right", "base_pose_xyz_yaw": rail_base,
            "base_y_shift_m": rail_base[1] - cfg["fr3"]["bases"]["right"][1],
            "travel_limits_m": [low, high], "within_travel_limits": True,
            "flange_rotation_world": B2_RAIL_ROTATION, "waypoints": rail_results,
            "max_joint_step_rad": float(np.max(np.abs(joint_steps))),
            "claim": "42 B2 IK samples only; approach, rail transition, continuous trajectories, collisions, suction and contact are untested.",
        },
    }


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        report = run(json.loads((ROOT / "fr3_four_wall_scene_config.json").read_text()))
        code = 0 if report["status"] == "PASS_KINEMATIC_RAIL_PLAN" else 2
    except Exception as exc:
        report = {"status": "FAIL", "scope": "fixed_base_and_proposed_B2_rail_waypoint_ik_only", "error": repr(exc), "traceback": traceback.format_exc()}
        code = 1
    try:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        import omni.kit.app

        omni.kit.app.get_app().post_quit(code)
        app.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
