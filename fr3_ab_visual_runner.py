"""Automatically execute the new FR3 scene in Isaac Sim.

A uses explicitly labelled temporary fixed joints as an attachment proxy.
This demo never claims Surface Gripper or collision-safety validation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import traceback

from fr3_ab_ik_preflight import SIDE_ROTATION, solve_waypoint

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "fr3_four_wall_scene_config.json").read_text())
REPORT = ROOT / "validation" / "fr3_ab_visual_runner.json"

def _box(stage, path, center, size, collision=False, dynamic=False, mass=None):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim()).CreateRestOffsetAttr().Set(0.0)
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(float(mass))
        PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateEnableCCDAttr(True)
    return cube.GetPrim()


def _set_start_targets(stage, root_path):
    from pxr import Usd

    for index, value in enumerate(CONFIG["fr3"]["official_start_q_rad"], start=1):
        joint = stage.GetPrimAtPath(f"{root_path}/fr3_link{index - 1}/fr3_joint{index}")
        if not joint.IsValid():
            continue
        attr = joint.GetAttribute("drive:angular:physics:targetPosition")
        if attr.IsValid():
            attr.Set(math.degrees(float(value)))


def _add_robot(stage, side, asset_url):
    from pxr import Gf, UsdGeom, UsdPhysics

    base = CONFIG["fr3"]["bases"][side]
    root = f"/World/{side}_fr3"
    robot = stage.DefinePrim(root, "Xform")
    robot.GetReferences().AddReference(asset_url)
    xf = UsdGeom.Xformable(robot)
    xf.AddTranslateOp(opSuffix="ab_base").Set(Gf.Vec3d(*base[:3]))
    xf.AddRotateZOp(opSuffix="ab_yaw").Set(math.degrees(float(base[3])))
    _set_start_targets(stage, root)
    # Collision-enabled support geometry is attached to the real FR3 link8;
    # suction joints are intentionally left to the later A/B runner.
    tool = f"{root}/fr3_link8/ABSideTool"
    _box(stage, f"{tool}/vertical", (0.0, 0.0, 0.04), (0.028, 0.028, 0.08), collision=True)
    sign = -1.0 if side == "left" else 1.0
    _box(stage, f"{tool}/lateral", (0.0, sign * 0.065, 0.08), (0.018, 0.13, 0.018), collision=True)
    _box(stage, f"{tool}/manifold", (0.0, sign * 0.138, 0.08), (0.07, 0.014, 0.07), collision=True)
    # The TCP is at local y=+/-0.155; keep the cup collision surface there.
    for row, z_offset in enumerate((-0.024, 0.024), start=1):
        for column, x_offset in enumerate((-0.024, 0.024), start=1):
            cup = UsdGeom.Cylinder.Define(stage, f"{tool}/cup_r{row}_c{column}")
            cup.CreateAxisAttr(UsdGeom.Tokens.y)
            cup.CreateRadiusAttr(0.01)
            cup.CreateHeightAttr(0.01)
            UsdGeom.Xformable(cup.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(x_offset, sign * 0.15, 0.08 + z_offset)
            )
            UsdPhysics.CollisionAPI.Apply(cup.GetPrim())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run the same simulation without a window")
    parser.add_argument("--close-on-finish", action="store_true", help="Close instead of retaining the result in the GUI")
    args = parser.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "width": 1400, "height": 900})
    print("FR3_AB_APP_READY", flush=True)
    report = {
        "scene_id": CONFIG["scene_id"], "status": "RUNNING", "ab_status": "UNVERIFIED",
        "scope": "automatic_articulation_visual_demo", "attachment_model": "temporary_PhysX_fixed_joints",
        "surface_gripper_validated": False, "collision_safety_validated": False, "phases": [],
    }
    world = None
    status_label = None
    try:
        import numpy as np
        import omni.client
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, interface_config_loader
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

        omni.usd.get_context().new_stage()
        app.update()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
        world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 60)
        for name, center, size in CONFIG["static_boxes"]:
            prim = _box(stage, f"/World/Environment/{name}", center, size, collision=True)
            color = (0.18, 0.45, 0.48) if name == "floor" else (0.20, 0.24, 0.28) if name == "table" else (0.70, 0.75, 0.80)
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
        prim = _box(stage, "/World/Cube", CONFIG["supply_cube_center_m"], [CONFIG["cube_size_m"]] * 3,
                    collision=True, dynamic=True, mass=CONFIG["physics"]["cube_mass_kg"])
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.35, 0.05)])
        for side, base in CONFIG["fr3"]["bases"].items():
            local = CONFIG["rails"]["beam_local_center_m"]
            _box(stage, f"/World/{side}_rail/beam", np.array(base[:3]) + local, CONFIG["rails"]["beam_size_m"])
            _box(stage, f"/World/{side}_rail/carriage", base[:3], CONFIG["rails"]["carriage_size_m"])
        print("FR3_AB_LOADING_ASSETS", flush=True)
        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError("Isaac asset root unavailable")
        for suffix in (CONFIG["fr3"]["asset"], "Isaac/" + CONFIG["fr3"]["asset"]):
            asset_url = f"{assets_root.rstrip('/')}/{suffix}"
            status, _ = omni.client.stat(asset_url)
            if status == omni.client.Result.OK:
                break
        else:
            raise RuntimeError("Official FR3 USD is unavailable")
        for side in ("left", "right"):
            _add_robot(stage, side, asset_url)
        dome = UsdLux.DomeLight.Define(stage, "/World/Lighting")
        dome.CreateIntensityAttr(1500.0)
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.65, -1.7, 2.05), Gf.Vec3d(0, 0.4, 0.8), Gf.Vec3d(0, 0, 1)).GetInverse())
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        camera.CreateFocalLengthAttr(22.0)
        if not args.headless:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().set_active_camera("/World/Camera")
            import omni.ui as ui
            status_window = ui.Window("FR3 automatic task", width=420, height=110)
            with status_window.frame:
                status_label = ui.Label("Loading FR3...", word_wrap=True)
        robots, solvers, joints, flanges = {}, {}, {}, {}
        for side, base in CONFIG["fr3"]["bases"].items():
            robots[side] = world.scene.add(SingleArticulation(f"/World/{side}_fr3", name=f"{side}_robot"))
            solver = LulaKinematicsSolver(**interface_config_loader.load_supported_lula_kinematics_solver_config("FR3"))
            solver.set_robot_base_pose(np.array(base[:3]), np.array([math.cos(base[3] / 2), 0, 0, math.sin(base[3] / 2)]))
            assert solver.get_joint_names() == [f"fr3_joint{i}" for i in range(1, 8)]
            solvers[side] = solver
            flanges[side] = SingleXFormPrim(f"/World/{side}_fr3/fr3_link8")
        cube = world.scene.add(SingleRigidPrim("/World/Cube", name="task_cube"))
        print("FR3_AB_INITIALIZING_PHYSICS", flush=True)
        world.reset()
        for side, robot in robots.items():
            joints[side] = np.array([robot.get_dof_index(f"fr3_joint{i}") for i in range(1, 8)])
            robot.set_joint_positions(np.array(CONFIG["fr3"]["official_start_q_rad"]), joint_indices=joints[side])
            robot.get_articulation_controller().apply_action(ArticulationAction(
                joint_positions=np.array(CONFIG["fr3"]["official_start_q_rad"]), joint_indices=joints[side]))
        print("FR3_AB_PHYSICS_READY", flush=True)
        tick = 0
        start_wall = time.monotonic()

        def step(count=1):
            nonlocal tick
            for _ in range(count):
                if not app.is_running():
                    raise RuntimeError("Viewer was closed before the task finished")
                world.step(render=not args.headless)
                tick += 1
                if not args.headless:
                    time.sleep(max(0.0, min(1 / 120, start_wall + tick / 120 - time.monotonic())))

        def record(name, state, **extra):
            position, orientation = cube.get_world_pose()
            item = {"phase": name, "status": state, "sim_time_s": tick / 120,
                    "cube_position_m": position.tolist(), "cube_orientation_wxyz": orientation.tolist(), **extra}
            report["active_phase"] = name
            report["phases"].append(item)
            if status_label is not None:
                status_label.text = f"{name}: {state}\nAttachment: fixed joint proxy (not suction validation)"
            print("FR3_AB " + json.dumps(item), flush=True)
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

        def move(targets, duration=1.0):
            current = {side: robots[side].get_joint_positions(joint_indices=joints[side]) for side in targets}
            delta = max(float(np.max(np.abs(np.array(targets[side]) - current[side]))) for side in targets)
            frames = max(2, int(max(duration, delta / 0.4) * 120))
            for alpha in np.linspace(0, 1, frames):
                blend = alpha * alpha * (3 - 2 * alpha)
                for side, target in targets.items():
                    q = current[side] + blend * (np.array(target) - current[side])
                    robots[side].get_articulation_controller().apply_action(ArticulationAction(joint_positions=q, joint_indices=joints[side]))
                step()
            step(24)
            for side, target in targets.items():
                error = float(np.max(np.abs(robots[side].get_joint_positions(joint_indices=joints[side]) - target)))
                if error > 0.08:
                    raise RuntimeError(f"{side} joint tracking error {error:.3f} rad; possible contact or unreachable drive target")

        def solve(side, tcp, rotation, continuous=False):
            seed = robots[side].get_joint_positions(joint_indices=joints[side])
            result = solve_waypoint(CONFIG, solvers[side], side, "runtime", list(tcp), rotation, seed)
            jump = float(np.max(np.abs(np.array(result["joint_positions_rad"]) - seed)))
            if continuous and (not result["success"] or jump > 0.35):
                from scipy.optimize import least_squares
                from scipy.spatial.transform import Rotation
                offset = np.array([0, (-1 if side == "left" else 1) * CONFIG["side_suction_tool"]["tcp_side_offset_m"], CONFIG["side_suction_tool"]["vertical_drop_m"]])
                def residual(q):
                    position, actual_rotation = solvers[side].compute_forward_kinematics("fr3_link8", q)
                    actual_tcp = np.asarray(position).reshape(3) + actual_rotation @ offset
                    return np.r_[actual_tcp - tcp, 0.2 * Rotation.from_matrix(np.asarray(rotation).T @ actual_rotation).as_rotvec()]
                properties = robots[side].dof_properties
                low = np.maximum(seed - 0.30, properties["lower"][joints[side]] + 1e-6)
                high = np.minimum(seed + 0.30, properties["upper"][joints[side]] - 1e-6)
                fit = least_squares(residual, np.clip(seed.astype(float), low, high), bounds=(low, high),
                                    max_nfev=100, xtol=1e-10, ftol=1e-10, gtol=1e-10)
                errors = residual(fit.x)
                if np.linalg.norm(errors[:3]) <= 0.001 and np.linalg.norm(errors[3:]) / 0.2 <= 0.01:
                    report["local_ik_recoveries"] = report.get("local_ik_recoveries", 0) + 1
                    return fit.x.tolist()
            if not result["success"]:
                raise RuntimeError(f"{side} IK failed at TCP {list(tcp)}: error {result['tcp_error_m']:.4f} m")
            return result["joint_positions_rad"]

        half = CONFIG["cube_size_m"] / 2

        def dual_target(center, gap=0, continuous=False):
            return {side: solve(side, np.array(center) + [sign * (half + gap), 0, 0], SIDE_ROTATION, continuous)
                    for side, sign in (("left", -1), ("right", 1))}

        def dual_path(start, end):
            distance = float(np.linalg.norm(np.array(end) - start))
            for center in np.linspace(start, end, max(2, int(distance / 0.015) + 1))[1:]:
                targets = dual_target(center, continuous=True)
                for side, target in targets.items():
                    current = robots[side].get_joint_positions(joint_indices=joints[side])
                    change = float(np.max(np.abs(np.array(target) - current)))
                    if change > 0.35:
                        report["blocked_motion"] = {"arm": side, "cube_target_m": center.tolist(),
                            "joint_start_rad": current.tolist(), "joint_target_rad": target, "max_joint_step_rad": change}
                        raise RuntimeError(f"{side} IK branch jumps {change:.3f} rad on a 15 mm path segment; refused execution")
                move(targets, duration=0.15)
                actual, _ = cube.get_world_pose()
                if np.linalg.norm(actual - center) > 0.035:
                    raise RuntimeError(f"Cube departed the A path by {np.linalg.norm(actual - center):.3f} m")

        def attach(side):
            path = f"/World/{side}_fr3/fr3_link8"
            assert stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI), "Attachment must reference a rigid body"
            anchor, flange_quat = flanges[side].get_world_pose()
            # Joint offsets are in the scaled Cube prim coordinates.
            cube_pos, cube_quat = cube.get_world_pose()
            joint = UsdPhysics.FixedJoint.Define(stage, f"/World/Attachment_{side}")
            joint.CreateBody0Rel().SetTargets([Sdf.Path(path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Cube")])
            joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
            from isaacsim.core.utils.rotations import quat_to_rot_matrix
            local_pos = (quat_to_rot_matrix(cube_quat).T @ (np.array(anchor) - cube_pos)) / CONFIG["cube_size_m"]
            assert np.allclose(quat_to_rot_matrix(cube_quat) @ (local_pos * CONFIG["cube_size_m"]) + cube_pos, anchor, atol=1e-6)
            joint.CreateLocalPos1Attr(Gf.Vec3f(*local_pos.tolist()))
            world_rot = Gf.Quatd(float(flange_quat[0]), Gf.Vec3d(*flange_quat[1:].tolist()))
            cube_rot = Gf.Quatd(float(cube_quat[0]), Gf.Vec3d(*cube_quat[1:].tolist()))
            relative = cube_rot.GetInverse() * world_rot
            joint.CreateLocalRot0Attr(Gf.Quatf(1))
            joint.CreateLocalRot1Attr(Gf.Quatf(relative))
            joint.CreateExcludeFromArticulationAttr(True)
            joint.CreateBreakForceAttr(500.0)
            joint.CreateBreakTorqueAttr(100.0)

        step(120)
        record("A_approach", "RUNNING")
        supply = CONFIG["supply_cube_center_m"]
        move(dual_target(supply, gap=0.035), duration=3.0)
        move(dual_target(supply, gap=0.0), duration=1.0)
        actual, _ = cube.get_world_pose()
        if np.linalg.norm(actual - supply) > 0.015:
            raise RuntimeError("Cube moved away before attachment")
        attach("left")
        attach("right")
        step(60)
        record("A_attach", "FIXED_JOINT_PROXY")
        previous = supply
        for name, center in (("A_lift", CONFIG["task"]["phase_A"]["lift_center_m"]),
                             ("A_over_front", CONFIG["task"]["phase_A"]["over_front_center_m"]),
                             ("A_landing", CONFIG["task"]["phase_A"]["landing_center_m"])):
            record(name, "RUNNING")
            dual_path(previous, center)
            record(name, "REACHED_WITH_ATTACHMENT_PROXY")
            previous = center
        for side in robots:
            stage.RemovePrim(f"/World/Attachment_{side}")
        move(dual_target(previous, gap=0.035), duration=1.0)
        step(120)
        actual, _ = cube.get_world_pose()
        landing_error = float(np.linalg.norm(actual - previous))
        if landing_error > 0.015:
            raise RuntimeError(f"A release landing error {landing_error:.3f} m")
        record("A_release", "REACHED_WITH_ATTACHMENT_PROXY", landing_error_m=landing_error)
        report["status"] = "BLOCKED_B_RUNTIME"
        report["ab_status"] = "BLOCKED"
        report["blocker"] = "A attachment-proxy demo completed; B1/B2 contact state machine and rail transfer are not validated or executed"
        record("B1_B2", "NOT_EXECUTED", reason=report["blocker"])
    except Exception as exc:
        report["status"] = "STOPPED"
        report["ab_status"] = "BLOCKED"
        report["error"] = str(exc)
        if "cube" in locals():
            position, orientation = cube.get_world_pose()
            report["stopped_cube_position_m"] = position.tolist()
            report["stopped_cube_orientation_wxyz"] = orientation.tolist()
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
        if status_label is not None:
            status_label.text = "STOPPED: " + str(exc)
    finally:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print("FR3_AB_RESULT " + json.dumps(report, ensure_ascii=False), flush=True)
        if world is not None:
            world.pause()
        if not args.headless and not args.close_on_finish:
            print("Task stopped. Inspect the scene; close the window to exit.", flush=True)
            while app.is_running():
                app.update()
        app.close()
    return 0 if report["status"] == "BLOCKED_B_RUNTIME" else 1


if __name__ == "__main__":
    raise SystemExit(main())
