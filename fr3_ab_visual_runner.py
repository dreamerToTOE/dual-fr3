"""Automatically execute the new FR3 scene in Isaac Sim.

A uses explicitly labelled temporary fixed joints as an attachment proxy.
This demo never claims Surface Gripper or collision-safety validation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import time
import traceback

from fr3_ab_ik_preflight import FRONT_ROTATION, SIDE_ROTATION, solve_waypoint
from fr3_trajectory import plan_collision_free_path, plan_joint_trajectory
from fr3_collision import FR3CollisionChecker
from fr3_contact_monitor import PhysxContactMonitor
from fr3_wrist_ft import add_wrist_sensor, WristForceRecorder

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "fr3_four_wall_scene_config.json").read_text())
TRAJECTORY_CACHE = ROOT / "validation" / "fr3_trajectory_cache.json"

ARUCO_4X4_23_BITS = (
    (1, 1, 1, 1, 1, 1),
    (1, 0, 0, 1, 0, 1),
    (1, 0, 0, 1, 0, 1),
    (1, 0, 1, 1, 1, 1),
    (1, 1, 1, 0, 1, 1),
    (1, 1, 1, 1, 1, 1),
)

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


def _add_aruco_marker(stage, cube_size, marker_size):
    """Add a visual DICT_4X4_50 marker to the Cube's top face."""
    from pxr import Gf, UsdGeom

    UsdGeom.Xform.Define(stage, "/World/Cube/ArUcoMarker")
    # Cube is authored as a unit cube with a parent scale of cube_size.
    # Marker dimensions therefore need to be expressed in the Cube's local
    # unit-cube coordinates to retain their requested world size.
    local_scale = float(cube_size)
    quiet = marker_size * 0.12
    backing_size = marker_size + 2 * quiet
    backing = UsdGeom.Cube.Define(stage, "/World/Cube/ArUcoMarker/white_backing")
    backing.CreateSizeAttr(1.0)
    backing_xf = UsdGeom.Xformable(backing.GetPrim())
    backing_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5 + 0.0004 / local_scale))
    backing_xf.AddScaleOp().Set(Gf.Vec3f(
        backing_size / local_scale, backing_size / local_scale, 0.001 / local_scale))
    backing.CreateDisplayColorAttr([Gf.Vec3f(0.98, 0.98, 0.98)])
    cell_size = marker_size / 6.0 / local_scale
    for row, bits in enumerate(ARUCO_4X4_23_BITS):
        for column, bit in enumerate(bits):
            cell = UsdGeom.Cube.Define(
                stage, f"/World/Cube/ArUcoMarker/cell_{row}_{column}"
            )
            cell.CreateSizeAttr(1.0)
            cell_xf = UsdGeom.Xformable(cell.GetPrim())
            cell_xf.AddTranslateOp().Set(Gf.Vec3d(
                (column - 2.5) * cell_size,
                (2.5 - row) * cell_size,
                0.5 + 0.008 / local_scale,
            ))
            cell_xf.AddScaleOp().Set(Gf.Vec3f(cell_size, cell_size, 0.001 / local_scale))
            color = 0.02 if bit else 0.98
            cell.CreateDisplayColorAttr([Gf.Vec3f(color, color, color)])


def _add_camera_mount(stage, vision_cfg):
    """Add a non-colliding, visible stand and housing for the rear D455."""
    from pxr import Gf, UsdGeom

    mount = vision_cfg.get("mount", {})
    if not mount.get("enabled", True):
        return
    camera = [float(v) for v in vision_cfg["camera_position_m"]]
    base = [float(v) for v in mount.get(
        "base_center_m", [camera[0], camera[1] + 0.07, 0.715])]
    base_size = mount.get("base_size_m", [0.34, 0.28, 0.03])
    pole_size = mount.get("pole_size_m", [0.05, 0.05, 1.07])
    housing_size = mount.get("housing_size_m", [0.18, 0.10, 0.10])

    def visual_box(path, center, size, color):
        prim = _box(stage, path, center, size)
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return prim

    root = "/World/D455/Mount"
    visual_box(f"{root}/base", base, base_size, (0.16, 0.19, 0.23))
    visual_box(
        f"{root}/pole",
        [base[0], base[1], base[2] + 0.5 * pole_size[2] + 0.015],
        pole_size,
        (0.20, 0.24, 0.29),
    )
    visual_box(
        f"{root}/top_arm",
        [camera[0], 0.5 * (base[1] + camera[1] + 0.08), camera[2] - 0.02],
        [pole_size[0], abs(camera[1] + 0.08 - base[1]) + 0.02, 0.04],
        (0.20, 0.24, 0.29),
    )
    body = UsdGeom.Xform.Define(stage, f"{root}/CameraBody")
    UsdGeom.Xformable(body.GetPrim()).AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*camera), Gf.Vec3d(*vision_cfg["camera_look_at_m"]),
            Gf.Vec3d(0, 0, 1),
        ).GetInverse()
    )
    # USD cameras look down -Z; visible hardware stays behind the optical plane.
    visual_box(f"{body.GetPath()}/housing", [0, 0, 0.07], housing_size, (0.07, 0.10, 0.14))

    for name, x_offset, radius in (("color_lens", 0.0, 0.019),
                                   ("left_lens", -0.058, 0.023),
                                   ("right_lens", 0.058, 0.023)):
        lens = UsdGeom.Cylinder.Define(stage, f"{body.GetPath()}/{name}")
        lens.CreateAxisAttr(UsdGeom.Tokens.z)
        lens.CreateRadiusAttr(radius)
        lens.CreateHeightAttr(0.022)
        UsdGeom.Xformable(lens.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(x_offset, 0, 0.014))
        lens.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.06, 0.08)])


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
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    base = CONFIG["fr3"]["bases"][side]
    root = f"/World/{side}_fr3"
    robot = stage.DefinePrim(root, "Xform")
    robot.GetReferences().AddReference(asset_url)
    xf = UsdGeom.Xformable(robot)
    # The referenced FR3 already contributes standard root ops; reuse them so
    # Isaac's articulation teleport API can update PhysX and USD together.
    translate = robot.GetAttribute("xformOp:translate")
    orient = robot.GetAttribute("xformOp:orient")
    if not translate.IsValid():
        translate = xf.AddTranslateOp().GetAttr()
    if not orient.IsValid():
        orient = xf.AddOrientOp().GetAttr()
    translate.Set(Gf.Vec3d(*base[:3]))
    orient.Set(Gf.Quatd(math.cos(float(base[3]) / 2), Gf.Vec3d(0, 0, math.sin(float(base[3]) / 2))))
    _set_start_targets(stage, root)
    # The scene uses the custom side suction tool; keep FR3's stock hand and
    # fingers in the articulation for kinematics, but hide their geometry.
    for name in ("fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_hand_tcp"):
        prim = stage.GetPrimAtPath(f"{root}/{name}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
    # Every custom tool shape remains collidable throughout execution.
    tool = add_wrist_sensor(stage, side, CONFIG["wrist_force_sensors"][side])
    # The sensor occupies the first 12 mm of the original stem: TCP and the
    # conservative collision envelope stay unchanged for cached trajectories.
    sensor = UsdGeom.Cylinder.Define(stage, f"{tool}/sensor_housing")
    sensor.CreateAxisAttr(UsdGeom.Tokens.z)
    sensor.CreateRadiusAttr(0.014)
    sensor.CreateHeightAttr(0.012)
    sensor.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.006))
    sensor.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.35, 0.65)])
    UsdPhysics.CollisionAPI.Apply(sensor.GetPrim())
    _box(stage, f"{tool}/vertical", (0.0, 0.0, 0.046), (0.028, 0.028, 0.068), collision=True)
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
    parser.add_argument("--output-dir", type=Path, help="Keep this run's artifacts separate from earlier experiments")
    args = parser.parse_args()
    output_directory = args.output_dir.resolve() if args.output_dir else ROOT / "validation" / "idea_one_wrist_ft"
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "runner_report.json"
    cache_output = output_directory / "trajectory_cache.json"
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "width": 1400, "height": 900})
    print("FR3_AB_APP_READY", flush=True)
    report = {
        "scene_id": CONFIG["scene_id"], "status": "RUNNING", "ab_status": "UNVERIFIED",
        "scope": "automatic_articulation_visual_demo", "attachment_model": "temporary_PhysX_fixed_joints",
        "surface_gripper_validated": False, "collision_safety_validated": False,
        "proxy_robot_collisions_disabled": False, "robot_collisions_enabled": True,
        "full_arm_planning_check": "official_Lula_spheres_and_full_tool_OBBs",
        "unintended_physx_contact_guard": True, "gravity_compensation": True,
        "b2_right_body_collision_disabled_for_contact": False, "phases": [],
        "vision": {
            "camera_model": CONFIG["vision"]["camera_model"],
            "camera_asset": CONFIG["vision"]["camera_asset"],
            "depth_camera_path": CONFIG["vision"]["depth_camera_path"],
            "camera_path": CONFIG["vision"]["camera_path"],
            "camera_position_m": CONFIG["vision"]["camera_position_m"],
            "camera_look_at_m": CONFIG["vision"]["camera_look_at_m"],
            "resolution": CONFIG["vision"]["resolution"],
            "aruco_dictionary": CONFIG["vision"]["aruco_dictionary"],
            "aruco_marker_id": CONFIG["vision"]["aruco_marker_id"],
            "aruco_marker_face": CONFIG["vision"]["aruco_marker_face"],
            "samples": 0,
            "detections": 0,
            "misses": 0,
            "last_detection": None,
            "detection_log": [],
        },
    }
    trajectory_cache = {
        "schema_version": 1,
        "scene_id": CONFIG["scene_id"],
        "status": "RUNNING",
        "source": "runtime_120hz_collision_checked_execution",
        "config_file": "fr3_four_wall_scene_config.json",
        "phases": [],
        "rail_transfers": [],
    }
    loaded_trajectory_cache = None
    cache_cursor = 0
    rail_cache_cursor = 0
    cache_enabled = False
    world = None
    status_label = None
    force_label = None
    contact_monitor = None
    vision_camera = None
    vision_viewport = None
    vision_loop = None
    vision_task = None
    vision_comparison = None
    force_recorder = None
    try:
        import numpy as np
        import omni.client
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
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
        _add_aruco_marker(stage, CONFIG["cube_size_m"], CONFIG["vision"]["aruco_marker_size_m"])
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
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.65, -1.7, 2.05), Gf.Vec3d(0, 0.6, 1.0), Gf.Vec3d(0, 0, 1)).GetInverse())
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        camera.CreateFocalLengthAttr(14.0)
        vision_cfg = CONFIG["vision"]
        d455_url = f"{assets_root.rstrip('/')}/{vision_cfg['camera_asset']}"
        d455_status, _ = omni.client.stat(d455_url)
        if d455_status == omni.client.Result.OK:
            vision_root = stage.DefinePrim(vision_cfg["camera_root_path"], "Xform")
            vision_root.GetReferences().AddReference(d455_url)
            app.update()
            rsd_root = stage.GetPrimAtPath(f"{vision_cfg['camera_root_path']}/RSD455")
            rsd_xf = UsdGeom.Xformable(rsd_root)
            rsd_xf.ClearXformOpOrder()
            rsd_xf.AddTransformOp().Set(Gf.Matrix4d(1.0))
            imported_visual = stage.GetPrimAtPath(f"{vision_cfg['camera_root_path']}/RSD455/Visual")
            if imported_visual.IsValid():
                UsdGeom.Imageable(imported_visual).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            vision_runtime_path = "/World/D455/CalibratedColorCamera"
            report["vision"]["camera_mode"] = "d455_calibrated_pinhole_fallback"
            report["vision"]["d455_color_prim"] = vision_cfg["camera_path"]
            report["vision"]["d455_color_prim_renderable"] = False
            report["vision"]["fallback_reason"] = (
                "Isaac Sim 4.5 D455 USD color prim carries unsupported cameraProjectionType metadata; "
                "capture uses a pinhole camera with the D455 color intrinsics"
            )
        else:
            vision_runtime_path = "/World/VisionCameraFallback"
            report["vision"]["camera_mode"] = "generic_fallback"
        _add_camera_mount(stage, vision_cfg)
        vision_camera_prim = UsdGeom.Camera.Define(stage, vision_runtime_path)
        vision_camera_prim.AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*vision_cfg["camera_position_m"]),
                Gf.Vec3d(*vision_cfg["camera_look_at_m"]),
                Gf.Vec3d(0, 0, 1),
            ).GetInverse()
        )
        vision_camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        vision_camera_prim.CreateFocalLengthAttr(1.93 if d455_status == omni.client.Result.OK else 24.0)
        vision_camera_prim.CreateHorizontalApertureAttr(3.896 if d455_status == omni.client.Result.OK else 36.0)
        vision_camera_prim.CreateVerticalApertureAttr(2.453 if d455_status == omni.client.Result.OK else 20.25)
        report["vision"]["camera_path"] = vision_runtime_path
        from isaacsim.sensors.camera import Camera as IsaacCamera
        if not args.headless:
            from omni.kit.viewport.utility import get_active_viewport
            vision_viewport = get_active_viewport()
        vision_camera = world.scene.add(IsaacCamera(
            prim_path=vision_runtime_path,
            name="vision_camera",
            frequency=vision_cfg["frequency_hz"],
            resolution=tuple(vision_cfg["resolution"]),
            # Keep the 640x480 sensor product separate from the 1400x900 overview viewport.
            render_product_path=None,
        ))
        vision_camera.set_projection_type("pinhole")
        if not args.headless:
            import omni.ui as ui
            status_window = ui.Window("FR3 automatic task", width=520, height=180)
            with status_window.frame:
                with ui.VStack():
                    status_label = ui.Label("Loading FR3...", word_wrap=True)
                    force_label = ui.Label("Wrist F/T: waiting for physics", word_wrap=True)
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
        contact_monitor = PhysxContactMonitor(stage)
        try:
            loaded_candidate = json.loads(TRAJECTORY_CACHE.read_text())
            if (loaded_candidate.get("status") == "COMPLETE"
                    and loaded_candidate.get("scene_id") == CONFIG["scene_id"]
                    and isinstance(loaded_candidate.get("phases"), list)):
                loaded_trajectory_cache = loaded_candidate
                cache_enabled = True
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            loaded_trajectory_cache = None
        world.reset()
        if vision_camera is not None:
            vision_camera.resume()
        if vision_viewport is not None:
            vision_viewport.set_active_camera("/World/Camera")
            vision_viewport.resolution = (1400, 900)
            assert vision_camera.get_render_product_path() != vision_viewport.get_render_product_path()
            report["vision"]["overview_resolution"] = list(vision_viewport.resolution)
            report["vision"]["independent_render_product"] = True
        for side, robot in robots.items():
            joints[side] = np.array([robot.get_dof_index(f"fr3_joint{i}") for i in range(1, 8)])
            robot.set_joint_positions(np.array(CONFIG["fr3"]["official_start_q_rad"]), joint_indices=joints[side])
            robot.get_articulation_controller().apply_action(ArticulationAction(
                joint_positions=np.array(CONFIG["fr3"]["official_start_q_rad"]), joint_indices=joints[side]))
        force_recorder = WristForceRecorder(robots, joints, CONFIG, output_directory)
        calibration_path = ROOT / "validation" / "idea_one_wrist_ft_load_check.json"
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text())
            force_recorder.metadata["calibration_passed"] = calibration.get("status") == "PASS"
            force_recorder.metadata["calibration_report"] = str(calibration_path.relative_to(ROOT))
        print("FR3_AB_PHYSICS_READY", flush=True)
        checker = FR3CollisionChecker(CONFIG, solvers)
        allowed_contacts = {"left", "right"}
        contact_monitor.allowed_cube_contacts = allowed_contacts
        report["trajectory_cache"] = {
            "loaded": bool(cache_enabled),
            "path": str(TRAJECTORY_CACHE.relative_to(ROOT)),
            "used_phases": [],
            "used_rail_transfers": 0,
            "fallback_reason": None,
        }
        tick = 0
        initial_physics_step = world.current_time_step_index
        last_control_wall = time.monotonic()
        last_render_wall = last_control_wall
        report["simulation_clock"] = {"control_hz": 120, "render_hz": 60,
                                      "render_count": 0, "uncommanded_physics_steps": 0}
        report["vision"]["sampling_mode"] = "async_sensor" if vision_viewport is not None else "sync_sensor"
        if vision_viewport is not None:
            vision_loop = asyncio.get_event_loop()

        def render_only():
            nonlocal last_render_wall
            before = world.current_time_step_index
            world.render()
            extra_steps = world.current_time_step_index - before
            report["simulation_clock"]["uncommanded_physics_steps"] += extra_steps
            if extra_steps:
                raise RuntimeError("Rendering advanced physics outside the control loop")
            last_render_wall = time.monotonic()
            report["simulation_clock"]["render_count"] += 1

        def planning_update(message, force=False):
            if args.headless:
                return
            if not app.is_running():
                raise RuntimeError("Viewer was closed during planning")
            if force or time.monotonic() - last_render_wall >= 0.1:
                status_label.text = message
                render_only()

        from fr3_vision import ArucoCubeEstimator
        from fr3_vision_comparison import VisionComparison, match_reference

        vision_period = max(1, int(round(120 / vision_cfg["frequency_hz"])))
        camera_matrix = np.asarray(vision_camera.get_intrinsics_matrix(), dtype=float)
        camera_to_world = np.linalg.inv(np.asarray(vision_camera.get_view_matrix_ros(), dtype=float))
        vision_estimator = ArucoCubeEstimator(
            camera_matrix, camera_to_world,
            marker_id=vision_cfg["aruco_marker_id"],
            marker_size_m=vision_cfg["aruco_marker_size_m"],
            cube_size_m=CONFIG["cube_size_m"],
            marker_surface_offset_m=0.0085,
            dictionary_name=vision_cfg["aruco_dictionary"],
        )
        comparison_directory = output_directory / "vision"
        vision_comparison = VisionComparison(comparison_directory, metadata={
            "scene_id": CONFIG["scene_id"], "camera_model": vision_cfg["camera_model"],
            "execution_mode": "headless" if args.headless else "visible",
            "simulation_time_origin_s": float(world.current_time),
            "camera_mode": report["vision"]["camera_mode"],
            "camera_matrix": camera_matrix.tolist(),
            "camera_to_world_ros": camera_to_world.tolist(),
            "resolution": vision_cfg["resolution"], "frequency_hz": vision_cfg["frequency_hz"],
            "aruco_dictionary": vision_cfg["aruco_dictionary"],
            "aruco_marker_id": vision_cfg["aruco_marker_id"],
            "aruco_marker_size_m": vision_cfg["aruco_marker_size_m"],
            "cube_size_m": CONFIG["cube_size_m"], "marker_surface_offset_m": 0.0085,
            "ground_truth_source": "Isaac Sim PhysX Cube.get_world_pose at exposure time",
            "camera_source": "RGB + ArUco + solvePnP + fixed camera calibration only",
            "camera_uses_cube_truth": False, "control_state_source": "unchanged Isaac Sim state",
            "pairing": "nearest 120 Hz reference within half a physics step; duplicate images excluded",
            "distortion": [0.0] * 5,
            "hardware_calibration_validated": False,
        })
        report["vision"]["comparison_directory"] = str(comparison_directory)
        report["vision"]["camera_uses_cube_truth"] = False
        report["vision"]["unique_frames"] = 0
        report["vision"]["duplicate_frames"] = 0
        initial_center, _ = cube.get_world_pose()
        truth_history = [{"time_s": float(world.current_time), "physics_step": tick,
                          "phase": contact_monitor.phase, "position_m": initial_center.tolist()}]
        seen_vision_frames = set()

        def sample_vision(render=True):
            if render:
                render_only()
            # Snapshot image and exposure metadata together; reading a newer RGB
            # buffer with an older timestamp would compare different instants.
            frame = vision_camera.get_current_frame(clone=True)
            frame_id = int(frame.get("rendering_frame", 0))
            raw_capture_time = float(frame.get("rendering_time", 0.0))
            receive_time = float(world.current_time)
            capture_time = (raw_capture_time if frame_id > 0 and np.isfinite(raw_capture_time)
                            else receive_time)
            reference = match_reference(truth_history, capture_time)
            observation = {"status": "no_frame", "cube_position_m": None}
            rgba = np.asarray(frame.get("rgba", []))
            expected_shape = (vision_cfg["resolution"][1], vision_cfg["resolution"][0], 4)
            if frame_id > 0 and np.isfinite(raw_capture_time) and rgba.shape == expected_shape:
                if frame_id in seen_vision_frames:
                    observation["status"] = "duplicate_frame"
                    report["vision"]["duplicate_frames"] += 1
                else:
                    seen_vision_frames.add(frame_id)
                    report["vision"]["unique_frames"] += 1
                    observation = vision_estimator.estimate(rgba.astype(np.uint8, copy=False))
            report["vision"]["samples"] += 1
            paired = vision_comparison.append(
                sample_id=report["vision"]["samples"],
                phase=contact_monitor.phase if reference is None else reference["phase"],
                physics_step=None if reference is None else reference["physics_step"],
                capture_time_s=capture_time,
                truth_position_m=None if reference is None else reference["position_m"],
                camera_position_m=observation["cube_position_m"],
                status=observation["status"], frame_id=frame_id,
                frame_metadata={
                    "reported_capture_time_s": raw_capture_time if np.isfinite(raw_capture_time) else None,
                    "received_time_s": receive_time, "received_physics_step": tick,
                    "truth_time_s": None if reference is None else reference["time_s"],
                    "pair_time_delta_s": None if reference is None else reference["time_s"] - capture_time,
                    "image_shape": list(rgba.shape),
                    "reprojection_error_px": observation.get("reprojection_error_px"),
                },
            )
            if observation["cube_position_m"] is None:
                report["vision"]["misses"] += 1
                return
            detection = {
                "physics_step": paired["physics_step"], "capture_time_s": capture_time,
                "received_physics_step": tick, "frame_id": frame_id,
                "pixel_center": observation["pixel_center"],
                "cube_position_estimate_m": observation["cube_position_m"],
                "cube_position_actual_m": paired["truth_position_m"],
                "position_error_m": paired["position_error_m"],
                "reprojection_error_px": observation["reprojection_error_px"],
            }
            report["vision"]["detections"] += 1
            report["vision"]["last_detection"] = detection
            report["vision"]["detection_log"].append(detection)
            report["vision"]["detection_log"] = report["vision"]["detection_log"][-20:]

        async def sample_vision_async():
            await asyncio.sleep(0)
            sample_vision(render=False)

        def step(count=1):
            nonlocal tick, last_control_wall, vision_task
            for _ in range(count):
                if not app.is_running():
                    raise RuntimeError("Viewer was closed before the task finished")
                if not args.headless:
                    time.sleep(max(0.0, 1 / 120 - (time.monotonic() - last_control_wall)))
                last_control_wall = time.monotonic()
                for side, robot in robots.items():
                    gravity = robot._articulation_view.get_generalized_gravity_forces(joint_indices=joints[side])[0]
                    if not np.all(np.isfinite(gravity)):
                        raise RuntimeError(f"{side} gravity compensation is not finite")
                    robot.get_articulation_controller().apply_action(
                        ArticulationAction(joint_efforts=gravity, joint_indices=joints[side]))
                # One command advances one physics tick in both execution modes.
                contact_monitor.physics_step_index = world.current_time_step_index + 1
                world.step(render=False)
                tick += 1
                force_sample = force_recorder.sample(float(world.current_time), tick, contact_monitor.phase)
                if force_label is not None and tick % 12 == 0:
                    lines = ["World axes, tool gravity removed; F in N, T in N m"]
                    for side in ("left", "right"):
                        wrench = force_sample[side]["gravity_corrected_world"]
                        force = ", ".join(f"{value:+.2f}" for value in wrench[:3])
                        torque = ", ".join(f"{value:+.3f}" for value in wrench[3:])
                        lines.append(f"{side}: F [{force}]  T [{torque}]")
                    force_label.text = "\n".join(lines)
                contact_monitor.raise_if_collision()
                reference_center, _ = cube.get_world_pose()
                truth_history.append({"time_s": float(world.current_time), "physics_step": tick,
                                      "phase": contact_monitor.phase,
                                      "position_m": reference_center.tolist()})
                if world.current_time_step_index - initial_physics_step != tick:
                    raise RuntimeError("Physics and command step counts diverged")
                report["simulation_clock"]["control_steps"] = tick
                report["simulation_clock"]["physics_steps"] = world.current_time_step_index - initial_physics_step
                if vision_task is not None and vision_task.done():
                    vision_task.result()
                if tick % vision_period == 0:
                    if vision_loop is None:
                        sample_vision()
                    elif vision_task is None or vision_task.done():
                        render_only()
                        vision_task = asyncio.ensure_future(sample_vision_async())
                if not args.headless and tick % 2 == 0:
                    render_only()

        def record(name, state, **extra):
            position, orientation = cube.get_world_pose()
            item = {"phase": name, "status": state, "sim_time_s": tick / 120,
                    "cube_position_m": position.tolist(), "cube_orientation_wxyz": orientation.tolist(), **extra}
            contact_monitor.phase = name
            report["active_phase"] = name
            report["phases"].append(item)
            if status_label is not None:
                status_label.text = f"{name}: {state}\nAttachment: fixed joint proxy (not suction validation)"
            print("FR3_AB " + json.dumps(item), flush=True)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

        def joint_state():
            return {side: np.asarray(robot.get_joint_positions(joint_indices=joints[side]), dtype=float)
                    for side, robot in robots.items()}

        def take_cached_phase(phase, current, requested_paths=None, validator=None):
            nonlocal cache_cursor, cache_enabled
            if not cache_enabled or loaded_trajectory_cache is None:
                return None
            phases = loaded_trajectory_cache["phases"]
            if cache_cursor >= len(phases):
                cache_enabled = False
                report["trajectory_cache"]["fallback_reason"] = f"cache exhausted before {phase}"
                return None
            entry = phases[cache_cursor]
            if entry.get("phase") != phase:
                cache_enabled = False
                report["trajectory_cache"]["fallback_reason"] = (
                    f"phase mismatch at cache index {cache_cursor}: expected {phase}, "
                    f"found {entry.get('phase')}"
                )
                return None
            samples = {side: np.asarray(values, dtype=float)
                       for side, values in entry.get("joint_samples_rad", {}).items()}
            if set(samples) != set(current) or any(path.ndim != 2 or path.shape[1] != 7
                                                    or not np.all(np.isfinite(path))
                                                    for path in samples.values()):
                cache_enabled = False
                report["trajectory_cache"]["fallback_reason"] = f"invalid samples in {phase}"
                return None
            if any(float(np.max(np.abs(samples[side][0] - current[side]))) > 0.08
                   for side in current):
                cache_enabled = False
                report["trajectory_cache"]["fallback_reason"] = f"joint start mismatch in {phase}"
                return None
            if requested_paths is not None:
                for side, path in requested_paths.items():
                    if float(np.max(np.abs(samples[side][-1] - np.asarray(path, dtype=float)[-1]))) > 0.08:
                        cache_enabled = False
                        report["trajectory_cache"]["fallback_reason"] = f"joint goal mismatch in {phase}"
                        return None
            if validator is not None and not validator(entry, samples):
                cache_enabled = False
                report["trajectory_cache"]["fallback_reason"] = f"pose mismatch in {phase}"
                return None
            cache_cursor += 1
            report["trajectory_cache"]["used_phases"].append(phase)
            return entry

        def cached_side_tcp(entry, samples, side):
            q = samples[side][-1]
            flange, rotation = solvers[side].compute_forward_kinematics("fr3_link8", q)
            sign = -1 if side == "left" else 1
            return np.asarray(flange).reshape(3) + np.asarray(rotation) @ np.array([
                0, sign * CONFIG["side_suction_tool"]["tcp_side_offset_m"],
                CONFIG["side_suction_tool"]["vertical_drop_m"]])

        def cached_dual_target_matches(entry, samples, center, half):
            for side, sign in (("left", -1), ("right", 1)):
                tcp = cached_side_tcp(entry, samples, side)
                if np.linalg.norm(tcp - (np.asarray(center) + [sign * half, 0, 0])) > 0.025:
                    return False
            return True

        def move(targets, duration=1.0):
            current = joint_state()
            sides = list(targets)
            phase = contact_monitor.phase + "_transit"
            # The runtime IK target may choose a different redundant FR3 joint
            # branch while reaching the same TCP pose.  Cache replay is gated
            # by the measured start state and the cached collision-checked end
            # state, so do not reject it on raw joint-goal equality here.
            cached = take_cached_phase(phase, current)
            if cached is not None:
                execute_joint_path(
                    {side: np.asarray(cached["waypoints_rad"][side], dtype=float) for side in sides},
                    phase, max_joint_speed=0.4, prefetched_cache=cached)
                return
            planning_message = f"{contact_monitor.phase}: RRT planning"
            planning_update(planning_message, force=True)
            start = np.concatenate([current[side] for side in sides])
            goal = np.concatenate([targets[side] for side in sides])
            lower = np.concatenate([robots[side].dof_properties["lower"][joints[side]] for side in sides])
            upper = np.concatenate([robots[side].dof_properties["upper"][joints[side]] for side in sides])
            center, orientation = cube.get_world_pose()
            from isaacsim.core.utils.rotations import quat_to_rot_matrix
            def clear(q):
                planning_update(planning_message)
                states = {**current, **{side: q[i * 7:(i + 1) * 7] for i, side in enumerate(sides)}}
                return checker.is_clear(states, cube_center=center, cube_rotation=quat_to_rot_matrix(orientation),
                                        cup_contacts=allowed_contacts)
            path = plan_collision_free_path(start, goal, clear, lower, upper)
            paths = {side: path[:, i * 7:(i + 1) * 7] for i, side in enumerate(sides)}
            execute_joint_path(paths, phase, max_joint_speed=0.4)

        def solve(side, tcp, rotation, continuous=False, seed_override=None):
            planning_update(f"{contact_monitor.phase}: {side} IK planning")
            seed = (robots[side].get_joint_positions(joint_indices=joints[side])
                    if seed_override is None else np.asarray(seed_override, dtype=float))
            result = solve_waypoint(CONFIG, solvers[side], side, "runtime", list(tcp), rotation, seed)
            jump = float(np.max(np.abs(np.array(result["joint_positions_rad"]) - seed)))
            if continuous and (not result["success"] or jump > 0.35):
                from scipy.optimize import least_squares
                from scipy.spatial.transform import Rotation
                offset = np.array([0, (-1 if side == "left" else 1) * CONFIG["side_suction_tool"]["tcp_side_offset_m"], CONFIG["side_suction_tool"]["vertical_drop_m"]])
                def residual(q):
                    planning_update(f"{contact_monitor.phase}: {side} IK planning")
                    position, actual_rotation = solvers[side].compute_forward_kinematics("fr3_link8", q)
                    actual_tcp = np.asarray(position).reshape(3) + actual_rotation @ offset
                    return np.r_[actual_tcp - tcp, 0.2 * Rotation.from_matrix(np.asarray(rotation).T @ actual_rotation).as_rotvec()]
                properties = robots[side].dof_properties
                low = np.maximum(seed - 0.30, properties["lower"][joints[side]] + 1e-6)
                high = np.minimum(seed + 0.30, properties["upper"][joints[side]] - 1e-6)
                # Resolve FR3's redundant joint direction near the current pose instead of drifting to a joint limit.
                fit = least_squares(lambda q: np.r_[residual(q), 1e-4 * (q - seed)],
                                    np.clip(seed.astype(float), low, high), bounds=(low, high),
                                    max_nfev=100, xtol=1e-10, ftol=1e-10, gtol=1e-10)
                errors = residual(fit.x)
                if np.linalg.norm(errors[:3]) <= 0.001 and np.linalg.norm(errors[3:]) / 0.2 <= 0.01:
                    report["local_ik_recoveries"] = report.get("local_ik_recoveries", 0) + 1
                    return fit.x.tolist()
            if not result["success"]:
                raise RuntimeError(f"{side} IK failed at TCP {list(tcp)}: error {result['tcp_error_m']:.4f} m")
            return result["joint_positions_rad"]

        half = CONFIG["cube_size_m"] / 2

        def dual_target(center, gap=0, continuous=False, seed_overrides=None):
            return {side: solve(side, np.array(center) + [sign * (half + gap), 0, 0], SIDE_ROTATION,
                                continuous, None if seed_overrides is None else seed_overrides.get(side))
                    for side, sign in (("left", -1), ("right", 1))}

        def execute_joint_path(paths, phase, max_joint_speed=0.4, check_step=None,
                                moving_cube_offset=None, cube_center_path=None,
                                prefetched_cache=None):
            planning_update(f"{phase}: trajectory planning", force=True)
            current = joint_state()
            count = {len(path) for path in paths.values()}
            if len(count) != 1 or not count or next(iter(count)) == 0:
                raise ValueError("Synchronized paths need the same positive number of waypoints")
            sides = list(current)
            cached_entry = prefetched_cache or take_cached_phase(phase, current, requested_paths=paths)
            if cached_entry is not None:
                times = np.asarray(cached_entry.get("times_s"), dtype=float)
                samples = {side: np.asarray(values, dtype=float)
                           for side, values in cached_entry.get("joint_samples_rad", {}).items()}
                if (times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times))
                        or np.any(np.diff(times) < 0)
                        or set(samples) != set(sides)
                        or any(path.shape != (len(times), 7) for path in samples.values())):
                    raise RuntimeError(f"Invalid cached trajectory samples in {phase}")
                all_paths = samples
                replay_source = "cached_trajectory"
            else:
                all_paths = {side: np.vstack((current[side], np.asarray(paths[side], dtype=float)))
                             if side in paths else np.tile(current[side], (next(iter(count)) + 1, 1))
                             for side in sides}
                times, combined = plan_joint_trajectory(np.hstack([all_paths[side] for side in sides]),
                                                        max_joint_speed=max_joint_speed, sample_hz=120.0)
                samples = {side: combined[:, i * 7:(i + 1) * 7] for i, side in enumerate(sides)}
                replay_source = "online_planner"
            def within_limits():
                return all(np.all(np.isfinite(path))
                           and np.all(path >= robots[side].dof_properties["lower"][joints[side]])
                           and np.all(path <= robots[side].dof_properties["upper"][joints[side]])
                           for side, path in samples.items())
            if not within_limits():
                raise RuntimeError(f"{phase} time-parameterized trajectory exceeds joint limits")
            center, orientation = cube.get_world_pose()
            from isaacsim.core.utils.rotations import quat_to_rot_matrix
            cube_centers = center
            cached_cube_path = None if cached_entry is None else cached_entry.get("cube_center_path_m")
            if cached_cube_path is not None:
                cube_centers = np.asarray(cached_cube_path, dtype=float)
                if cube_centers.shape != (len(times), 3) or not np.all(np.isfinite(cube_centers)):
                    raise RuntimeError(f"Invalid cached Cube path in {phase}")
            elif cube_center_path is not None:
                cube_centers = np.asarray(cube_center_path(samples, times), dtype=float)
                if cube_centers.shape != (len(times), 3):
                    raise ValueError(f"{phase} cube_center_path must return one 3D center per sample")
            elif moving_cube_offset is not None:
                side, offset = moving_cube_offset
                sign = -1 if side == "left" else 1
                tool_offset = np.array([0, sign * CONFIG["side_suction_tool"]["tcp_side_offset_m"],
                                        CONFIG["side_suction_tool"]["vertical_drop_m"]])
                cube_centers = []
                for q in samples[side]:
                    planning_update(f"{phase}: carried Cube path")
                    flange, rotation = solvers[side].compute_forward_kinematics("fr3_link8", q)
                    cube_centers.append(np.asarray(flange).reshape(3) + rotation @ tool_offset + offset)
                cube_centers = np.asarray(cube_centers)
            def planning_progress(checked):
                planning_update(f"{phase}: collision checking {checked} samples")
            if cached_entry is not None:
                validation = {"status": "CLEAR", "samples_checked": len(times),
                              "issues": [], "source": "cached_trajectory"}
            else:
                validation = checker.validate_path(samples, cube_centers=cube_centers,
                                                   cube_rotations=quat_to_rot_matrix(orientation),
                                                   cup_contacts=allowed_contacts,
                                                   progress_callback=planning_progress)
            if validation["status"] != "CLEAR" and moving_cube_offset is None:
                # Retain the checked straight RRT edges if corner blending cuts into an obstacle.
                times, combined = plan_joint_trajectory(np.hstack([all_paths[side] for side in sides]),
                    max_joint_speed=max_joint_speed, sample_hz=120.0, stop_at_waypoints=True)
                samples = {side: combined[:, i * 7:(i + 1) * 7] for i, side in enumerate(sides)}
                if not within_limits():
                    raise RuntimeError(f"{phase} corner trajectory exceeds joint limits")
                if cube_center_path is not None:
                    cube_centers = np.asarray(cube_center_path(samples, times), dtype=float)
                    if cube_centers.shape != (len(times), 3):
                        raise ValueError(f"{phase} cube_center_path must return one 3D center per sample")
                validation = checker.validate_path(samples, cube_centers=cube_centers,
                    cube_rotations=quat_to_rot_matrix(orientation), cup_contacts=allowed_contacts,
                    progress_callback=planning_progress)
            if validation["status"] != "CLEAR":
                report["blocked_motion"] = {"phase": phase, **validation}
                raise RuntimeError(f"Planned collision in {phase}: {validation}")
            report.setdefault("trajectory_plans", []).append({
                "phase": phase, "arm": list(paths), "samples": len(times), "duration_s": float(times[-1]),
                "collision_check": validation, "source": replay_source,
            })
            trajectory_cache["phases"].append({
                "phase": phase,
                "arms": list(paths),
                "duration_s": float(times[-1]),
                "times_s": times.tolist(),
                "waypoints_rad": {side: np.asarray(path, dtype=float).tolist() for side, path in paths.items()},
                "joint_samples_rad": {side: np.asarray(path, dtype=float).tolist() for side, path in samples.items()},
                "cube_center_path_m": (None if np.asarray(cube_centers).ndim == 1
                                        else np.asarray(cube_centers, dtype=float).tolist()),
                "collision_check": validation,
            })
            print(f"FR3_PLANNED {phase}: {len(times)} collision-checked samples", flush=True)
            planning_update(f"{phase}: executing", force=True)
            contact_monitor.phase = phase
            for index in range(len(times)):
                for side in sides:
                    robots[side].get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=samples[side][index], joint_indices=joints[side]))
                step()
                actual_center, actual_orientation = cube.get_world_pose()
                issues = checker.check(joint_state(), cube_center=actual_center,
                                       cube_rotation=quat_to_rot_matrix(actual_orientation),
                                       cup_contacts=allowed_contacts, margin_m=0.0, first_only=True)
                if issues:
                    raise RuntimeError(f"Measured collision in {phase}: {issues[0]}")
                if moving_cube_offset is not None or cube_center_path is not None:
                    deviation = float(np.linalg.norm(actual_center - cube_centers[index]))
                    if deviation > 0.035 and moving_cube_offset is not None:
                        raise RuntimeError(f"Cube lost contact with the {phase} trajectory: {deviation:.3f} m")
                if check_step is not None:
                    check_step(index, len(times))
            for side in paths:
                error = float(np.max(np.abs(joint_state()[side] - all_paths[side][-1])))
                if error > 0.08:
                    raise RuntimeError(f"{side} {phase} joint tracking error {error:.3f} rad")

        def dual_path(start, end):
            start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
            def check_step(index, count):
                # The fixed-joint proxy can lag the Cartesian interpolation
                # during a high-acceleration segment. Validate the endpoint;
                # B contact paths retain per-step checks below.
                if index != count - 1:
                    return
                actual, _ = cube.get_world_pose()
                deviation = float(np.linalg.norm(np.asarray(actual) - end))
                if deviation > 0.08:
                    raise RuntimeError(f"Cube departed the A path by {deviation:.3f} m")
            cached = take_cached_phase(
                "A_transfer", joint_state(),
                validator=lambda entry, samples: cached_dual_target_matches(entry, samples, end, half))
            if cached is not None:
                execute_joint_path(
                    {side: np.asarray(cached["waypoints_rad"][side], dtype=float)
                     for side in ("left", "right")},
                    "A_transfer", check_step=check_step,
                    moving_cube_offset=("left", np.array([half, 0, 0])),
                    prefetched_cache=cached)
                return
            centers = np.linspace(start, end, max(2, int(np.linalg.norm(end - start) / 0.015) + 1))[1:]
            targets_by_side = {"left": [], "right": []}
            seeds = {side: np.asarray(robots[side].get_joint_positions(joint_indices=joints[side]), dtype=float)
                     for side in targets_by_side}
            for center in centers:
                targets = dual_target(center, continuous=True, seed_overrides=seeds)
                for side, target in targets.items():
                    target = np.asarray(target, dtype=float)
                    change = float(np.max(np.abs(target - seeds[side])))
                    if change > 0.35:
                        report["blocked_motion"] = {"arm": side, "cube_target_m": center.tolist(),
                            "joint_start_rad": seeds[side].tolist(), "joint_target_rad": target.tolist(), "max_joint_step_rad": change}
                        raise RuntimeError(f"{side} IK branch jumps {change:.3f} rad on a 15 mm path segment; refused execution")
                    targets_by_side[side].append(target)
                    seeds[side] = target
            execute_joint_path(targets_by_side, "A_transfer", check_step=check_step,
                               moving_cube_offset=("left", np.array([half, 0, 0])))

        def contact_path(side, start, end, rotation, tcp_offset, phase, max_step_m=0.015):
            start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
            def check_step(index, count):
                # Check the moving contact against the intended straight path,
                # while allowing one cube-width of compliant PhysX response.
                ratio = index / max(1, count - 1)
                expected = start + (end - start) * ratio
                actual, _ = cube.get_world_pose()
                deviation = float(np.linalg.norm(np.asarray(actual) - expected))
                if deviation > 0.08:
                    raise RuntimeError(f"Cube departed {phase} path by {deviation:.3f} m")
            cached = take_cached_phase(
                phase, joint_state(),
                validator=lambda entry, samples: (
                    np.linalg.norm(cached_side_tcp(entry, samples, side)
                                   - (end + np.asarray(tcp_offset))) <= 0.025))
            if cached is not None:
                execute_joint_path(
                    {side: np.asarray(cached["waypoints_rad"][side], dtype=float)},
                    phase, check_step=check_step,
                    moving_cube_offset=(side, -np.asarray(tcp_offset)),
                    prefetched_cache=cached)
                return
            distance = float(np.linalg.norm(end - start))
            centers = np.linspace(start, end, max(2, int(distance / max_step_m) + 1))[1:]
            targets = []
            seed = np.asarray(robots[side].get_joint_positions(joint_indices=joints[side]), dtype=float)
            for center in centers:
                target = np.asarray(solve(side, center + tcp_offset, rotation, continuous=True,
                                          seed_override=seed), dtype=float)
                change = float(np.max(np.abs(target - seed)))
                if change > 0.35:
                    raise RuntimeError(f"{side} {phase} joint step {change:.3f} rad; refused contact execution")
                targets.append(target)
                seed = target
            execute_joint_path({side: targets}, phase, check_step=check_step,
                               moving_cube_offset=(side, -np.asarray(tcp_offset)))

        def set_robot_base_pose(side, base):
            root = stage.GetPrimAtPath(f"/World/{side}_fr3")
            attr = root.GetAttribute("xformOp:translate")
            if not attr.IsValid():
                raise RuntimeError(f"{side} rail base transform is unavailable")
            base = np.asarray(base, dtype=float)
            attr.Set(Gf.Vec3d(*base[:3]))
            # Teleport the articulation root through its physics view as well
            # as the standard USD root op; the singular wrapper only updates USD.
            robots[side]._articulation_view.set_world_poses(
                positions=np.asarray([base[:3]], dtype=float),
                orientations=np.asarray([[math.cos(base[3] / 2), 0, 0, math.sin(base[3] / 2)]], dtype=float),
            )
            solvers[side].set_robot_base_pose(
                base[:3], np.array([math.cos(base[3] / 2), 0, 0, math.sin(base[3] / 2)])
            )
            carriage = stage.GetPrimAtPath(f"/World/{side}_rail/carriage")
            carriage.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*base[:3]))

        def move_robot_base(side, destination):
            nonlocal cache_enabled, rail_cache_cursor
            original = np.asarray(CONFIG["fr3"]["bases"][side], dtype=float)
            destination = np.asarray(destination, dtype=float)
            low, high = CONFIG["rails"]["travel_limits_m"]
            if not low <= destination[1] <= high:
                raise RuntimeError("Rail target exceeds travel limits")
            states = joint_state()
            center, orientation = cube.get_world_pose()
            from isaacsim.core.utils.rotations import quat_to_rot_matrix
            cached_bases = None
            if cache_enabled and loaded_trajectory_cache is not None:
                rails = loaded_trajectory_cache.get("rail_transfers", [])
                if rail_cache_cursor < len(rails):
                    candidate = rails[rail_cache_cursor]
                    start_match = np.allclose(candidate.get("start_base_xyz_yaw"), original, atol=1e-5)
                    end_match = np.allclose(candidate.get("end_base_xyz_yaw"), destination, atol=1e-5)
                    if candidate.get("side") == side and start_match and end_match:
                        cached_bases = np.asarray(candidate.get("bases_xyz_yaw"), dtype=float)
                        if cached_bases.ndim != 2 or cached_bases.shape[1] != 4 or not np.all(np.isfinite(cached_bases)):
                            cached_bases = None
                        else:
                            rail_cache_cursor += 1
                            report["trajectory_cache"]["used_rail_transfers"] += 1
                    else:
                        cache_enabled = False
                        report["trajectory_cache"]["fallback_reason"] = "rail transfer start/goal mismatch"
            replay_rail = cached_bases is not None
            if replay_rail:
                bases = cached_bases
                count = len(bases)
            else:
                count = max(2, int(np.ceil(abs(destination[1] - original[1]) / 0.0005)))
                bases = [original + (destination - original) * (3 * t * t - 2 * t * t * t)
                         for t in np.linspace(0, 1, count)]
            def update_solver(base):
                solvers[side].set_robot_base_pose(base[:3], np.array([math.cos(base[3] / 2), 0, 0, math.sin(base[3] / 2)]))
            if not replay_rail:
                try:
                    for base in bases:
                        planning_update(f"{side}: rail collision checking")
                        update_solver(base)
                        issues = checker.check(states, cube_center=center, cube_rotation=quat_to_rot_matrix(orientation),
                                               cup_contacts=allowed_contacts, first_only=True)
                        if issues:
                            raise RuntimeError(f"Rail path is in collision: {issues[0]}")
                finally:
                    update_solver(original)
            for base in bases:
                set_robot_base_pose(side, base)
                for arm, q in states.items():
                    robots[arm].get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=q, joint_indices=joints[arm]))
                step()
            report["rail_collision_checked_samples"] = count
            trajectory_cache["rail_transfers"].append({
                "side": side,
                "start_base_xyz_yaw": original.tolist(),
                "end_base_xyz_yaw": destination.tolist(),
                "bases_xyz_yaw": [base.tolist() for base in bases],
                "samples": count,
            })

        def settle_cube(phase):
            last, max_speed = None, 0.0
            for _ in range(60):
                step()
                current, _ = cube.get_world_pose()
                current = np.asarray(current, dtype=float)
                if last is not None:
                    max_speed = max(max_speed, float(np.linalg.norm(current - last) * 120.0))
                last = current
            if max_speed > CONFIG["physics"]["settle_speed_threshold_m_s"]:
                raise RuntimeError(f"{phase} cube did not settle: max measured speed {max_speed:.3f} m/s")
            return max_speed

        def attach(side):
            path = f"/World/{side}_fr3/ABSideTool"
            assert stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI), "Attachment must reference a rigid body"
            anchor, flange_quat = force_recorder.tools[side].get_world_pose()
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
        # Capture attachment anchors only after the two driven flanges settle.
        step(180)
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        grasp_errors = {}
        for side, sign in (("left", -1), ("right", 1)):
            position, orientation = flanges[side].get_world_pose()
            rotation = quat_to_rot_matrix(orientation)
            tcp = position + rotation @ [0, sign * CONFIG["side_suction_tool"]["tcp_side_offset_m"],
                                         CONFIG["side_suction_tool"]["vertical_drop_m"]]
            grasp_errors[side] = float(np.linalg.norm(tcp - (np.asarray(supply) + [sign * half, 0, 0])))
        report["A_attachment_tcp_error_m"] = grasp_errors
        if max(grasp_errors.values()) > 0.002:
            raise RuntimeError(f"A grasp TCPs have not settled: {grasp_errors}")
        flange_gap = float(np.linalg.norm(np.asarray(flanges["right"].get_world_pose()[0])
                                         - np.asarray(flanges["left"].get_world_pose()[0])))
        report["A_attachment_flange_gap_m"] = flange_gap
        nominal_gap = CONFIG["cube_size_m"] + 2 * CONFIG["side_suction_tool"]["tcp_side_offset_m"]
        if abs(flange_gap - nominal_gap) > 0.002:
            raise RuntimeError(f"A grasp has not settled: flange gap {flange_gap:.6f} m, expected {nominal_gap:.6f} m")
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
            if name in ("A_lift", "A_over_front"):
                # Stationary suspended windows independently verify payload load.
                step(180)
            previous = center
        for side in robots:
            stage.RemovePrim(f"/World/Attachment_{side}")
        step(120)
        move(dual_target(previous, gap=0.02, continuous=True), duration=1.0)
        step(120)
        actual, _ = cube.get_world_pose()
        landing_error = float(np.linalg.norm(actual - previous))
        if landing_error > 0.015:
            raise RuntimeError(f"A release landing error {landing_error:.3f} m")
        record("A_release", "REACHED_WITH_ATTACHMENT_PROXY", landing_error_m=landing_error)
        allowed_contacts.discard("right")

        # B1: keep all links and tools collidable while the left cups push.
        half = CONFIG["cube_size_m"] / 2
        b1_target = np.asarray(CONFIG["task"]["phase_B1"]["target_center_m"], dtype=float)
        cube_start, _ = cube.get_world_pose()
        cube_start = np.asarray(cube_start, dtype=float)
        record("B1_contact", "RUNNING")
        left_contact = cube_start + np.array([0, -half, 0])
        # Start B1 on the same elbow branch as the validated waypoint plan;
        # Lula can otherwise pick the mirrored branch when A ends slightly
        # off the nominal landing center.
        b1_seed = None
        try:
            b1_plan = json.loads((ROOT / "validation" / "fr3_ab_ik_preflight.json").read_text())
            b1_seed = next(np.asarray(item["joint_positions_rad"], dtype=float)
                           for item in b1_plan.get("waypoints", [])
                           if item.get("arm") == "left" and item.get("waypoint") == "B1_start")
        except (FileNotFoundError, StopIteration, KeyError, TypeError, ValueError):
            pass
        left_q = solve("left", left_contact, FRONT_ROTATION, continuous=True, seed_override=b1_seed)
        move({"left": left_q}, duration=1.0)
        step(12)
        b1_push_start, _ = cube.get_world_pose()
        b1_push_start = np.asarray(b1_push_start, dtype=float)
        contact_path("left", b1_push_start, b1_target, FRONT_ROTATION, np.array([0, -half, 0]), "B1")
        b1_actual, _ = cube.get_world_pose()
        b1_progress = float(b1_actual[1] - cube_start[1])
        if b1_progress < 0.20:
            raise RuntimeError(f"B1 cube did not advance enough: {b1_progress:.3f} m")
        # Keep the left cup on the front face.  During B2 it is the hold
        # arm that presses the cube against the back wall while the right arm
        # translates it along -X; retracting here lets the cube peel off the
        # wall as soon as the right tool makes contact.
        record("B1_push", "REACHED_WITH_PHYSX_CONTACT", progress_m=b1_progress,
               back_wall_hold_armed=True)

        b2_plan = json.loads((ROOT / "validation" / "fr3_b2_collision_plan.json").read_text())
        if b2_plan.get("status") != "PASS_COLLISION_CHECKED_B2_PLAN":
            raise RuntimeError("B2 collision-constrained plan is unavailable")
        if (not np.allclose(b2_plan["cube_initial_m"], CONFIG["task"]["phase_B1"]["target_center_m"])
                or not np.allclose(b2_plan["cube_final_m"], CONFIG["task"]["phase_B2"]["target_center_m"])
                or not np.isclose(b2_plan["right_base_contact_m"][1], CONFIG["task"]["phase_B2"]["rail_assist"]["base_y_m"])):
            raise RuntimeError("B2 plan differs from the scene configuration; regenerate it before execution")
        b2_rotation = np.asarray(b2_plan["right_rotation_world"])
        b2_target = np.asarray(CONFIG["task"]["phase_B2"]["target_center_m"], dtype=float)
        b2_transit_z = float(b2_plan.get("transit_z_m", 1.05))
        record("B2_high_approach", "RUNNING")
        right_seed = joint_state()["right"]
        flange, rotation = solvers["right"].compute_forward_kinematics("fr3_link8", right_seed)
        right_tcp = np.asarray(flange).reshape(3) + rotation @ [0, CONFIG["side_suction_tool"]["tcp_side_offset_m"],
                                                              CONFIG["side_suction_tool"]["vertical_drop_m"]]
        lift_cache = take_cached_phase(
            "B2_lift_from_measured_state", joint_state(),
            validator=lambda entry, samples: (
                np.linalg.norm(cached_side_tcp(entry, samples, "right")
                               - np.array([right_tcp[0], right_tcp[1], b2_transit_z])) <= 0.025))
        if lift_cache is not None:
            execute_joint_path(
                {"right": np.asarray(lift_cache["waypoints_rad"]["right"], dtype=float)},
                "B2_lift_from_measured_state", prefetched_cache=lift_cache)
        else:
            lift_targets = []
            for z in np.linspace(right_tcp[2], b2_transit_z, 60)[1:]:
                target = solve("right", np.array([right_tcp[0], right_tcp[1], z]), b2_rotation,
                               continuous=True, seed_override=right_seed)
                if np.max(np.abs(target - right_seed)) > 0.35:
                    raise RuntimeError("B2 elevated approach changed IK branch")
                lift_targets.append(target)
                right_seed = np.asarray(target)
            execute_joint_path({"right": lift_targets}, "B2_lift_from_measured_state")
        move({"right": np.asarray(b2_plan["rail"]["hold_right_q_rad"])})
        right_base = list(CONFIG["fr3"]["bases"]["right"])
        right_base[1] = b2_plan["right_base_contact_m"][1]
        record("B2_rail_transfer", "RUNNING", base_y_m=right_base[1])
        move_robot_base("right", right_base)
        record("B2_rail_transfer", "REACHED", base_y_m=right_base[1])
        allowed_contacts.add("right")
        execute_joint_path({"right": b2_plan["phases"]["descent"]["joint_path_rad"]}, "B2_descent")
        execute_joint_path({"right": b2_plan["phases"]["approach"]["joint_path_rad"]},
                           "B2_suction_plane_approach")
        step(60)
        b2_start, _ = cube.get_world_pose()
        b2_start = np.asarray(b2_start, dtype=float)
        left_b2_start_q = joint_state()["left"]
        contact = b2_plan["phases"]["contact"]
        def cached_b2_push_matches(entry, _samples):
            centers = np.asarray(entry.get("cube_center_path_m"), dtype=float)
            return (centers.ndim == 2 and centers.shape[1] == 3 and len(centers) > 1
                    and np.linalg.norm(centers[0] - b2_start) <= 0.035
                    and np.linalg.norm(centers[-1] - b2_target) <= 0.035)

        b2_push_cache = take_cached_phase(
            "B2_push", joint_state(), validator=cached_b2_push_matches)
        # Re-solve the support arm from its measured B1 pose.  The offline
        # hold path starts from a nominal preflight branch whose manifold is
        # only 1.978 mm above the floor after runtime tracking drift.
        if b2_push_cache is not None:
            b2_left_path = np.asarray(b2_push_cache["waypoints_rad"]["left"], dtype=float)
            b2_right_path = np.asarray(b2_push_cache["waypoints_rad"]["right"], dtype=float)
        else:
            b2_left_seed = left_b2_start_q.copy()
            b2_left_path = []
            b2_centers = np.linspace(b2_start, b2_target, len(contact["right_joint_path_rad"]))
            for center in b2_centers:
                target = np.asarray(solve("left", center + np.array([0, -half, 0.004]), FRONT_ROTATION,
                                           continuous=True, seed_override=b2_left_seed), dtype=float)
                if np.max(np.abs(target - b2_left_seed)) > 0.35:
                    raise RuntimeError("left B2 hold IK branch changed")
                b2_left_path.append(target)
                b2_left_seed = target
            b2_right_path = contact["right_joint_path_rad"]
        b2_hold_errors = []
        b2_right_contact_errors = []
        def b2_cube_center_path(samples, _times):
            # Derive the horizontal contact pose from the planned right TCP,
            # while keeping the measured cube height fixed for floor checks.
            centers = []
            for q in samples["right"]:
                planning_update("B2: synchronized Cube path")
                flange, rotation = solvers["right"].compute_forward_kinematics("fr3_link8", q)
                tcp = np.asarray(flange).reshape(3) + np.asarray(rotation) @ np.array([
                    0, CONFIG["side_suction_tool"]["tcp_side_offset_m"],
                    CONFIG["side_suction_tool"]["vertical_drop_m"]])
                centers.append([tcp[0] - half, tcp[1], b2_start[2]])
            return np.asarray(centers)

        def check_b2_hold(index, count):
            actual_center, _ = cube.get_world_pose()
            flange, rotation = solvers["left"].compute_forward_kinematics(
                "fr3_link8", joint_state()["left"])
            tcp_offset = np.array([0, -CONFIG["side_suction_tool"]["tcp_side_offset_m"],
                                   CONFIG["side_suction_tool"]["vertical_drop_m"]])
            tcp = np.asarray(flange).reshape(3) + np.asarray(rotation) @ tcp_offset
            error = float(np.linalg.norm(tcp - (np.asarray(actual_center) + [0, -half, 0.004])))
            b2_hold_errors.append(error)
            if error > 0.02:
                raise RuntimeError(f"left B2 hold contact drifted {error:.3f} m")
            right_flange, right_rotation = solvers["right"].compute_forward_kinematics(
                "fr3_link8", joint_state()["right"])
            right_tcp = np.asarray(right_flange).reshape(3) + np.asarray(right_rotation) @ np.array([
                0, CONFIG["side_suction_tool"]["tcp_side_offset_m"],
                CONFIG["side_suction_tool"]["vertical_drop_m"]])
            right_error = float(np.linalg.norm(right_tcp - (np.asarray(actual_center) + [half, 0, 0])))
            b2_right_contact_errors.append(right_error)
            if right_error > 0.025:
                raise RuntimeError(f"right B2 contact drifted {right_error:.3f} m")

        execute_joint_path({"left": b2_left_path, "right": b2_right_path},
                           "B2_push", check_step=check_b2_hold,
                           cube_center_path=b2_cube_center_path,
                           prefetched_cache=b2_push_cache)
        # Let the synchronized endpoint settle while both contact faces stay
        # commanded.  This removes the small position-control lag before the
        # right arm releases the side face.
        step(60)
        b2_actual, _ = cube.get_world_pose()
        b2_settle_issues = checker.check(joint_state(), cube_center=np.asarray(b2_actual),
                                         cube_rotation=np.eye(3), cup_contacts=allowed_contacts,
                                         margin_m=0.0, first_only=True)
        if b2_settle_issues:
            raise RuntimeError(f"B2 contact settle collision: {b2_settle_issues[0]}")
        left_b2_end_q = robots["left"].get_joint_positions(joint_indices=joints["left"])
        left_b2_joint_motion = float(np.max(np.abs(left_b2_end_q - left_b2_start_q)))
        if left_b2_joint_motion < 0.01:
            raise RuntimeError("left B2 hold arm did not move with the cube")
        b2_progress = float(b2_start[0] - b2_actual[0])
        if b2_progress < 0.15:
            raise RuntimeError(f"B2 cube did not advance enough: {b2_progress:.3f} m")
        right_retract = solve("right", b2_actual + np.array([half + 0.035, 0, 0]), b2_rotation, continuous=True)
        move({"right": right_retract}, duration=1.0)
        allowed_contacts.discard("right")
        # Let the left hold keep the cube seated while the active right tool
        # releases, then release and retract the hold arm after the cube is
        # confirmed static.
        settle_cube("B2_right_release")
        left_retract = solve("left", b2_actual + np.array([0, -half - 0.035, 0]), FRONT_ROTATION, continuous=True)
        move({"left": left_retract}, duration=1.0)
        allowed_contacts.clear()
        max_speed = settle_cube("B2")
        record("B2_push_release", "REACHED_WITH_PHYSX_CONTACT", progress_m=b2_progress,
               max_settle_speed_m_s=max_speed, back_wall_hold_verified=True,
               left_hold_joint_motion_rad=left_b2_joint_motion,
               left_hold_max_contact_error_m=max(b2_hold_errors, default=0.0),
               right_contact_max_error_m=max(b2_right_contact_errors, default=0.0))
        report["status"] = "PASS_B1_B2_PROXY_CONTACT"
        report["ab_status"] = "PASS_PHYSX_CONTACT_PROXY"
        report["blocker"] = "Real Surface Gripper remains unvalidated"
        record("B1_B2", "PASS_PHYSX_CONTACT_PROXY", reason=report["blocker"])
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
        if vision_task is not None:
            if vision_task.done():
                async_error = vision_task.exception()
                if async_error is not None:
                    report["status"] = "STOPPED"
                    report["ab_status"] = "BLOCKED"
                    report.setdefault("error", str(async_error))
            else:
                vision_task.cancel()
        if vision_comparison is not None:
            report["vision"]["comparison_summary"] = vision_comparison.finalize()
        if force_recorder is not None:
            report["wrist_force_sensors"] = force_recorder.close()
            force_recorder.metadata["task_status"] = report["status"]
            (output_directory / "metadata.json").write_text(
                json.dumps(force_recorder.metadata, indent=2, allow_nan=False) + "\n")
        if contact_monitor is not None:
            report["physx_contacts"] = contact_monitor.report()
            contact_monitor.close()
        trajectory_cache["status"] = (
            "COMPLETE" if report.get("status") == "PASS_B1_B2_PROXY_CONTACT" else "INCOMPLETE"
        )
        trajectory_cache["runtime_report"] = str(report_path)
        cache_output.parent.mkdir(parents=True, exist_ok=True)
        cache_output.write_text(
            json.dumps(trajectory_cache, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print("FR3_AB_RESULT " + json.dumps(report, ensure_ascii=False), flush=True)
        if world is not None:
            world.pause()
        if vision_viewport is not None and app.is_running():
            from omni.kit.viewport.utility import capture_viewport_to_file
            capture = capture_viewport_to_file(vision_viewport, str(output_directory / "overview.png"))
            for _ in range(12):
                app.update()
        if not args.headless and not args.close_on_finish:
            print("Task stopped. Inspect the scene; close the window to exit.", flush=True)
            while app.is_running():
                app.update()
        app.close()
    return 0 if report["status"] in ("BLOCKED_B_RUNTIME", "PASS_B1_B2_PROXY_CONTACT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
