"""Automatically execute the new FR3 scene in Isaac Sim.

A uses contact-gated, breakable Isaac Sim Surface Gripper constraints.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from fr3_contact_control import NormalForceController, oriented_wall_gap
from fr3_surface_gripper import ATTACHMENT_PARAMETERS, FR3SurfaceGrippers

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "fr3_four_wall_scene_config.json").read_text())
TRAJECTORY_CACHE = ROOT / "validation" / "fr3_trajectory_cache.json"
FORCE_TRAJECTORY_CACHE = ROOT / "validation" / "fr3_force_control_trajectory_cache.json"
FULL_FORCE_TRAJECTORY_CACHE = ROOT / "validation" / "fr3_full_force_trajectory_cache.json"
A_POSTURE_GUIDES = ROOT / "validation" / "fr3_a_posture_guides.json"
FORCE_CONTROLLER_REVISION = "B2_wall_and_neighbor_seating_loaded_preload_v1"
B2_CONTROLLER_PARAMETERS = {
    "type": "absolute_normal_position_admittance_with_Cartesian_position_tracking",
    "left_target_force_N": 2.0, "gain_m_per_Ns": 2e-5, "unload_gain_m_per_Ns": 1e-4,
    "normal_position_speed_limit_m_s": 0.0002, "normal_position_max_offset_m": .002,
    "overload_limit_N": 10.0, "camera_axis_gate": "loaded_wrist_normal_above_0.1N",
    "push_duration_s": 15.0, "retreat_duration_s": 3.0, "seating_gap_tolerance_m": .0005,
    "seating_guard_source": "wrist_force_and_encoder_contact_geometry",
    "touch_seek_step_m": 5e-6, "touch_contact_force_N": .20, "touch_confirmation_frames": 6,
    "wall_seating_probe": {"entry_force_N": 6., "samples": 3,
                           "mean_force_N": 4., "target_uncertainty_m": .0005},
    "neighbor_seating_probe": {"entry_force_N": 8., "samples": 30,
                               "minimum_force_N": 6., "target_uncertainty_m": .005},
    "support_approach_preload_N": 4., "active_sync_loaded_force_N": .5,
    "release": "force ramp to zero, then straight outward Cartesian motion",
}
A_TRANSFER_REVISION = "shared_rigid_Cartesian_translation_velocity_v3_continuous_command_reference"
A_FORCE_LIMITS = {
    "required_phases": ["A_approach_transit", "A_attach", "A_transfer", "A_lift",
                        "A_over_front", "A_release"],
    "max_force_N": 30.0, "max_tcp_torque_Nm": 3.0, "max_internal_force_N": 15.0,
}
B1_FORCE_LIMITS = {"required_phases": ["B1"], "max_force_N": 30.0, "max_tcp_torque_Nm": 3.0}


def compatible_force_cache(candidate):
    return (isinstance(candidate, dict) and candidate.get("status") == "COMPLETE"
            and candidate.get("controller_revision") == FORCE_CONTROLLER_REVISION
            and candidate.get("controller_parameters") == B2_CONTROLLER_PARAMETERS
            and candidate.get("scene_config") == CONFIG
            and candidate.get("force_validation_status") == "PASS_B2_FORCE_CONTROL"
            and isinstance(candidate.get("phases"), list))


def compatible_full_force_cache(candidate):
    return (compatible_force_cache(candidate)
            and candidate.get("a_transfer_revision") == A_TRANSFER_REVISION
            and candidate.get("a_force_limits") == A_FORCE_LIMITS
            and candidate.get("b1_force_limits") == B1_FORCE_LIMITS
            and candidate.get("task_force_status") == "FULL_FLOW_FORCE_PASS")

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
    from pxr import Gf, UsdGeom, UsdPhysics

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
    # Hiding the stock hand leaves 0.73 kg of ghost inertia; remove its bodies and joints.
    for name in ("fr3_link8/fr3_hand_joint", "fr3_hand", "fr3_leftfinger",
                 "fr3_rightfinger", "fr3_hand_tcp"):
        prim = stage.GetPrimAtPath(f"{root}/{name}")
        if prim.IsValid():
            prim.SetActive(False)
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
    global CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run the same simulation without a window")
    parser.add_argument("--close-on-finish", action="store_true", help="Close instead of retaining the result in the GUI")
    parser.add_argument("--output-dir", type=Path, help="Keep this run's artifacts separate from earlier experiments")
    parser.add_argument("--five-cube-row", action="store_true", help="Load and pack five Cubes in alternating order")
    parser.add_argument("--collision-workers", type=int, help="Processes for final path validation (1 for serial)")
    args = parser.parse_args()
    workers = args.collision_workers if args.collision_workers is not None else (4 if args.five_cube_row else 1)
    if workers < 1:
        parser.error("--collision-workers must be positive")
    if args.five_cube_row:
        from fr3_five_cube_row import row_config
        CONFIG = row_config(CONFIG)
    output_directory = args.output_dir.resolve() if args.output_dir else ROOT / "validation" / (
        "five_cube_row" if args.five_cube_row else "idea_one_contact_state")
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "runner_report.json"
    cache_output = output_directory / "trajectory_cache.json"
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "width": 1400, "height": 900})
    print("FR3_AB_APP_READY", flush=True)
    report = {
        "scene_id": CONFIG["scene_id"], "status": "RUNNING", "ab_status": "UNVERIFIED",
        "scene_parameters": json.loads(json.dumps(CONFIG)),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for pattern in ("*.py", "*.json") for path in sorted(ROOT.glob(pattern))},
        "scope": "automatic_articulation_visual_demo", "attachment_model": "Isaac_Sim_Surface_Gripper",
        "attachment_parameters": ATTACHMENT_PARAMETERS,
        "surface_gripper_validated": False, "collision_safety_validated": False,
        "proxy_robot_collisions_disabled": False, "robot_collisions_enabled": True,
        "stock_gripper_bodies_removed": True,
        "full_arm_planning_check": "official_Lula_spheres_and_full_tool_OBBs",
        "unintended_physx_contact_guard": True, "gravity_compensation": True,
        "b2_right_body_collision_disabled_for_contact": False, "phases": [],
        "control_input_boundary": {
            "target_sources": ["fresh RGB ArUco on calibrated support planes",
                               "same-step joint encoders and flange-to-TCP calibration",
                               "wrist F/T confirmed contact and encoder propagation"],
            "cube_truth": "reference publisher, event logs and stopped-state diagnostics only",
            "pose_loss": "fresh observation required at phase transitions; missing detections stop acquisition",
        },
        "remaining_model_assumptions": [
            "Known Cube dimensions and surveyed static wall/support geometry",
            "Planar yaw-only resting pose on a calibrated support plane",
            "One breakable central vacuum constraint represents each four-cup array",
            "Last observed yaw propagates through occlusion; force and FK alone do not measure slip yaw",
            "Ideal joint encoders and noiseless simulated wrist wrench",
            "Simulator model gravity compensation and commanded rail base motion",
            "Physical contact callbacks provide independent collision safety stops",
        ],
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
        "controller_revision": FORCE_CONTROLLER_REVISION,
        "controller_parameters": B2_CONTROLLER_PARAMETERS,
        "scene_config": CONFIG,
        "force_validation_status": "UNVERIFIED",
        "a_transfer_revision": A_TRANSFER_REVISION,
        "a_force_limits": A_FORCE_LIMITS, "b1_force_limits": B1_FORCE_LIMITS,
        "task_force_status": "UNVERIFIED",
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
    contact_experiment = None
    parallel_validator = None
    try:
        import numpy as np
        import omni.client
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, interface_config_loader
        from isaacsim.storage.native import get_assets_root_path
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

        enable_extension("isaacsim.robot.surface_gripper")

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
        active_cube_path = "/World/Cube_1" if args.five_cube_row else "/World/Cube"
        prim = _box(stage, active_cube_path, CONFIG["supply_cube_center_m"], [CONFIG["cube_size_m"]] * 3,
                    collision=True, dynamic=True, mass=CONFIG["physics"]["cube_mass_kg"])
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.35, 0.05)])
        if args.five_cube_row:
            from fr3_cube_supply import add_cube_marker
            add_cube_marker(stage, active_cube_path, CONFIG["cube_size_m"],
                            CONFIG["vision"]["aruco_marker_size_m"], CONFIG["cube_sequence"][0]["marker_id"])
        else:
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
        supply_prim = UsdGeom.Camera.Define(stage, "/World/SupplyCamera")
        supply_prim.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*vision_cfg["supply_camera_position_m"]),
            Gf.Vec3d(*vision_cfg["supply_camera_look_at_m"]),
            Gf.Vec3d(0, 0, 1),
        ).GetInverse())
        supply_prim.CreateFocalLengthAttr(1.93)
        supply_prim.CreateHorizontalApertureAttr(3.896)
        supply_prim.CreateVerticalApertureAttr(2.453)
        supply_prim.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
        supply_camera = world.scene.add(IsaacCamera(
            prim_path="/World/SupplyCamera", name="supply_camera",
            frequency=vision_cfg["frequency_hz"], resolution=tuple(vision_cfg["resolution"]),
        ))
        supply_camera.set_projection_type("pinhole")
        if not args.headless:
            import omni.ui as ui
            status_window = ui.Window("FR3 automatic task", width=580, height=230)
            with status_window.frame:
                with ui.VStack():
                    status_label = ui.Label("Loading FR3...", word_wrap=True)
                    force_label = ui.Label("Wrist F/T: waiting for physics", word_wrap=True)
        robots, solvers, joints, flanges = {}, {}, {}, {}
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("FR3")
        for side, base in CONFIG["fr3"]["bases"].items():
            robots[side] = world.scene.add(SingleArticulation(f"/World/{side}_fr3", name=f"{side}_robot"))
            solver = LulaKinematicsSolver(**kinematics_config)
            solver.set_robot_base_pose(np.array(base[:3]), np.array([math.cos(base[3] / 2), 0, 0, math.sin(base[3] / 2)]))
            assert solver.get_joint_names() == [f"fr3_joint{i}" for i in range(1, 8)]
            solvers[side] = solver
            flanges[side] = SingleXFormPrim(f"/World/{side}_fr3/fr3_link8")
        cube = world.scene.add(SingleRigidPrim(active_cube_path, name="task_cube"))
        print("FR3_AB_INITIALIZING_PHYSICS", flush=True)
        contact_monitor = PhysxContactMonitor(stage)
        if args.five_cube_row:
            contact_monitor.set_active_cube(active_cube_path)
        loaded_cache_path = TRAJECTORY_CACHE
        for cache_path in (FULL_FORCE_TRAJECTORY_CACHE, FORCE_TRAJECTORY_CACHE, TRAJECTORY_CACHE):
            try:
                candidate = json.loads(cache_path.read_text())
                if cache_path == FULL_FORCE_TRAJECTORY_CACHE:
                    valid = compatible_full_force_cache(candidate)
                elif cache_path == FORCE_TRAJECTORY_CACHE:
                    valid = compatible_force_cache(candidate)
                else:
                    valid = (candidate.get("status") == "COMPLETE"
                             and candidate.get("scene_id") == CONFIG["scene_id"]
                             and isinstance(candidate.get("phases"), list))
                if valid:
                    loaded_trajectory_cache, loaded_cache_path, cache_enabled = candidate, cache_path, True
                    break
            except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError, AttributeError):
                continue
        world.reset()
        if vision_camera is not None:
            vision_camera.resume()
        supply_camera.pause()
        if vision_viewport is not None:
            vision_viewport.set_active_camera("/World/Camera")
            vision_viewport.resolution = (1400, 900)
            assert vision_camera.get_render_product_path() != vision_viewport.get_render_product_path()
            report["vision"]["overview_resolution"] = list(vision_viewport.resolution)
            report["vision"]["independent_render_product"] = True
        contact_drive_scale = float(CONFIG["fr3"]["contact_drive_gain_scale"])
        if not np.isfinite(contact_drive_scale) or contact_drive_scale <= 0:
            raise ValueError("Contact drive gain scale must be positive and finite")
        solver_position = CONFIG["physics"]["articulation_solver_position_iterations"]
        solver_velocity = CONFIG["physics"]["articulation_solver_velocity_iterations"]
        if (not isinstance(solver_position, int) or solver_position < 1
                or not isinstance(solver_velocity, int) or solver_velocity < 0):
            raise ValueError("Solver counts require positive position and nonnegative velocity integers")
        cube_api = PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(active_cube_path))
        report["physics_solver"] = {
            "scene_type": world.get_physics_context().get_solver_type(),
            "requested_articulation_iterations": {"position": solver_position, "velocity": solver_velocity},
            "arms": {},
            "cube_default_iterations": {"position": cube_api.GetSolverPositionIterationCountAttr().Get(),
                                        "velocity": cube_api.GetSolverVelocityIterationCountAttr().Get()},
            "reason": "PhysX5.3+ TGS compatibility: official32/16 excess velocity iterations become44/4.",
        }
        original_arm_gains, original_arm_efforts, drive_types = {}, {}, {}
        for side, robot in robots.items():
            if robot.num_dof != 7:
                raise RuntimeError(f"{side}: expected seven arm DoFs after stock gripper removal")
            original_iterations = {"position": int(robot.get_solver_position_iteration_count()),
                                   "velocity": int(robot.get_solver_velocity_iteration_count())}
            robot.set_solver_position_iteration_count(solver_position)
            robot.set_solver_velocity_iteration_count(solver_velocity)
            actual_iterations = {"position": int(robot.get_solver_position_iteration_count()),
                                 "velocity": int(robot.get_solver_velocity_iteration_count())}
            if actual_iterations != {"position": solver_position, "velocity": solver_velocity}:
                raise RuntimeError(f"{side}: physics solver iteration readback differs from configuration")
            report["physics_solver"]["arms"][side] = {"original_asset": original_iterations,
                                                      "actual": actual_iterations}
            joints[side] = np.array([robot.get_dof_index(f"fr3_joint{i}") for i in range(1, 8)])
            controller = robot.get_articulation_controller()
            original_arm_gains[side] = [
                np.asarray(values)[joints[side]].copy() for values in controller.get_gains()]
            original_arm_efforts[side] = np.asarray(controller.get_max_efforts())[joints[side]].copy()
            drive_types[side] = [
                UsdPhysics.DriveAPI(stage.GetPrimAtPath(
                    f"/World/{side}_fr3/fr3_link{index-1}/fr3_joint{index}"), "angular").GetTypeAttr().Get()
                for index in range(1, 8)]

            robot.set_joint_positions(np.array(CONFIG["fr3"]["official_start_q_rad"]), joint_indices=joints[side])
            robot.get_articulation_controller().apply_action(ArticulationAction(
                joint_positions=np.array(CONFIG["fr3"]["official_start_q_rad"]), joint_indices=joints[side]))
        force_recorder = WristForceRecorder(robots, joints, CONFIG, output_directory, solvers=solvers)
        force_recorder.metadata["physics_solver"] = report["physics_solver"]
        report["joint_drive_model"] = {
            "contact_gain_scale": contact_drive_scale, "free_space_gain_scale": 1.,
            "drive_types": drive_types, "scope": "seven arm joints with original effort limits retained",
            "hardware_calibrated": False,
            "assumption": "uncalibrated simulation contact compliance with original gains for A transport",
        }
        force_recorder.metadata["joint_drive_model"] = report["joint_drive_model"]
        nominal_targets = {side: np.array(CONFIG["fr3"]["official_start_q_rad"], dtype=float)
                           for side in robots}
        applied_targets = {side: q.copy() for side, q in nominal_targets.items()}
        applied_velocities = {side: np.zeros(7) for side in robots}
        left_hold = None
        support_side = "left"
        push_side = "right"
        push_sign = -1
        left_hold_z = None
        left_hold_rotation = None
        current_task = (CONFIG["cube_sequence"][0] if args.five_cube_row else
                        {"cube_id": 1, "support_side": "left", "push_side": "right", "push_direction": -1,
                         "contact_target": "wall_left",
                         "goal_center_m": CONFIG["task"]["phase_B2"]["target_center_m"]})
        rail_states = {side: np.asarray(base, dtype=float).copy()
                       for side, base in CONFIG["fr3"]["bases"].items()}
        last_force_sample = None
        force_gain = B2_CONTROLLER_PARAMETERS["gain_m_per_Ns"]
        force_limits = {
            "role_source": "packing_sequence",
            "held_phases": ["B2_hold", "B2_high_approach", "B2_lift_from_measured_state",
                            "B2_lift_from_measured_state_transit", "B2_rail_transfer", "B2_descent",
                            "B2_suction_plane_approach", "B2_touch", "B2_contact_sync", "B2_push",
                            "B2_push_seat", "B2_push_unload", "B2_push_retreat", "B2_push_arm_release"],
            "held_window": {"start_phase": "B2_hold", "start_status": "FORCE_CONFIRMED",
                            "end_phase": "B2_support_unload", "end_status": "RUNNING"},
            "release_phases": {"push": ["B2_push_retreat"], "support": ["B2_support_retreat"]},
            "normal_force_min_N": 0.1, "normal_force_max_N": 6.0,
            "min_contact_coverage": 0.98, "max_dropout_s": 0.05,
            "max_force_N": 10.0, "max_tcp_torque_Nm": 1.0,
            "max_inward_m": 0.00002, "minimum_retreat_m": 0.03,
        }
        force_recorder.metadata["b2_force_control_validation"] = force_limits
        force_recorder.metadata["b2_controller"] = B2_CONTROLLER_PARAMETERS
        force_recorder.metadata["a_force_control_validation"] = A_FORCE_LIMITS
        force_recorder.metadata["a_transfer_revision"] = A_TRANSFER_REVISION
        force_recorder.metadata["a_velocity_feedforward"] = True
        force_recorder.metadata["joint_drive_gains"] = {
            side: {name: np.asarray(values)[joints[side]].tolist()
                   for name, values in zip(("stiffness", "damping"),
                       robot.get_articulation_controller().get_gains())}
            for side, robot in robots.items()
        }
        force_recorder.metadata["b1_force_control_validation"] = B1_FORCE_LIMITS
        trajectory_cache["execution_overlay"] = (
            "B2 nominal joint samples have online force correction; actual position targets are in wrench.jsonl.gz control")
        calibration_path = ROOT / "validation" / "idea_one_wrist_ft_load_check.json"
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text())
            force_recorder.metadata["calibration_passed"] = calibration.get("status") == "PASS"
            force_recorder.metadata["calibration_report"] = str(calibration_path.relative_to(ROOT))
        tcp_calibration_path = ROOT / "validation" / "idea_one_tcp_wrist_ft_load_check.json"
        if tcp_calibration_path.exists():
            tcp_calibration = json.loads(tcp_calibration_path.read_text())
            force_recorder.metadata["tcp_calibration_passed"] = tcp_calibration.get("status") == "PASS"
            force_recorder.metadata["tcp_calibration_report"] = str(tcp_calibration_path.relative_to(ROOT))
        print("FR3_AB_PHYSICS_READY", flush=True)
        checker = FR3CollisionChecker(CONFIG, solvers)
        if workers > 1:
            from fr3_parallel_collision import ParallelPathValidator
            parallel_validator = ParallelPathValidator(workers, CONFIG, kinematics_config)
        report["path_validation_timing"] = []
        allowed_contacts = {"left", "right"}
        contact_monitor.allowed_cube_contacts = allowed_contacts
        # The previous plans encode Cartesian endpoints from a different pose source.
        if cache_enabled:
            loaded_trajectory_cache, cache_enabled = None, False
        report["trajectory_cache"] = {
            "loaded": bool(cache_enabled),
            "path": str(loaded_cache_path.relative_to(ROOT)),
            "used_phases": [],
            "used_rail_transfers": 0,
            "fallback_reason": "legacy trajectory targets predate sensor-based localization",
        }
        tick = 0
        supply_transition = False
        grippers = FR3SurfaceGrippers(force_recorder.tcp_offsets)
        seals_required = False
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

        from fr3_vision import ArucoCubeEstimator, PlanarPoseMedian
        from fr3_vision_comparison import VisionComparison, match_reference

        vision_period = max(1, int(round(120 / vision_cfg["frequency_hz"])))
        camera_matrix = np.asarray(vision_camera.get_intrinsics_matrix(), dtype=float)
        camera_to_world = np.linalg.inv(np.asarray(vision_camera.get_view_matrix_ros(), dtype=float))
        def make_vision_estimator(intrinsics, optical_to_world):
            return ArucoCubeEstimator(
                intrinsics, optical_to_world, marker_id=vision_cfg["aruco_marker_id"],
                marker_size_m=vision_cfg["aruco_marker_size_m"],
                cube_size_m=CONFIG["cube_size_m"], marker_surface_offset_m=0.0085,
                dictionary_name=vision_cfg["aruco_dictionary"],
            )
        vision_estimator = make_vision_estimator(camera_matrix, camera_to_world)
        supply_estimator = make_vision_estimator(
            supply_camera.get_intrinsics_matrix(), np.linalg.inv(supply_camera.get_view_matrix_ros()),
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
            "camera_uses_cube_truth": False, "control_state_source": "RGB, encoder FK and wrist F/T",
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
        latest_camera_observation = None
        control_center = control_rotation = carry_offsets = None
        push_faces = {}
        latched_contact_axis = None
        contact_plan_rotation = None
        supported_pose_time = None
        supported_pose_filter = PlanarPoseMedian()
        neighbor_filters = {}
        live_neighbors = {}
        report["vision"]["supported_pose_filter"] = {
            "type": "causal_planar_median", "distinct_frames": 3,
            "timestamp": "middle optical exposure",
        }
        report["target_observations"] = []
        from fr3_contact_state_validation import ContactStateExperiment
        contact_experiment = ContactStateExperiment(output_directory / "contact_states", CONFIG)

        def sample_vision(render=True):
            nonlocal latest_camera_observation, control_center, control_rotation, supported_pose_time
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
                    try:
                        task = current_task or {}
                    except NameError:
                        task = {}
                    task_cube_id = task.get("cube_id")
                    task_contact_target = task.get("contact_target")
                    task_goal = task.get("goal_center_m") or [0.]
                    task_goal_x = task_goal[0]
                    neighbor_name = ("Cube_8" if task_cube_id == 10 else task_contact_target)
                    neighbor_phase = (contact_monitor.phase in ("B2_push", "B2_push_seat", "B2_push_unload")
                                      or (task_cube_id == 10
                                          and contact_monitor.phase.startswith("upper_B")))
                    if (neighbor_phase and neighbor_name is not None
                            and control_center is not None
                            and abs(control_center[0] - task_goal_x) < .025
                            and 0 <= receive_time - capture_time <= .20):
                        neighbor_index = int(neighbor_name.removeprefix("Cube_")) - 1
                        active_id = vision_estimator.marker_id
                        try:
                            vision_estimator.marker_id = CONFIG["cube_sequence"][neighbor_index]["marker_id"]
                            neighbor = vision_estimator.estimate(rgba.astype(np.uint8, copy=False))
                        finally:
                            vision_estimator.marker_id = active_id
                        pose = vision_estimator.resting_pose(neighbor, CONFIG["floor_top_m"])
                        if pose is not None:
                            if neighbor_name not in neighbor_filters:
                                neighbor_filters[neighbor_name] = PlanarPoseMedian()
                            filtered = neighbor_filters[neighbor_name].update(frame_id, capture_time, *pose)
                            if filtered is not None:
                                live_neighbors[neighbor_name] = filtered
            latest_camera_observation = {
                "status": observation["status"], "position_m": observation["cube_position_m"],
                "cube_rotation_world": observation.get("cube_rotation_world"),
                "capture_time_s": capture_time, "received_time_s": receive_time, "frame_id": frame_id,
            }
            if push_faces and receive_time - capture_time <= .20:
                raw_rotation = np.asarray(observation.get("cube_rotation_world"), dtype=float)
                supported_pose = None
                # A highly tilted fitted marker is inconsistent with the floor-supported pose.
                if (raw_rotation.shape == (3, 3) and np.all(np.isfinite(raw_rotation))
                        and raw_rotation[2, 2] >= math.cos(math.radians(35))):
                    supported_pose = vision_estimator.resting_pose(observation, CONFIG["floor_top_m"])
                elif observation["status"] == "detected":
                    latest_camera_observation["control_pose_rejection"] = "marker_plane_tilt"
                if supported_pose is not None:
                    filtered_pose = supported_pose_filter.update(
                        frame_id, capture_time, *supported_pose, reference_rotation=contact_plan_rotation)
                    if filtered_pose is not None:
                        supported_center, supported_rotation, filtered_time = filtered_pose
                        if (contact_plan_rotation is not None
                                and contact_plan_rotation[:, 0] @ supported_rotation[:, 0] < math.cos(math.radians(3))):
                            raise RuntimeError("RGB observed more than 3 degrees of Cube rotation; contact replan required")
                        control_rotation = supported_rotation
                        contact_axes = {
                            axis for side, (axis, outward, _) in push_faces.items()
                            if np.dot(last_force_sample[side]["tcp_gravity_corrected_world"][:3], outward) > .1
                        }
                        if latched_contact_axis is not None:
                            contact_axes.add(latched_contact_axis)
                        for axis in set(range(3)) - contact_axes:
                            control_center[axis] = supported_center[axis]
                        supported_pose_time = filtered_time
                        latest_camera_observation["supported_position_m"] = supported_center.tolist()
                        latest_camera_observation["supported_rotation_world"] = supported_rotation.tolist()
                        latest_camera_observation["supported_capture_time_s"] = filtered_time
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

        def observe_resting_cube(phase, camera, estimator, support_z):
            nonlocal control_center, control_rotation, supported_pose_time
            estimates, frames, exposure_ids = [], set(), []
            supported_pose_filter.reset()
            started = float(world.current_time)
            camera.resume()
            diagnostics = []
            try:
                for _ in range(30):
                    step(vision_period)
                    render_only()
                    frame = camera.get_current_frame(clone=True)
                    frame_id = int(frame.get("rendering_frame", 0))
                    age = float(world.current_time) - float(frame.get("rendering_time", -1.0))
                    diagnostics.append({"id": frame_id, "age": age, "shape": list(np.shape(frame.get("rgba", [])))})
                    if (frame_id <= 0 or frame_id in frames or not 0 <= age <= .20
                            or float(frame["rendering_time"]) < started):
                        continue
                    frames.add(frame_id)
                    rgba = np.asarray(frame.get("rgba", []))
                    if rgba.size == 0:
                        continue
                    observation = estimator.estimate(rgba.astype(np.uint8, copy=False))
                    diagnostics[-1]["status"] = observation["status"]
                    pose = estimator.resting_pose(observation, support_z)
                    if pose is not None:
                        estimates.append(pose)
                        exposure_ids.append((frame_id, float(frame["rendering_time"])))
                    if len(estimates) >= 3:
                        break
            finally:
                if camera is supply_camera:
                    camera.pause()
            if len(estimates) < 3:
                import cv2
                rgba = np.asarray(camera.get_current_frame(clone=True).get("rgba", []))
                if rgba.ndim == 3:
                    cv2.imwrite(str(output_directory / f"{phase}_unobserved.png"),
                                cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR))
                report["failed_observation"] = diagnostics
                raise RuntimeError(f"{phase}: no stable fresh ArUco observation")
            positions, rotations = zip(*estimates)
            if np.max(np.linalg.norm(positions - np.median(positions, axis=0), axis=1)) > .006:
                raise RuntimeError(f"{phase}: Cube observations disagree by more than 6 mm")
            supported_pose_filter.reset()
            filtered_pose = None
            for (frame_id, stamp), pose in zip(exposure_ids, estimates):
                filtered_pose = supported_pose_filter.update(
                    frame_id, stamp, *pose, reference_rotation=contact_plan_rotation)
            if filtered_pose is None:
                raise RuntimeError(f"{phase}: three chronological optical frames are required")
            control_center, control_rotation, supported_pose_time = filtered_pose
            report["target_observations"].append({
                "phase": phase, "source": "supply_rgb" if camera is supply_camera else "rear_rgb",
                "frame_count": len(estimates), "position_m": control_center.tolist(),
                "yaw_deg": math.degrees(math.atan2(control_rotation[1, 0], control_rotation[0, 0])),
            })
            return control_center.copy(), control_rotation.copy()

        def step(count=1):
            nonlocal tick, last_control_wall, vision_task, last_force_sample
            for _ in range(count):
                if not app.is_running():
                    raise RuntimeError("Viewer was closed before the task finished")
                if not args.headless:
                    time.sleep(max(0.0, 1 / 120 - (time.monotonic() - last_control_wall)))
                last_control_wall = time.monotonic()
                if left_hold is not None:
                    normal = -last_force_sample[support_side]["tcp_gravity_corrected_world"][1]
                    y = left_hold.update(normal)
                    nominal = nominal_targets[support_side]
                    position, rotation = tcp_fk(support_side, nominal)
                    position[1] = y
                    position[2] = left_hold_z
                    command_positions(support_side, cartesian_joint_target(
                        support_side, nominal, position, left_hold_rotation), nominal=False,
                        velocity=applied_velocities[support_side])
                for side, robot in robots.items():
                    gravity = robot._articulation_view.get_generalized_gravity_forces(joint_indices=joints[side])[0]
                    if not np.all(np.isfinite(gravity)):
                        raise RuntimeError(f"{side} gravity compensation is not finite")
                    robot.get_articulation_controller().apply_action(
                        ArticulationAction(joint_efforts=gravity, joint_indices=joints[side]))
                # One command advances one physics tick in both execution modes.
                physics_before = world.current_time_step_index
                contact_monitor.physics_step_index = physics_before + 1
                world.step(render=False)
                grippers.update()
                if seals_required and not all(grippers.is_closed(side) for side in robots):
                    report["lost_suction"] = [side for side in robots if not grippers.is_closed(side)]
                    raise RuntimeError("Surface Gripper seal lost while carrying the Cube")
                tick += 1
                if world.current_time_step_index - initial_physics_step != tick:
                    raise RuntimeError(
                        f"Physics and command step counts diverged at {tick}: "
                        f"physics {physics_before} -> {world.current_time_step_index}, "
                        f"expected {initial_physics_step + tick}, timeline playing={world.is_playing()}"
                    )
                report["simulation_clock"]["control_steps"] = tick
                report["simulation_clock"]["physics_steps"] = world.current_time_step_index - initial_physics_step
                force_recorder.command_context = {
                    "position_targets_rad": {side: q.tolist() for side, q in applied_targets.items()},
                    "velocity_targets_rad_s": {side: q.tolist() for side, q in applied_velocities.items()},
                    "left_hold_position_y_m": None if left_hold is None else left_hold.position_m,
                }
                force_sample = force_recorder.sample(float(world.current_time), tick, contact_monitor.phase)
                last_force_sample = force_sample
                if contact_monitor.phase.startswith("A"):
                    forces = [np.asarray(force_sample[side]["tcp_gravity_corrected_world"][:3])
                              for side in robots]
                    if (np.linalg.norm((forces[0] - forces[1]) / 2) > A_FORCE_LIMITS["max_internal_force_N"]
                            or any(np.linalg.norm(force) > A_FORCE_LIMITS["max_force_N"] for force in forces)
                            or any(np.linalg.norm(force_sample[side]["tcp_gravity_corrected_world"][3:])
                                   > A_FORCE_LIMITS["max_tcp_torque_Nm"] for side in robots)):
                        raise RuntimeError("A measured force or TCP torque exceeds the declared acceptance envelope")
                if force_label is not None and tick % 12 == 0:
                    lines = ["World axes, tool gravity removed; F in N, T in N m"]
                    for side in ("left", "right"):
                        wrench = force_sample[side]["gravity_corrected_world"]
                        tcp_wrench = force_sample[side]["tcp_gravity_corrected_world"]
                        force = ", ".join(f"{value:+.2f}" for value in wrench[:3])
                        torque = ", ".join(f"{value:+.3f}" for value in wrench[3:])
                        tcp_torque = ", ".join(f"{value:+.3f}" for value in tcp_wrench[3:])
                        lines.append(f"{side}: F [{force}] (same at sensor/TCP)\n"
                                     f"  T_sensor [{torque}]  T_TCP [{tcp_torque}]")
                    force_label.text = "\n".join(lines)
                contact_monitor.raise_if_collision()
                reference_center, reference_orientation = cube.get_world_pose()
                truth_history.append({"time_s": float(world.current_time), "physics_step": tick,
                                      "phase": contact_monitor.phase,
                                      "position_m": reference_center.tolist()})
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
                measured = joint_state()
                sensor_poses = {side: tcp_fk(side, measured[side]) for side in robots}
                reference_contacts = contact_monitor.cube_contact_snapshot()
                if supply_transition:
                    reference_contacts["valid"] = False
                    report.setdefault("reference_invalid_transition_steps", []).append(tick)
                recognized = contact_experiment.append(
                    physics_step=tick,
                    sensors={
                        "time_s": float(world.current_time),
                        "left_wrench_world": force_sample["left"]["tcp_gravity_corrected_world"],
                        "right_wrench_world": force_sample["right"]["tcp_gravity_corrected_world"],
                        "tcp_positions_m": {side: pose[0].tolist() for side, pose in sensor_poses.items()},
                        "cup_normals_world": {side: (pose[1] @ np.array([
                            0., -1. if side == "left" else 1., 0.])).tolist()
                            for side, pose in sensor_poses.items()},
                        "camera": latest_camera_observation,
                    },
                    truth={
                        "cube_position_m": reference_center.tolist(),
                        "cube_rotation_world": quat_to_rot_matrix(reference_orientation).tolist(),
                        "attachments": {side: grippers.is_closed(side) for side in robots},
                        "cube_contacts": reference_contacts,
                    },
                    auxiliary={
                        "cube_id": None if current_task is None else current_task.get("cube_id"),
                        "neighbors": {
                            path: {
                                "position_m": np.asarray(value[0], dtype=float).tolist(),
                                "rotation_world": np.asarray(value[1], dtype=float).tolist(),
                                "capture_time_s": float(value[2]),
                                "received_time_s": float(world.current_time),
                            }
                            for path, value in live_neighbors.items()
                        },
                    })
                if status_label is not None and tick % 12 == 0:
                    status_label.text = (f"{contact_monitor.phase}: executing\n"
                                         f"Contact estimate: {recognized['state']}\n"
                                         f"Skills: {', '.join(recognized['recommended_skills'])}")

        def record(name, state, **extra):
            position, orientation = cube.get_world_pose()
            item = {"phase": name, "status": state, "sim_time_s": tick / 120,
                    "cube_position_m": position.tolist(), "cube_orientation_wxyz": orientation.tolist(),
                    "cube_id": current_task["cube_id"], **extra}
            contact_monitor.phase = name
            report["active_phase"] = name
            report["phases"].append(item)
            if status_label is not None:
                status_label.text = f"{name}: {state}"
            print("FR3_AB " + json.dumps(item), flush=True)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

        def joint_state():
            return {side: np.asarray(robot.get_joint_positions(joint_indices=joints[side]), dtype=float)
                    for side, robot in robots.items()}

        def command_positions(side, q, nominal=True, velocity=None):
            q = np.asarray(q, dtype=float)
            if nominal:
                nominal_targets[side] = q.copy()
            applied_targets[side] = q.copy()
            applied_velocities[side] = np.zeros_like(q) if velocity is None else np.asarray(velocity).copy()
            robots[side].get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=q, joint_velocities=applied_velocities[side],
                                   joint_indices=joints[side]))

        def set_contact_drive_mode(enabled):
            scale = contact_drive_scale if enabled else 1.
            event = {"physics_step": tick, "time_s": float(world.current_time),
                     "mode": "contact" if enabled else "free_space", "gain_scale": scale,
                     "zero_velocity_command": True, "arms": {}}
            for side, robot in robots.items():
                q = joint_state()[side]
                command_positions(side, q)
                controller = robot.get_articulation_controller()
                gains = [np.asarray(values).copy() for values in controller.get_gains()]
                for values, original in zip(gains, original_arm_gains[side]):
                    values[joints[side]] = scale * original
                controller.set_gains(kps=gains[0], kds=gains[1])
                actual = {name: np.asarray(values)[joints[side]].tolist()
                          for name, values in zip(("stiffness", "damping"), controller.get_gains())}
                efforts = np.asarray(controller.get_max_efforts())[joints[side]]
                if not np.array_equal(efforts, original_arm_efforts[side]):
                    raise RuntimeError(f"{side} contact drive transition changed joint effort limits")
                event["arms"][side] = {**actual, "effort_limits": efforts.tolist(),
                                       "rebased_joint_positions_rad": q.tolist()}
            report.setdefault("joint_drive_mode_events", []).append(event)
            force_recorder.metadata["joint_drive_mode_events"] = report["joint_drive_mode_events"]
            print("FR3_DRIVE_MODE " + json.dumps(event), flush=True)

        def tcp_fk(side, q):
            position, rotation = solvers[side].compute_forward_kinematics("fr3_link8", q)
            rotation = np.asarray(rotation, dtype=float)
            return (np.asarray(position, dtype=float).reshape(3) + rotation @ force_recorder.tcp_offsets[side], rotation)

        def sensed_cube():
            nonlocal control_center, control_rotation
            if carry_offsets is not None:
                measured = joint_state()
                carried = {}
                for side, (offset, relative_rotation) in carry_offsets.items():
                    tcp, rotation = tcp_fk(side, measured[side])
                    carried[side] = (tcp + rotation @ offset, rotation @ relative_rotation)
                left, right = carried["left"], carried["right"]
                if np.linalg.norm(left[0] - right[0]) > .012:
                    raise RuntimeError("The two encoder-based payload estimates disagree by over 12 mm")
                control_center = (left[0] + right[0]) / 2
                control_rotation = left[1]
            elif push_faces:
                measured = joint_state()
                for side, (axis, outward, local_offset) in push_faces.items():
                    force = np.asarray(last_force_sample[side]["tcp_gravity_corrected_world"][:3])
                    if force @ outward > .1:
                        tcp, _ = tcp_fk(side, measured[side])
                        measured_axis = (tcp - control_rotation @ local_offset)[axis]
                        control_center[axis] = measured_axis
            report["control_pose"] = {
                "position_m": control_center.tolist(), "rotation_world": control_rotation.tolist(),
                "source": "encoder_grasp_geometry" if carry_offsets is not None else
                          "contact_encoder_and_last_visual_yaw" if push_faces else "RGB_support_plane",
                "visual_age_s": None if supported_pose_time is None else
                                float(world.current_time) - supported_pose_time,
                "yaw_during_occlusion": "last valid camera estimate",
            }
            return control_center.copy(), control_rotation.copy()

        def touch_face(side, inward, distance, phase):
            record(phase, "TACTILE_APPROACH")
            q = applied_targets[side].copy()
            tcp, rotation = tcp_fk(side, q)
            stable = 0
            seek_step = B2_CONTROLLER_PARAMETERS["touch_seek_step_m"]
            force_threshold = B2_CONTROLLER_PARAMETERS["touch_contact_force_N"]
            confirmation_frames = B2_CONTROLLER_PARAMETERS["touch_confirmation_frames"]
            delta = 0.
            for _ in range(math.ceil(distance / seek_step) + confirmation_frames + 1):
                q = cartesian_joint_target(side, q, tcp + delta * inward, rotation)
                command_positions(side, q)
                step()
                # Wrist seeking allows 5 mm RGB pose uncertainty for the intended cups.
                # The separate PhysX collision guard retains its actual 3 mm limit.
                check_current_collision(phase, contact_uncertainty_m=.005)
                force = np.asarray(last_force_sample[side]["tcp_gravity_corrected_world"][:3])
                compression = float(-force @ inward)
                if compression > 10.0:
                    raise RuntimeError(f"{phase}: unexpected contact force {compression:.2f} N")
                stable = stable + 1 if compression >= force_threshold else 0
                if stable >= confirmation_frames:
                    report["target_observations"].append({
                        "phase": phase, "source": "encoder_fk_and_wrist_force",
                        "force_N": compression, "search_m": float(delta),
                    })
                    return
                if compression < force_threshold:
                    if delta >= distance - 1e-12:
                        break
                    delta = min(distance, delta + seek_step)
            raise RuntimeError(f"{phase}: wrist did not confirm contact within {distance * 1000:g} mm")

        def cartesian_joint_target(side, seed, target, target_rotation, retarget_guide=False):
            from scipy.optimize import least_squares
            from scipy.spatial.transform import Rotation

            q = np.asarray(seed, dtype=float).copy()
            yaw = CONFIG["fr3"]["bases"][side][3]
            base_rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                      [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.]])
            position_tolerance = 1e-7 if retarget_guide else 2e-9
            angle_tolerance = 1e-6 if retarget_guide else 2e-8
            norm_order = 2 if retarget_guide else np.inf
            if retarget_guide:
                props = robots[side].dof_properties
                lower = props["lower"][joints[side]]
                upper = props["upper"][joints[side]]
                gradient = (np.clip(.15 - (q - lower), 0, .15)
                            - np.clip(.15 - (upper - q), 0, .15)) / .15
                if np.max(np.abs(gradient)) > 0:
                    _, rotation = tcp_fk(side, q)
                    jac = np.asarray(solvers[side]._kinematics.jacobian(q, "fr3_link8"), dtype=float)
                    jac[:3] = base_rotation @ jac[:3]
                    jac[3:] = base_rotation @ jac[3:]
                    lever = rotation @ force_recorder.tcp_offsets[side]
                    jac[:3] += np.cross(jac[3:].T, lever).T
                    # Move redundant joints away from limits before exact TCP projection.
                    null = gradient - jac.T @ np.linalg.solve(
                        jac @ jac.T + 1e-8 * np.eye(6), jac @ gradient)
                    correction = .04 * null
                    correction *= min(1., .0015 / max(np.max(np.abs(correction)), 1e-12))
                    q = np.clip(q + correction, lower + 1e-9, upper - 1e-9)
            for _ in range(30 if retarget_guide else 6):
                position, rotation = tcp_fk(side, q)
                error = np.r_[target - position, Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()]
                if (np.linalg.norm(error[:3], ord=norm_order) < position_tolerance
                        and np.linalg.norm(error[3:], ord=norm_order) < angle_tolerance):
                    return q
                jac = np.asarray(solvers[side]._kinematics.jacobian(q, "fr3_link8"), dtype=float)
                jac[:3] = base_rotation @ jac[:3]
                jac[3:] = base_rotation @ jac[3:]
                lever = rotation @ force_recorder.tcp_offsets[side]
                jac[:3] += np.cross(jac[3:].T, lever).T
                delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-8 * np.eye(6), error)
                if np.max(np.abs(delta)) > 0.05:
                    if not retarget_guide:
                        raise RuntimeError(f"{side} local Cartesian correction would jump IK branch")
                    delta *= .05 / np.max(np.abs(delta))
                props = robots[side].dof_properties
                candidate = q + delta
                if (np.any(candidate < props["lower"][joints[side]])
                        or np.any(candidate > props["upper"][joints[side]])):
                    if not retarget_guide:
                        raise RuntimeError(f"{side} Cartesian correction exceeds joint limits")
                    break
                q = candidate
            if not retarget_guide:
                raise RuntimeError(f"{side} local Cartesian correction did not converge")
            props = robots[side].dof_properties
            lower = props["lower"][joints[side]]
            upper = props["upper"][joints[side]]
            def residual(candidate):
                position, rotation = tcp_fk(side, candidate)
                return np.r_[target - position,
                             .2 * Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()]
            fit = least_squares(residual, np.clip(q, lower + 1e-9, upper - 1e-9),
                                bounds=(lower, upper), max_nfev=200,
                                xtol=1e-13, ftol=1e-13, gtol=1e-13)
            error = residual(fit.x)
            if (np.linalg.norm(error[:3]) < position_tolerance
                    and np.linalg.norm(error[3:]) / .2 < angle_tolerance):
                return fit.x
            raise RuntimeError(f"{side} bounded Cartesian correction failed: "
                               f"position error {np.linalg.norm(error[:3]):.3e} m, "
                               f"orientation error {np.linalg.norm(error[3:]) / .2:.3e} rad")

        def check_current_collision(phase, contact_uncertainty_m=0.):
            center, rotation = sensed_cube()
            joints_now = joint_state()
            options = dict(cube_center=center, cube_rotation=rotation, cup_contacts=allowed_contacts,
                           margin_m=0., contact_tolerance_m=.003 + contact_uncertainty_m, first_only=True)
            issues = checker.check(joints_now, **options)
            target = current_task.get("contact_target")
            if (issues and phase in ("B2_push_seat", "B2_push_unload")
                    and target in live_neighbors and issues[0]["kind"] == "cube_obstacle"
                    and issues[0]["b"] == target):
                neighbor_center, neighbor_rotation, capture = live_neighbors[target]
                age = float(world.current_time) - capture
                force = np.asarray(last_force_sample[push_side]["tcp_gravity_corrected_world"][:3])
                normal = float(-push_sign * force[0])
                unloaded = phase == "B2_push_unload"
                if 0 <= age <= .20 and np.linalg.norm(force) <= 10. and (normal >= .5 or unloaded):
                    previous = issues[0]
                    checker.set_other_cubes([(target, neighbor_center, neighbor_rotation)], preserve_unobserved=True)
                    issues = checker.check(joints_now, **options)
                    checks = report.setdefault("neighbor_geometry_rechecks", {})
                    key = f"{current_task['cube_id']}_{phase}"
                    if key not in checks:
                        stored = next(row for packet in reversed(report["neighbor_observations"])
                                      for row in packet["neighbors"] if row["name"] == target)
                        checks[key] = {"time_s": float(world.current_time), "target": target,
                                       "old_clearance_m": previous["clearance_m"],
                                       "previous_rgb_center_m": stored["position_m"],
                                       "previous_rgb_age_s": float(world.current_time) - stored["capture_time_s"],
                                       "fresh_rgb_center_m": neighbor_center.tolist(), "fresh_rgb_age_s": age,
                                       "fused_active_center_m": center.tolist(), "axial_force_N": normal,
                                       "target_pair_tolerance_m": options["contact_tolerance_m"],
                                       "fresh_rgb_rotation": neighbor_rotation.tolist(), "fused_active_rotation": rotation.tolist()}
            if issues:
                raise RuntimeError(f"Measured collision in {phase}: {issues[0]}")

        def cartesian_move(side, destination, duration, phase):
            record(phase, "RUNNING")
            start, rotation = tcp_fk(side, applied_targets[side])
            duration = max(duration, 1.875 * float(np.linalg.norm(np.asarray(destination) - start)) / .08)
            seed = applied_targets[side].copy()
            for t in np.linspace(0, 1, max(2, int(duration * 120))):
                blend = 10 * t**3 - 15 * t**4 + 6 * t**5
                target = start + blend * (np.asarray(destination) - start)
                seed = cartesian_joint_target(side, seed, target, rotation)
                command_positions(side, seed)
                step()
                check_current_collision(phase)

        def align_contact_plane(side, target_rotation, phase):
            from scipy.spatial.transform import Rotation
            record(phase, "ALIGN_TO_CALIBRATED_WALL")
            q = applied_targets[side].copy()
            position, start_rotation = tcp_fk(side, q)
            axis_angle = Rotation.from_matrix(start_rotation.T @ target_rotation).as_rotvec()
            for t in np.linspace(0., 1., 241):
                blend = (10 - 15*t + 6*t*t) * t**3
                rotation = start_rotation @ Rotation.from_rotvec(blend * axis_angle).as_matrix()
                q = cartesian_joint_target(side, q, position, rotation)
                command_positions(side, q)
                step()
                check_current_collision(phase)
                if abs(last_force_sample[side]["tcp_gravity_corrected_world"][1]) > 10.:
                    raise RuntimeError(f"{phase}: contact overloaded during plane alignment")

        def paired_cartesian_move(destinations, duration, phase):
            times = np.arange(math.ceil(duration * 120) + 1) / 120
            t = times / times[-1]
            blend = (10 - 15*t + 6*t**2) * t**3
            current = joint_state()
            samples = {}
            for side, q in current.items():
                position, rotation = tcp_fk(side, q)
                path = []
                for fraction in blend:
                    planning_update(f"{phase}: paired Cartesian trajectory")
                    target = position + fraction * (np.asarray(destinations[side]) - position)
                    q = cartesian_joint_target(side, q, target, rotation)
                    path.append(q.copy())
                samples[side] = np.asarray(path)
            peak_speed = max(float(np.max(np.abs(np.diff(path, axis=0)))) * 120
                             for path in samples.values())
            if peak_speed > .4:
                raise RuntimeError(f"{phase}: paired Cartesian speed {peak_speed:.3f} rad/s exceeds 0.4")
            execute_joint_path(samples, phase, sampled_trajectory=(times, samples))

        def seat_against_wall(side, axis, sign, center_target, phase, *, contact_target, follow_side=None, probe_override=None):
            neighbor = contact_target in ("Cube_1", "Cube_2")
            if follow_side is not None and not (neighbor or contact_target in ("wall_left", "wall_right")):
                raise ValueError(f"Unknown lateral contact target: {contact_target}")
            probe = probe_override or B2_CONTROLLER_PARAMETERS["neighbor_seating_probe" if neighbor else "wall_seating_probe"]
            record(phase, "RUNNING", contact_target=contact_target)
            position, tool_rotation = tcp_fk(side, applied_targets[side])
            seed = applied_targets[side].copy()
            probe_forces = []
            probe_positions = []
            probe_paused = False
            entry_count = 0
            for _ in range(7200 if follow_side is not None or probe_override is not None else 3600):
                actual, cube_rotation = sensed_cube()
                gap = oriented_wall_gap(actual[axis], cube_rotation[axis],
                                         half, center_target, sign)
                measured_force = np.asarray(last_force_sample[side]["tcp_gravity_corrected_world"][:3])
                force = -sign * measured_force[axis]
                force_norm = float(np.linalg.norm(measured_force))
                if abs(force) > 10 or ((follow_side is not None or probe_override is not None) and force_norm > 10):
                    raise RuntimeError(f"{side} wall acquisition overload: "
                                       f"normal {force:.3f} N, resultant {force_norm:.3f} N")
                previous_axis = position[axis]
                if follow_side is None and probe_override is None:
                    if gap <= B2_CONTROLLER_PARAMETERS["seating_gap_tolerance_m"] and force > .5:
                        return
                    speed = .0001 if gap > .0001 else .00002
                    position[axis] += sign * speed / 120
                else:
                    near_target = gap <= max(.001, probe["target_uncertainty_m"])
                    if probe_paused:
                        if near_target and (not neighbor or force >= probe["minimum_force_N"]):
                            probe_forces.append(float(force))
                            probe_positions.append(float(actual[axis]))
                        else:
                            probe_positions.clear()
                            probe_forces.clear()
                            probe_paused = False
                    else:
                        loaded = near_target and force >= probe["entry_force_N"]
                        entry_count = entry_count + 1 if loaded else 0
                        if entry_count >= probe.get("entry_samples", 1):
                            probe_paused = True
                            entry_count = 0
                            if not neighbor:
                                probe_forces.append(float(force))
                    if len(probe_forces) >= probe["samples"]:
                        mean_force = float(np.mean(probe_forces))
                        travel = max(probe_positions) - min(probe_positions) if probe_positions else 0.
                        if (mean_force >= probe.get("mean_force_N", -np.inf)
                                and travel <= probe.get("max_axial_travel_m", np.inf)):
                            force_criterion = ("minimum_force_N" if neighbor else "mean_force_N")
                            record(phase, "FORCE_CONFIRMED", normal_force_N=force, mean_normal_force_N=mean_force,
                                   probe_normal_samples_N=probe_forces.copy(), probe_entry_force_N=probe["entry_force_N"],
                                   required_mean_force_N=probe.get("mean_force_N"),
                                   entry_samples=probe.get("entry_samples", 1),
                                   sliding_baseline_N=probe.get("sliding_baseline_N"),
                                   allowed_axial_travel_m=probe.get("max_axial_travel_m"),
                                   measured_axial_travel_m=travel,
                                   estimated_gap_m=float(gap), target_uncertainty_m=probe["target_uncertainty_m"],
                                   follow_arm=follow_side, contact_target=contact_target,
                                   **{f"probe_{force_criterion}": probe[force_criterion]})
                            return
                        probe_positions.clear()
                        probe_forces.clear()
                        probe_paused = False
                    if not probe_paused:
                        speed = .0005 if gap > .001 and force_norm < 4. else .0001
                        if neighbor and probe_override is not None and gap <= probe["target_uncertainty_m"]:
                            # Slow down before a corner-to-face load can develop at the staging lane.
                            speed = .00010 if force < 1.6 else .000020
                        position[axis] += sign * speed / 120
                    if follow_side is not None:
                        follow_position, follow_rotation = tcp_fk(follow_side, applied_targets[follow_side])
                        follow_position[axis] += sign * max(0., sign * (position[axis] - previous_axis))
                        follow_q = cartesian_joint_target(
                            follow_side, applied_targets[follow_side], follow_position, follow_rotation)
                        command_positions(follow_side, follow_q)
                seed = cartesian_joint_target(side, seed, position, tool_rotation)
                command_positions(side, seed)
                step()
                check_current_collision(phase)
            raise RuntimeError(f"{side} could not acquire wall contact")

        def settle_b2_contacts(active_side, direction, phase, support_target=None):
            record(phase, "STABILIZING_DUAL_CONTACT")
            seed = applied_targets[active_side].copy()
            position, rotation = tcp_fk(active_side, seed)
            controller = NormalForceController(
                direction * position[0], target_force_N=1., gain_m_per_Ns=1e-4,
                max_speed_m_s=B2_CONTROLLER_PARAMETERS["normal_position_speed_limit_m_s"],
                max_offset_m=.002, max_force_N=10.)
            support = "right" if active_side == "left" else "left"
            filtered = {
                support: -last_force_sample[support]["tcp_gravity_corrected_world"][1],
                active_side: -direction * last_force_sample[active_side]["tcp_gravity_corrected_world"][0],
            }
            if support_target is None:
                support_target = B2_CONTROLLER_PARAMETERS["left_target_force_N"]
            support_low, support_high = ((.3, 1.2) if support_target < 1. else (1., 3.))
            stable = 0
            for _ in range(2400):
                force = -direction * last_force_sample[active_side]["tcp_gravity_corrected_world"][0]
                position[0] = direction * controller.update(force)
                seed = cartesian_joint_target(active_side, seed, position, rotation)
                command_positions(active_side, seed)
                step()
                check_current_collision(phase)
                current = {
                    support: -last_force_sample[support]["tcp_gravity_corrected_world"][1],
                    active_side: -direction * last_force_sample[active_side]["tcp_gravity_corrected_world"][0],
                }
                if any(abs(force) > 10. for force in current.values()):
                    raise RuntimeError(f"{phase}: dual-contact force exceeds 10 N")
                for side, force in current.items():
                    filtered[side] += .05 * (force - filtered[side])
                if (left_hold.target_force_N > support_target
                        and filtered[active_side] >= B2_CONTROLLER_PARAMETERS["active_sync_loaded_force_N"]):
                    left_hold.target_force_N = support_target
                    record(phase, "REAR_PRELOAD_REDUCED_AFTER_ACTIVE_LOAD",
                           filtered_normal_force_N=filtered.copy())
                ready = (left_hold.target_force_N == support_target
                         and support_low <= filtered[support] <= support_high and .5 <= filtered[active_side] <= 1.5)
                stable = stable + 1 if ready else 0
                if stable >= 60:
                    record(phase, "FORCE_CONFIRMED", filtered_normal_force_N=filtered)
                    return
            raise RuntimeError(f"{phase}: dual contact did not stabilize within 20 seconds")

        def unload_normal(side, axis, sign, phase):
            record(phase, "RUNNING")
            position, rotation = tcp_fk(side, applied_targets[side])
            seed = applied_targets[side].copy()
            controller = (left_hold if side == support_side and axis == 1 and left_hold is not None else
                          NormalForceController(sign * position[axis], target_force_N=0,
                                                gain_m_per_Ns=B2_CONTROLLER_PARAMETERS["unload_gain_m_per_Ns"]))
            controller.gain_m_per_Ns = B2_CONTROLLER_PARAMETERS["unload_gain_m_per_Ns"]
            initial_force = max(0.0, -sign * last_force_sample[side]["tcp_gravity_corrected_world"][axis])
            stable = 0
            for index in range(1800):
                force = -sign * last_force_sample[side]["tcp_gravity_corrected_world"][axis]
                target_force = initial_force * max(0.0, 1 - index / 120)
                if side == support_side and axis == 1 and left_hold is not None:
                    controller.target_force_N = target_force
                else:
                    normal_position = min(
                        sign * position[axis], controller.update(force, target_force_N=target_force))
                    controller.position_m = normal_position
                    position[axis] = sign * normal_position
                    seed = cartesian_joint_target(side, seed, position, rotation)
                    command_positions(side, seed)
                step()
                check_current_collision(phase)
                force = -sign * last_force_sample[side]["tcp_gravity_corrected_world"][axis]
                stable = stable + 1 if index >= 120 and abs(force) < 0.1 else 0
                if stable >= 30:
                    return
            raise RuntimeError(f"{side} could not unload contact force")

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
            center, rotation = sensed_cube()
            def clear(q):
                planning_update(planning_message)
                states = {**current, **{side: q[i * 7:(i + 1) * 7] for i, side in enumerate(sides)}}
                return checker.is_clear(states, cube_center=center, cube_rotation=rotation,
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

        def dual_target(center, gap=0, continuous=False, seed_overrides=None, rotation=None):
            rotation = np.eye(3) if rotation is None else rotation
            return {side: solve(side, np.array(center) + rotation @ [sign * (half + gap), 0, 0],
                                rotation @ SIDE_ROTATION,
                                continuous, None if seed_overrides is None else seed_overrides.get(side))
                    for side, sign in (("left", -1), ("right", 1))}

        def execute_joint_path(paths, phase, max_joint_speed=0.4, check_step=None,
                                moving_cube_offset=None, cube_center_path=None,
                                prefetched_cache=None, sampled_trajectory=None):
            planning_update(f"{phase}: trajectory planning", force=True)
            current = joint_state()
            count = {len(path) for path in paths.values()}
            if len(count) != 1 or not count or next(iter(count)) == 0:
                raise ValueError("Synchronized paths need the same positive number of waypoints")
            sides = list(current)
            cached_entry = (None if sampled_trajectory is not None else
                            prefetched_cache or take_cached_phase(phase, current, requested_paths=paths))
            if sampled_trajectory is not None:
                times, samples = sampled_trajectory
                all_paths = samples
                replay_source = "Cartesian_quintic_with_force_overlay"
            elif cached_entry is not None:
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
            center, cube_rotation = sensed_cube()
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
            if cached_entry is not None:
                validation = {"status": "CLEAR", "samples_checked": len(times),
                              "issues": [], "source": "cached_trajectory"}
            else:
                validation = validate_planned_path(phase, samples, cube_centers, cube_rotation)
            if (validation["status"] != "CLEAR" and moving_cube_offset is None
                    and sampled_trajectory is None):
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
                validation = validate_planned_path(phase, samples, cube_centers, cube_rotation)
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
            velocities = ({side: np.gradient(samples[side], 1 / 120, axis=0)
                           for side in sides} if phase in ("A_transfer", "B2_push") else {})
            for index in range(len(times)):
                for side in sides:
                    velocity = (velocities[side][index] if side in velocities
                                and 0 < index < len(times) - 1 else None)
                    command_positions(side, samples[side][index], velocity=velocity)
                step()
                actual_center, actual_rotation = sensed_cube()
                issues = checker.check(joint_state(), cube_center=actual_center,
                                       cube_rotation=actual_rotation,
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

        def validate_planned_path(phase, samples, cube_centers, cube_rotation):
            start = previous_refresh = time.monotonic()
            start_tick = tick
            render_count = report["simulation_clock"]["render_count"]
            max_gap = 0.

            def progress(label):
                nonlocal previous_refresh, max_gap
                planning_update(f"{phase}: collision checking {label}")
                if last_render_wall > previous_refresh:
                    max_gap = max(max_gap, last_render_wall - previous_refresh)
                    previous_refresh = last_render_wall

            parallel = parallel_validator is not None and len(next(iter(samples.values()))) >= 240
            if parallel:
                bases = {side: (base[:3].copy(), quats_to_rot_matrices(
                    np.array([math.cos(base[3] / 2), 0, 0, math.sin(base[3] / 2)])))
                         for side, base in rail_states.items()}
                result = parallel_validator.validate(
                    samples, bases=bases, neighbors=checker.other_cubes,
                    centers=cube_centers, rotations=cube_rotation, contacts=tuple(allowed_contacts),
                    progress=lambda done, total: progress(f"{done}/{total} intervals"),
                )
            else:
                result = checker.validate_path(
                    samples, cube_centers=cube_centers, cube_rotations=cube_rotation,
                    cup_contacts=allowed_contacts, progress_callback=lambda checked: progress(f"{checked} samples"),
                )
            end = time.monotonic()
            timing = {"phase": phase, "status": result["status"], "workers": workers if parallel else 1,
                      "samples_checked": result["samples_checked"], "wall_s": end - start, "control_steps": tick - start_tick,
                      "render_updates": report["simulation_clock"]["render_count"] - render_count,
                      "max_render_gap_s": None if args.headless else max(max_gap, end - previous_refresh)}
            report["path_validation_timing"].append(timing)
            print("FR3_PATH_VALIDATION " + json.dumps(timing), flush=True)
            return result

        def dual_path(start, end):
            start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
            def check_step(index, count):
                if index != count - 1:
                    return
                actual, _ = sensed_cube()
                deviation = float(np.linalg.norm(np.asarray(actual) - end))
                if deviation > 0.08:
                    raise RuntimeError(f"Cube departed the A path by {deviation:.3f} m")
            cached = take_cached_phase(
                "A_transfer", joint_state(),
                validator=lambda entry, samples: cached_dual_target_matches(entry, samples, end, half))
            if cached is not None:
                report["trajectory_cache"]["used_phases"].pop()
            current = {side: applied_targets[side].copy() for side in robots}
            poses = {side: tcp_fk(side, current[side]) for side in robots}
            actual_center, _ = sensed_cube()
            delta = end - np.asarray(actual_center)
            # Retain the validated redundant posture while retargeting every TCP sample.
            guide_cache = json.loads(A_POSTURE_GUIDES.read_text())
            guides = [entry for entry in guide_cache["phases"] if entry["phase"] == "A_transfer"]
            guide_index = ("A_lift", "A_over_front", "A_landing").index(report["active_phase"])
            guide = guides[guide_index]
            guide_times = np.asarray(guide["times_s"], dtype=float)
            guide_paths = {side: np.asarray(guide["joint_samples_rad"][side], dtype=float)
                           for side in robots}
            if (len(guide_times) < 2 or np.any(np.diff(guide_times) <= 0)
                    or not np.all(np.isfinite(guide_times))
                    or any(path.shape != (len(guide_times), 7) or not np.all(np.isfinite(path))
                           for path in guide_paths.values())):
                raise RuntimeError("Invalid A redundant-posture guide")
            duration = max(float(guide_times[-1]), 1.875 * float(np.linalg.norm(delta)) / 0.08)
            max_joint_speed = .4
            for attempt in range(3):
                times = np.arange(int(np.ceil(duration * 120)) + 1) / 120
                seeds = {side: q.copy() for side, q in current.items()}
                previous_guides = {side: guide_paths[side][0].copy() for side in robots}
                samples = {side: [] for side in robots}
                centers = []
                max_rigidity_error = 0.0
                for t in times / times[-1]:
                    planning_update("A: shared rigid Cartesian path")
                    blend = 10 * t**3 - 15 * t**4 + 6 * t**5
                    centers.append(np.asarray(actual_center) + blend * delta)
                    for side in robots:
                        position, rotation = poses[side]
                        target = position + blend * delta
                        guided = np.array([np.interp(t * guide_times[-1], guide_times, values)
                                           for values in guide_paths[side].T])
                        seed = seeds[side] + guided - previous_guides[side]
                        previous_guides[side] = guided
                        target_q = cartesian_joint_target(side, seed, target, rotation, retarget_guide=True)
                        if np.max(np.abs(target_q - seeds[side])) > 0.02:
                            raise RuntimeError(f"{side} retargeted A guide has a discontinuous joint step")
                        seeds[side] = target_q
                        reached, reached_rotation = tcp_fk(side, seeds[side])
                        error = float(np.linalg.norm(reached - target))
                        max_rigidity_error = max(max_rigidity_error, error)
                        if error > 1e-6 or np.max(np.abs(reached_rotation - rotation)) > 1e-6:
                            raise RuntimeError("A Cartesian sample violates rigid attachment geometry")
                        samples[side].append(seeds[side].copy())
                samples = {side: np.asarray(path) for side, path in samples.items()}
                peak_speed = max(float(np.max(np.abs(np.diff(path, axis=0)))) * 120
                                 for path in samples.values())
                if peak_speed <= max_joint_speed:
                    break
                duration *= 1.05 * peak_speed / max_joint_speed
            else:
                raise RuntimeError("A retiming could not satisfy the 0.4 rad/s joint speed limit")
            execute_joint_path(samples, "A_transfer", check_step=check_step,
                               cube_center_path=lambda _samples, _times: np.asarray(centers),
                               sampled_trajectory=(times, samples))
            trajectory_cache["phases"][-1]["a_transfer_revision"] = A_TRANSFER_REVISION
            trajectory_cache["phases"][-1]["max_rigid_pose_position_error_m"] = max_rigidity_error
            timing = {"joint_speed_limit_rad_s": max_joint_speed,
                      "max_joint_speed_rad_s": peak_speed, "retiming_attempts": attempt + 1}
            trajectory_cache["phases"][-1].update(timing)
            report["trajectory_plans"][-1].update(timing)
            trajectory_cache["phases"][-1]["posture_guide"] = {
                "path": str(A_POSTURE_GUIDES.relative_to(ROOT)), "index": guide_index,
                "retargeted_from": "continuous commanded TCP poses with sensed Cube common translation",
            }

        def contact_path(side, start, end, phase):
            nonlocal latched_contact_axis
            start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
            duration = max(4.0, 1.875 * float(np.linalg.norm(end - start)) / .08)
            times = np.arange(math.ceil(duration * 120) + 1) / 120
            t = times / times[-1]
            blend = (10 - 15*t + 6*t*t) * t**3
            centers = start + blend[:, None] * (end - start)
            current = joint_state()
            samples = {arm: np.repeat(q[None, :], len(times), axis=0)
                       for arm, q in current.items()}
            q = applied_targets[side].copy()
            tcp_start, rotation = tcp_fk(side, q)
            for index, center in enumerate(centers):
                planning_update(f"{phase}: measured contact trajectory")
                q = cartesian_joint_target(side, q, tcp_start + (center - start), rotation)
                samples[side][index] = q
            def check_step(index, count):
                actual, _ = sensed_cube()
                if np.linalg.norm(actual - centers[index]) > .035:
                    raise RuntimeError(f"{phase}: sensed Cube departed contact path by over 35 mm")
            # Keep the touched normal anchored while wrist load briefly drops between commands.
            latched_contact_axis = push_faces[side][0]
            try:
                execute_joint_path(samples, phase, check_step=check_step,
                                   cube_center_path=lambda _samples, _times: centers,
                                   sampled_trajectory=(times, samples))
            finally:
                latched_contact_axis = None

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
            rail_states[side] = base.copy()

        def move_robot_base(side, destination):
            nonlocal cache_enabled, rail_cache_cursor
            original = rail_states[side].copy()
            destination = np.asarray(destination, dtype=float)
            low, high = CONFIG["rails"]["travel_limits_m"]
            if not low <= destination[1] <= high:
                raise RuntimeError("Rail target exceeds travel limits")
            states = joint_state()
            center, rotation = sensed_cube()
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
                        issues = checker.check(states, cube_center=center, cube_rotation=rotation,
                                               cup_contacts=allowed_contacts, first_only=True)
                        if issues:
                            raise RuntimeError(f"Rail path is in collision: {issues[0]}")
                finally:
                    update_solver(original)
            for base in bases:
                set_robot_base_pose(side, base)
                for arm, q in states.items():
                    command_positions(arm, q)
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
            step(60)
            before, _ = observe_resting_cube(phase, vision_camera, vision_estimator, CONFIG["floor_top_m"])
            start = float(world.current_time)
            step(30)
            after, _ = observe_resting_cube(phase, vision_camera, vision_estimator, CONFIG["floor_top_m"])
            drift_speed = float(np.linalg.norm(after - before) / (world.current_time - start))
            if drift_speed > CONFIG["physics"]["settle_speed_threshold_m_s"]:
                raise RuntimeError(f"{phase}: vision observed Cube drift of {drift_speed:.3f} m/s")
            return drift_speed

        def try_attach_pair():
            nonlocal seals_required
            record("A_seal_contact", "RUNNING")
            targets = {side: tcp_fk(side, applied_targets[side]) for side in robots}
            normals = {side: rotation[:, 1] * (-1 if side == "left" else 1)
                       for side, (_, rotation) in targets.items()}
            if np.dot(normals["left"], normals["right"]) > -0.95:
                raise RuntimeError("A suction approach needs opposing tool normals")
            seek_speed_m_s = 0.0005
            max_travel_m = 0.015
            stable = 0
            travel = dict.fromkeys(robots, 0.0)
            controllers = dict.fromkeys(robots)
            for _ in range(int((max_travel_m / seek_speed_m_s + 10) * 120)):
                forces = {side: -float(np.dot(last_force_sample[side]["tcp_gravity_corrected_world"][:3],
                                              normals[side]))
                          for side in robots}
                stable = stable + 1 if all(.5 <= force <= 1.5 for force in forces.values()) else 0
                if stable >= 30:
                    for side in robots:
                        if not grippers.close(side):
                            for connected in robots:
                                grippers.open(connected)
                            raise RuntimeError(f"{side} Surface Gripper found no seal after wrist contact")
                    seals_required = True
                    record("A_attach", "SURFACE_GRIPPERS_CLOSED",
                           compression_N=forces, seek_travel_m=travel)
                    return
                for side, (position, rotation) in targets.items():
                    if controllers[side] is None and forces[side] > .1:
                        controllers[side] = NormalForceController(
                            travel[side], target_force_N=1., gain_m_per_Ns=2e-5,
                            max_offset_m=max_travel_m, max_force_N=A_FORCE_LIMITS["max_force_N"])
                    travel[side] = (controllers[side].update(forces[side]) if controllers[side] is not None
                                    else travel[side] + seek_speed_m_s / 120)
                    if abs(travel[side]) > max_travel_m:
                        raise RuntimeError(f"{side} suction cup did not reach stable contact within {max_travel_m * 1000:g} mm")
                    q = cartesian_joint_target(side, applied_targets[side],
                                               position + travel[side] * normals[side], rotation)
                    command_positions(side, q)
                step()
            raise RuntimeError("A tactile preload did not stabilize on both tools")

        def run_cube():
            nonlocal half
            nonlocal seals_required, carry_offsets, contact_plan_rotation
            nonlocal left_hold, left_hold_z, left_hold_rotation
            nonlocal support_side, push_side, push_sign

            step(120)
            record("A_approach", "RUNNING")
            supply_top = next(center[2] + size[2] / 2 for name, center, size in CONFIG["static_boxes"]
                              if name == "supply")
            supply, supply_rotation = observe_resting_cube("A_pickup_pose", supply_camera, supply_estimator, supply_top)
            move(dual_target(supply, gap=.035, rotation=supply_rotation), duration=3.)
            step(60)
            supply, supply_rotation = observe_resting_cube("A_contact_pose", supply_camera, supply_estimator, supply_top)
            approach_targets = {side: supply + supply_rotation @ [sign * (half + .007), 0, 0]
                                for side, sign in (("left", -1), ("right", 1))}
            paired_cartesian_move(approach_targets, 2., "A_contact_approach")
            try_attach_pair()
            carry_offsets = {}
            grasped = joint_state()
            # Opposing force-confirmed face contacts refine the normal-axis RGB bias.
            tcps = {side: tcp_fk(side, q) for side, q in grasped.items()}
            normal = supply_rotation[:, 0]
            midpoint = (tcps["left"][0] + tcps["right"][0]) / 2
            supply += normal * ((midpoint - supply) @ normal)
            for side, (tcp, rotation) in tcps.items():
                carry_offsets[side] = (rotation.T @ (supply - tcp), rotation.T @ supply_rotation)
            report["target_observations"].append({"phase": "A_carry", "source": "encoder_fk_from_tactile_grasp"})
            step(60)
            previous = supply
            for name, center in (("A_lift", CONFIG["task"]["phase_A"]["lift_center_m"]),
                                 ("A_over_front", CONFIG["task"]["phase_A"]["over_front_center_m"]),
                                 ("A_landing", CONFIG["task"]["phase_A"]["landing_center_m"])):
                record(name, "RUNNING")
                dual_path(previous, center)
                record(name, "REACHED_WITH_SURFACE_GRIPPERS")
                if name in ("A_lift", "A_over_front"):
                    # Stationary suspended windows independently verify payload load.
                    step(180)
                previous = center
            record("A_release", "RELEASING_SURFACE_GRIPPERS")
            sensed_cube()
            seals_required = False
            for side in robots:
                grippers.open(side)
            carry_offsets = None
            report["surface_gripper_validated"] = True
            step(120)
            release_segments = [("outward", .025, 3., "A_release_retreat")]
            if args.five_cube_row and current_task["cube_id"] == 4:
                # The right wrist is beside the wall: unload before clearing it vertically.
                release_segments = [("outward", .005, 3., "A_release_short_retreat"),
                                    ("lift", 1.05, 5., "A_release_lift_clearance"),
                                    ("outward", .020, 3., "A_release_retreat")]
            for motion, distance, duration, phase in release_segments:
                retreat_targets = {}
                for side in robots:
                    tcp, rotation = tcp_fk(side, joint_state()[side])
                    normal = rotation[:, 1] * (-1 if side == "left" else 1)
                    retreat_targets[side] = (tcp + [0, 0, max(0., distance - tcp[2])]
                                             if motion == "lift" else tcp - distance * normal)
                paired_cartesian_move(retreat_targets, duration, phase)
            step(120)
            cube_start, b1_rotation = observe_resting_cube("B1_landing_pose", vision_camera, vision_estimator,
                                                          CONFIG["floor_top_m"])
            landing_error = float(np.linalg.norm(cube_start - previous))
            if landing_error > 0.015:
                raise RuntimeError(f"A release landing error {landing_error:.3f} m")
            record("A_release", "REACHED_AFTER_SUCTION_RELEASE", landing_error_m=landing_error)
            set_contact_drive_mode(True)
            allowed_contacts.discard("right")

            # B1 and B2 choose support/push arms from the current packing slot.
            from fr3_bilateral_b2 import b2_role_seeds, mirror_joints

            support_side = current_task["support_side"]
            push_side = current_task["push_side"]
            push_sign = current_task["push_direction"]
            half = CONFIG["cube_size_m"] / 2
            b1_target = np.asarray(CONFIG["task"]["phase_B1"]["target_center_m"], dtype=float)
            allowed_contacts.clear()
            if args.five_cube_row and abs(rail_states[support_side][1] - .40) > 1e-6:
                position, _ = tcp_fk(support_side, applied_targets[support_side])
                support_clearance_z = max(position[2], CONFIG["floor_top_m"] + half + .25)
                cartesian_move(support_side, [position[0], position[1], support_clearance_z],
                               2.0, "B1_support_lift")
                support_base = rail_states[support_side].copy()
                support_base[1] = .40
                move_robot_base(support_side, support_base)
            cube_start, b1_rotation = observe_resting_cube("B1_contact_pose", vision_camera, vision_estimator,
                                                          CONFIG["floor_top_m"])
            record("B1_contact", "RUNNING", support_arm=support_side)
            staged = "staging_offset_from_rear_m" in current_task
            front_height = .004 if staged else 0.
            front_offset = b1_rotation @ [0, -half, front_height]
            front_normal = b1_rotation[:, 1]
            front_rotation = np.asarray(FRONT_ROTATION) if support_side == "left" else np.diag([-1., 1., -1.])
            front_tool_rotation = b1_rotation @ front_rotation
            contact = cube_start + front_offset - .007 * front_normal
            b1_seed = None
            try:
                b1_plan = json.loads((ROOT / "validation" / "fr3_ab_ik_preflight.json").read_text())
                b1_seed = next(np.asarray(item["joint_positions_rad"], dtype=float)
                               for item in b1_plan.get("waypoints", [])
                               if item.get("arm") == "left" and item.get("waypoint") == "B1_start")
                if support_side == "right":
                    b1_seed = mirror_joints(b1_seed)
            except (FileNotFoundError, StopIteration, KeyError, TypeError, ValueError):
                pass
            q = solve(support_side, contact, front_tool_rotation, continuous=True, seed_override=b1_seed)
            move({support_side: q}, duration=1.0)
            allowed_contacts.add(support_side)
            contact_plan_rotation = b1_rotation.copy()
            push_faces[support_side] = (1, -front_normal, np.array([0., -half, front_height]))
            touch_face(support_side, front_normal, .02, "B1_touch")
            b1_push_start, _ = sensed_cube()
            if push_side is not None:
                b1_target[0] = b1_push_start[0]
            b1_end = b1_target.copy()
            if staged:
                b1_end[1] -= current_task["staging_offset_from_rear_m"]
            b1_end[1] -= half * (np.abs(b1_rotation[1]).sum() - 1) + .0002
            contact_path(support_side, b1_push_start, b1_end, "B1")
            b1_actual, rear_rotation = observe_resting_cube("B2_rear_pose", vision_camera, vision_estimator,
                                                           CONFIG["floor_top_m"])
            b1_progress = float(b1_actual[1] - cube_start[1])
            if b1_progress < 0.20:
                raise RuntimeError(f"B1 cube did not advance enough: {b1_progress:.3f} m")
            if staged:
                record("B1_push", "REACHED_INTERMEDIATE_LANE", progress_m=b1_progress,
                       support_arm=support_side, observed_center_m=b1_actual.tolist())
                align_contact_plane(support_side, front_rotation, "B2_stage_front_alignment")
                position, left_hold_rotation = tcp_fk(support_side, applied_targets[support_side])
                left_hold_z = float(position[2])
                left_hold = NormalForceController(position[1], target_force_N=.65, gain_m_per_Ns=4e-5,
                                                   max_speed_m_s=.0002, max_offset_m=.002)
            else:
                record("B1_push", "REACHED_WITH_PHYSX_CONTACT", progress_m=b1_progress,
                       back_wall_hold_armed=True, support_arm=support_side)

                align_contact_plane(support_side, front_rotation, "B2_wall_alignment")
                position, _ = tcp_fk(support_side, applied_targets[support_side])
                cartesian_move(support_side, position + [0, 0, .004], 1.0, "B2_acquire")
                contact_plan_rotation = rear_rotation.copy()
                push_faces[support_side] = (1, np.array([0., -1., 0.]), np.array([0., -half, 0.]))
                seat_against_wall(support_side, 1, 1, b1_target[1], "B2_acquire", contact_target="wall_back")
                position, left_hold_rotation = tcp_fk(support_side, applied_targets[support_side])
                left_hold_z = float(position[2])
                left_hold = NormalForceController(
                    position[1], target_force_N=B2_CONTROLLER_PARAMETERS["support_approach_preload_N"],
                    gain_m_per_Ns=1e-4,
                    max_speed_m_s=B2_CONTROLLER_PARAMETERS["normal_position_speed_limit_m_s"],
                    max_offset_m=B2_CONTROLLER_PARAMETERS["normal_position_max_offset_m"])
                record("B2_alignment_preload", "TACTILE_PLANE_CONFIRMATION",
                       target_force_N=left_hold.target_force_N)
                aligned = 0
                filtered_alignment_force = 0.
                for _ in range(3600):
                    step()
                    check_current_collision("B2_alignment_preload")
                    wrench = last_force_sample[support_side]["tcp_gravity_corrected_world"]
                    filtered_alignment_force += .05 * (-wrench[1] - filtered_alignment_force)
                    aligned = aligned + 1 if 3.2 <= filtered_alignment_force <= 5.0 else 0
                    if aligned >= 30:
                        break
                else:
                    raise RuntimeError("Rear contact preload did not stabilize")
                report["target_observations"].append({
                    "phase": "B2_preload", "source": "filtered_wrist_normal_force",
                    "force_N": filtered_alignment_force, "yaw_source": "fresh rear RGB",
                })
                left_hold.gain_m_per_Ns = force_gain
            if push_side is None:
                left_hold.target_force_N = B2_CONTROLLER_PARAMETERS["left_target_force_N"]
                stable = 0
                for _ in range(3600):
                    step()
                    check_current_collision("B2_acquire")
                    normal = -last_force_sample[support_side]["tcp_gravity_corrected_world"][1]
                    stable = stable + 1 if 1.0 <= normal <= 3.0 else 0
                    if stable >= 60:
                        break
                else:
                    raise RuntimeError(f"{support_side} B2 preload failed to stabilize")
                record("B2_hold", "FORCE_CONFIRMED",
                       target_force_N=left_hold.target_force_N, support_arm=support_side)
                step(60)
                unload_normal(support_side, 1, 1, "B2_support_unload")
                left_hold = None
                push_faces.clear()
                position, _ = tcp_fk(support_side, applied_targets[support_side])
                cartesian_move(support_side, position + [0, -.035, 0],
                               B2_CONTROLLER_PARAMETERS["retreat_duration_s"], "B2_support_retreat")
                allowed_contacts.clear()
                speed = settle_cube("B2_center_cube")
                record("B2_push_release", "CENTER_CUBE_STRAIGHT_PUSH_COMPLETE", progress_m=0.,
                       max_settle_speed_m_s=speed, back_wall_hold_verified=True)
                return

            record("B2_hold", "MAINTAINED_DURING_APPROACH",
                   target_force_N=left_hold.target_force_N, support_arm=support_side)
            step(60)

            b2_plan = json.loads((ROOT / "validation" / "fr3_b2_collision_plan.json").read_text())
            role_seeds = b2_role_seeds(b2_plan, push_side)
            b2_center, rear_rotation = observe_resting_cube("B2_approach_pose", vision_camera, vision_estimator,
                                                            CONFIG["floor_top_m"])
            b2_rotation = rear_rotation @ role_seeds["push_rotation_world"]
            b2_target = np.asarray(CONFIG["task"]["phase_B2"]["target_center_m"], dtype=float)
            active_height = current_task.get("active_contact_height_m", 0.)
            approach_standoff = current_task.get("approach_standoff_m", .06)
            push_face_offset = np.array([-push_sign * half, 0., active_height])
            b2_transit_z = float(current_task.get("staged_transit_z_m", role_seeds["transit_z_m"]))
            record("B2_high_approach", "RUNNING", push_arm=push_side, direction=push_sign)
            position, _ = tcp_fk(push_side, applied_targets[push_side])
            cartesian_move(push_side, position + [0, 0, max(0., b2_transit_z - position[2])],
                           2.0, "B2_lift_from_measured_state")
            move({push_side: role_seeds["rail_hold_push_q"]})
            push_base = rail_states[push_side].copy()
            push_base[1] = CONFIG["task"]["phase_B2"]["rail_assist"]["base_y_m"]
            record("B2_rail_transfer", "RUNNING", base_y_m=push_base[1], arm=push_side)
            move_robot_base(push_side, push_base)
            outward = -push_sign * rear_rotation[:, 0]
            outside = b2_center + outward * (half + approach_standoff) + [0, 0, active_height]
            high = outside.copy()
            high[2] = b2_transit_z
            if args.five_cube_row:
                guide = json.loads((ROOT / "validation" / "fr3_row_b2_final_feasibility.json").read_text())
                case = next(row for row in guide["cases"] if row["cube_id"] == current_task["cube_id"])
                if (case["status"] != "PASS" or case["push_side"] != push_side
                        or not np.isclose(case["rail_y_m"], rail_states[push_side][1])):
                    raise RuntimeError("The B2 posture guide differs from the active arm or rail station")
                outside_q = cartesian_joint_target(
                    push_side, current_task.get("staged_precontact_q", case["precontact_q"][push_side]),
                    outside, b2_rotation, retarget_guide=True)
                descent_times = np.arange(601) / 120
                lifted = []
                q = outside_q.copy()
                for time_s in descent_times:
                    planning_update("B2: extending the observed contact branch upward")
                    t = time_s / descent_times[-1]
                    blend = 10 * t**3 - 15 * t**4 + 6 * t**5
                    target = outside + blend * (high - outside)
                    q = cartesian_joint_target(push_side, q, target, b2_rotation, retarget_guide=True)
                    lifted.append(q.copy())
                descent = np.asarray(lifted[::-1])
                descent_times, descent = plan_joint_trajectory(
                    descent, max_joint_speed=.4, min_segment_time_s=1/120, sample_hz=120.)
                move({push_side: descent[0]})
                descent_samples = {
                    side: descent if side == push_side else np.repeat(state[None, :], len(descent), axis=0)
                    for side, state in joint_state().items()
                }
                report["target_observations"].append({
                    "phase": "B2_descent", "cube_id": current_task["cube_id"],
                    "source": "RGB_and_contact_plane_retargeted_joint_branch",
                    "outside_position_m": outside.tolist(), "sample_count": len(descent),
                })
                execute_joint_path(descent_samples, "B2_descent",
                                   sampled_trajectory=(descent_times, descent_samples))
            else:
                q = solve(push_side, high, b2_rotation, continuous=True, seed_override=role_seeds["descent"][0])
                move({push_side: q})
                cartesian_move(push_side, outside, 3.0, "B2_descent")
            allowed_contacts.add(push_side)
            cartesian_move(push_side, b2_center + outward * (half + .007) + [0, 0, active_height],
                           2.0, "B2_suction_plane_approach")
            push_faces[push_side] = (0, outward, push_face_offset)
            touch_face(push_side, -outward, .02, "B2_touch")
            settle_b2_contacts(push_side, push_sign, "B2_contact_sync", support_target=.65 if staged else None)
            b2_start, _ = sensed_cube()
            support_start_q = joint_state()[support_side]
            push_sample_count = int(round(B2_CONTROLLER_PARAMETERS["push_duration_s"] * 120)) + 1
            push_times = np.arange(push_sample_count) / 120
            push_samples = {side: [] for side in robots}
            seeds = {side: applied_targets[side].copy() for side in robots}
            start_poses = {side: tcp_fk(side, seed) for side, seed in seeds.items()}
            target_face = b2_target + rear_rotation @ push_face_offset
            target_face[0] -= push_sign * .005
            distance = target_face[0] - start_poses[push_side][0][0]
            for time_s in push_times:
                planning_update("B2: smooth Cartesian force-controlled path")
                t = time_s / push_times[-1]
                blend = 10 * t**3 - 15 * t**4 + 6 * t**5
                for side in robots:
                    position, rotation = start_poses[side]
                    target = position + [distance * blend, 0, 0]
                    seeds[side] = cartesian_joint_target(
                        side, seeds[side], target, rotation, retarget_guide=True)
                    push_samples[side].append(seeds[side].copy())
            push_samples = {side: np.asarray(path) for side, path in push_samples.items()}
            push_peak_speed = max(float(np.max(np.abs(np.diff(path, axis=0)))) * 120
                                  for path in push_samples.values())
            if push_peak_speed > .4:
                raise RuntimeError(f"B2 Cartesian joint speed {push_peak_speed:.3f} rad/s exceeds 0.4")
            hold_errors, push_errors = [], []
            sliding_forces = []

            def b2_cube_center_path(samples, _times):
                centers = []
                for q in samples[push_side]:
                    planning_update("B2: synchronized Cube path")
                    tcp, _ = tcp_fk(push_side, q)
                    center = tcp - rear_rotation @ push_face_offset
                    centers.append([center[0], center[1], b2_start[2]])
                return np.asarray(centers)

            def check_b2_hold(index, count):
                active_force = -push_sign * float(
                    last_force_sample[push_side]["tcp_gravity_corrected_world"][0])
                if staged and .2 * count <= index <= .7 * count:
                    sliding_forces.append(active_force)
                if abs(active_force) > 10.:
                    raise RuntimeError(f"{push_side} B2 push overload: {active_force:.3f} N")
                center, rotation = sensed_cube()
                measured = joint_state()
                support_tcp, _ = tcp_fk(support_side, measured[support_side])
                support_error = float(np.linalg.norm(support_tcp - (center + rotation @ [0, -half, .004])))
                hold_errors.append(support_error)
                if support_error > .02:
                    raise RuntimeError(f"{support_side} B2 hold contact drifted {support_error:.3f} m")
                push_tcp, _ = tcp_fk(push_side, measured[push_side])
                push_error = float(np.linalg.norm(push_tcp - (center + rotation @ push_face_offset)))
                push_errors.append(push_error)
                if push_error > .025:
                    raise RuntimeError(f"{push_side} B2 contact drifted {push_error:.3f} m")

            execute_joint_path(push_samples, "B2_push", check_step=check_b2_hold,
                               cube_center_path=b2_cube_center_path, sampled_trajectory=(push_times, push_samples))
            step(60)
            b2_actual, b2_actual_rotation = sensed_cube()
            issues = checker.check(joint_state(), cube_center=b2_actual, cube_rotation=b2_actual_rotation,
                                   cup_contacts=allowed_contacts, margin_m=0., first_only=True)
            if issues:
                raise RuntimeError(f"B2 contact settle collision: {issues[0]}")
            support_motion = float(np.max(np.abs(joint_state()[support_side] - support_start_q)))
            support_end, _ = tcp_fk(support_side, joint_state()[support_side])
            support_x_motion = float(push_sign * (support_end[0] - start_poses[support_side][0][0]))
            if support_x_motion < max(.002, abs(distance) - .003):
                raise RuntimeError(f"{support_side} B2 support TCP did not follow the Cube")
            progress = float(push_sign * (b2_actual[0] - b2_start[0]))
            required = max(.005, abs(b2_target[0] - b2_start[0]) - .010)
            if progress < required:
                raise RuntimeError(f"B2 Cube advanced {progress:.3f} m, required {required:.3f} m")
            side_probe = None
            if staged:
                # Measure sliding friction before entering the final 5 mm of the side approach.
                baseline = float(np.percentile(sliding_forces, 90))
                side_probe = {"sliding_baseline_N": baseline, "entry_force_N": max(2., baseline + .5),
                              "entry_samples": 3, "samples": 30, "minimum_force_N": 1.5,
                              "mean_force_N": max(1.8, baseline + .4), "max_axial_travel_m": .0005,
                              "target_uncertainty_m": .003}
            seat_against_wall(push_side, 0, push_sign, b2_target[0], "B2_push_seat",
                              follow_side=support_side, contact_target=current_task["contact_target"],
                              probe_override=side_probe)
            unload_normal(push_side, 0, push_sign, "B2_push_unload")
            if args.five_cube_row and current_task.get("contact_target") in ("Cube_1", "Cube_2"):
                record("B2_post_unload_observation", "ACQUIRING_FRESH_ACTIVE_AND_NEIGHBOR_RGB")
                refreshed = observe_placed_cubes(required_names=[current_task["contact_target"]])
                checker.set_other_cubes(refreshed, preserve_unobserved=True)
                fresh_center, fresh_rotation = observe_resting_cube(
                    "B2_post_unload_active_pose", vision_camera, vision_estimator, CONFIG["floor_top_m"])
                neighbor_capture_times = {row["name"]: row["capture_time_s"]
                                          for packet in report["neighbor_observations"]
                                          for row in packet["neighbors"] if "capture_time_s" in row}
                report.setdefault("post_unload_rgb_checks", []).append({
                    "cube_id": current_task["cube_id"], "time_s": float(world.current_time),
                    "neighbor_pose_ages_s": {name: float(world.current_time) - stamp
                                             for name, stamp in neighbor_capture_times.items()},
                    "required_neighbor_names": [current_task["contact_target"]],
                    "occluded_retained_names": report["neighbor_observation_attempts"][-1]["missing_names"],
                    "active_position_m": fresh_center.tolist(), "active_rotation_world": fresh_rotation.tolist(),
                    "contact_target": current_task["contact_target"], "geometry_tolerance_m": .003})
                check_current_collision("B2_post_unload_observation")
                contact_plan_rotation = fresh_rotation.copy()
                record("B2_post_unload_observation", "FRESH_RGB_GEOMETRY_RECHECKED")
            push_faces.pop(push_side, None)
            position, _ = tcp_fk(push_side, applied_targets[push_side])
            if args.five_cube_row and current_task["cube_id"] == 4:
                short_outward = position + [-push_sign * .008, 0, 0]
                elevated = short_outward.copy()
                elevated[2] = max(position[2], b2_transit_z)
                full_outward = elevated + [-push_sign * .027, 0, 0]
                for segment, destination, duration in (
                    ("short_outward", short_outward, B2_CONTROLLER_PARAMETERS["retreat_duration_s"]),
                    ("vertical_clearance", elevated, 5.),
                    ("full_outward", full_outward, B2_CONTROLLER_PARAMETERS["retreat_duration_s"]),
                ):
                    destinations = {side: tcp_fk(side, q)[0] for side, q in joint_state().items()}
                    destinations[push_side] = destination
                    record("B2_push_retreat", "RUNNING", segment=segment)
                    paired_cartesian_move(destinations, duration, "B2_push_retreat")
            else:
                cartesian_move(push_side, position + [-push_sign * .035, 0, 0],
                               B2_CONTROLLER_PARAMETERS["retreat_duration_s"], "B2_push_retreat")
            allowed_contacts.discard(push_side)
            record("B2_push_release", "PUSH_ARM_RETRACTED_SUPPORT_HOLDING", push_arm=push_side)
            settle_cube("B2_push_arm_release")
            if staged:
                left_hold = None
                observed, rotation = observe_resting_cube("B2_staged_rear_pose", vision_camera,
                                                           vision_estimator, CONFIG["floor_top_m"])
                if abs(observed[0] - b2_target[0]) > .004:
                    raise RuntimeError("Staged side seating leaves over 4 mm lateral error")
                align_contact_plane(support_side, front_rotation, "B2_staged_rear_alignment")
                push_faces[support_side] = (1, np.array([0., -1., 0.]), np.array([0., -half, front_height]))
                rear = observed.copy()
                rear[1] = b2_target[1] - half * (np.abs(rotation[1]).sum() - 1.) - .005
                contact_path(support_side, observed, rear, "B2_staged_rear_insert")
                seat_against_wall(support_side, 1, 1, b2_target[1], "B2_staged_rear_seat",
                                  contact_target="wall_back",
                                  probe_override={"entry_force_N": 5., "mean_force_N": 3.,
                                                  "target_uncertainty_m": .003, "samples": 30})
            unload_normal(support_side, 1, 1, "B2_support_unload")
            left_hold = None
            push_faces.clear()
            position, _ = tcp_fk(support_side, applied_targets[support_side])
            cartesian_move(support_side, position + [0, -.035, 0],
                           B2_CONTROLLER_PARAMETERS["retreat_duration_s"], "B2_support_retreat")
            allowed_contacts.clear()
            max_speed = settle_cube("B2")
            record("B2_push_release", "REACHED_WITH_PHYSX_CONTACT", progress_m=progress,
                   max_settle_speed_m_s=max_speed, back_wall_hold_verified=True,
                   support_arm=support_side, push_arm=push_side, support_joint_motion_rad=support_motion,
                   support_x_motion_m=support_x_motion,
                   support_max_contact_error_m=max(hold_errors, default=0.),
                   push_max_contact_error_m=max(push_errors, default=0.))
        placed_bodies = [cube]
        placed_observations = []

        def observe_placed_cubes(count=None, required_names=None):
            if count is None:
                count = current_task["cube_id"] - 1
            names = {f"Cube_{index + 1}" for index in range(count)}
            required = names if required_names is None else set(required_names)
            if not required <= names:
                raise ValueError("Requested neighbor observation must name an existing prior Cube")
            if count == 0:
                return []
            filters = {index: PlanarPoseMedian() for index in range(count)}
            observations, exposure_windows = {}, {index: [] for index in range(count)}
            frames = set()
            active_id = vision_estimator.marker_id
            started = float(world.current_time)
            for _ in range(30):
                step(vision_period)
                render_only()
                frame = vision_camera.get_current_frame(clone=True)
                frame_id = int(frame.get("rendering_frame", 0))
                stamp = float(frame.get("rendering_time", -1.))
                if (frame_id <= 0 or frame_id in frames or stamp < started
                        or not 0 <= float(world.current_time) - stamp <= .20):
                    continue
                frames.add(frame_id)
                rgba = np.asarray(frame.get("rgba", []))
                if rgba.size == 0:
                    continue
                try:
                    for index in range(count):
                        vision_estimator.marker_id = CONFIG["cube_sequence"][index]["marker_id"]
                        result = vision_estimator.estimate(rgba.astype(np.uint8, copy=False))
                        pose = vision_estimator.resting_pose(result, CONFIG["floor_top_m"])
                        if pose is not None:
                            position, rotation = pose
                            exposure_windows[index].append({"frame_id": frame_id, "capture_time_s": stamp,
                                "position_m": position.tolist(), "yaw_rad": float(np.arctan2(rotation[1, 0], rotation[0, 0]))})
                            filtered = filters[index].update(frame_id, stamp, position, rotation)
                            if filtered is not None:
                                observations[index] = filtered
                finally:
                    vision_estimator.marker_id = active_id
                if all(index in observations and 0 <= float(world.current_time) - observations[index][2] <= .20
                       for index in range(count) if f"Cube_{index + 1}" in required):
                    break
            available = {f"Cube_{index + 1}" for index in range(count)
                         if index in observations and 0 <= float(world.current_time) - observations[index][2] <= .20}
            report.setdefault("neighbor_observation_attempts", []).append({
                "cube_id": current_task["cube_id"], "started_s": started, "ended_s": float(world.current_time),
                "required_names": sorted(required), "available_names": sorted(available),
                "missing_names": sorted(names - available),
                "required_marker_ids": [CONFIG["cube_sequence"][index]["marker_id"] for index in range(count)
                                        if f"Cube_{index + 1}" in required],
                "missing_marker_ids": [CONFIG["cube_sequence"][index]["marker_id"] for index in range(count)
                                       if f"Cube_{index + 1}" not in available],
                "available_pose_ages_s": {f"Cube_{index + 1}": float(world.current_time) - observations[index][2]
                                         for index in range(count) if f"Cube_{index + 1}" in available},
                "raw_exposures_by_name": {
                    f"Cube_{index + 1}": exposure_windows[index][-3:] for index in range(count)}})
            missing_required = sorted(required - available)
            if missing_required:
                raise RuntimeError("Fresh median neighbor pose unavailable: " + ", ".join(missing_required))
            observed, rows = [], []
            for index in range(count):
                if f"Cube_{index + 1}" not in available:
                    continue
                position, rotation, capture_time = observations[index]
                observed.append((f"Cube_{index + 1}", position, rotation))
                raw = exposure_windows[index][-3:]
                ref_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                yaw_residuals = np.arctan2(np.sin([sample["yaw_rad"] - ref_yaw for sample in raw]),
                                           np.cos([sample["yaw_rad"] - ref_yaw for sample in raw]))
                rows.append({"name": f"Cube_{index + 1}", "position_m": position.tolist(),
                             "rotation_world": rotation.tolist(), "capture_time_s": capture_time,
                             "filter": "PlanarPoseMedian_three_unique_exposures",
                             "exposure_window": raw,
                             "within_window_position_std_m": np.std([sample["position_m"] for sample in raw], axis=0).tolist(),
                             "within_window_yaw_std_rad": float(np.std(yaw_residuals)),
                             "uncertainty_note": "Window repeatability only; absolute marker bias is not measured here."})
            report.setdefault("neighbor_observations", []).append({
                "cube_id": current_task["cube_id"], "source": "rear_RGB_marker_support_plane",
                "neighbors": rows,
            })
            for row in rows:
                live_neighbors["/World/" + row["name"]] = (
                    np.asarray(row["position_m"], dtype=float),
                    np.asarray(row["rotation_world"], dtype=float),
                    float(row["capture_time_s"]),
                )
            return observed

        sequence = CONFIG.get("cube_sequence", [current_task])
        report["packing_sequence"] = sequence
        report["completed_cubes"] = []
        for cube_index, current_task in enumerate(sequence):
            if cube_index:
                allowed_contacts.clear()
                push_faces.clear()
                left_hold = None
                # Park the arms before returning both rail carriages to the supply station.
                move({side: np.asarray(CONFIG["fr3"]["official_start_q_rad"]) for side in robots})
                step(120)
                set_contact_drive_mode(False)
                for side in robots:
                    base = np.asarray(CONFIG["fr3"]["bases"][side], dtype=float)
                    if not np.allclose(rail_states[side], base):
                        move_robot_base(side, base)
                from fr3_cube_supply import spawn_supply_cube
                next_cube = spawn_supply_cube(stage, world, CONFIG, cube_index, contact_monitor)
                supply_transition = True
                step()
                next_cube.initialize()
                cube = next_cube
                truth_history.clear()
                supply_transition = False
                placed_bodies.append(cube)
            marker_id = current_task.get("marker_id", CONFIG["vision"]["aruco_marker_id"])
            vision_estimator.marker_id = marker_id
            supply_estimator.marker_id = marker_id
            latest_camera_observation = None
            control_center = control_rotation = carry_offsets = contact_plan_rotation = None
            supported_pose_time = None
            push_faces.clear()
            allowed_contacts.clear()
            allowed_contacts.update(robots)
            support_side = current_task["support_side"]
            push_side = current_task["push_side"]
            push_sign = current_task["push_direction"]
            goal = np.asarray(current_task["goal_center_m"], dtype=float).copy()
            if args.five_cube_row:
                placed_observations = observe_placed_cubes()
                checker.set_other_cubes(placed_observations)
                if current_task["contact_target"] in ("Cube_1", "Cube_2"):
                    neighbor = next(row for row in placed_observations if row[0] == current_task["contact_target"])
                    neighbor_extent = half * np.abs(neighbor[2][0]).sum()
                    goal[0] = neighbor[1][0] - push_sign * (neighbor_extent + half)
                if push_side is None:
                    left = next(row for row in placed_observations if row[0] == "Cube_3")
                    right = next(row for row in placed_observations if row[0] == "Cube_4")
                    gap_left = left[1][0] + half * np.abs(left[2][0]).sum()
                    gap_right = right[1][0] - half * np.abs(right[2][0]).sum()
                    if gap_right - gap_left < 2 * half + .002:
                        raise RuntimeError("The measured fifth-Cube slot has less than 2 mm insertion clearance")
                    goal[0] = (gap_left + gap_right) / 2
                CONFIG["task"]["phase_B2"]["target_center_m"] = goal.tolist()
                landing_x = goal[0] if push_side is None else current_task["landing_x_m"]
                opposite_id = {2: "Cube_1", 3: "Cube_2", 4: "Cube_3"}.get(current_task["cube_id"])
                if "staging_offset_from_rear_m" in current_task:
                    landing_x = goal[0] - push_sign * current_task["staged_side_travel_m"]
                    report.setdefault("landing_observations", []).append({
                        "cube_id": current_task["cube_id"], "source": "target_neighbor_RGB_and_staged_side_travel",
                        "landing_x_m": float(landing_x),
                        "stage_y_m": float(goal[1] - current_task["staging_offset_from_rear_m"]),
                    })
                elif opposite_id:
                    opposite = next(row for row in placed_observations if row[0] == opposite_id)
                    extent = half * np.abs(opposite[2][0]).sum()
                    stem_space = (half + current_task["approach_standoff_m"]
                                  + CONFIG["side_suction_tool"]["tcp_side_offset_m"] + .014 + .008)
                    if push_side == "left":
                        landing_x = max(landing_x, opposite[1][0] + extent + stem_space)
                    else:
                        landing_x = min(landing_x, opposite[1][0] - extent - stem_space)
                    seating_travel = push_sign * (goal[0] - landing_x)
                    report.setdefault("landing_observations", []).append({
                        "cube_id": current_task["cube_id"], "source": "rear_RGB_neighbor_boundary_and_tool_geometry",
                        "landing_x_m": float(landing_x), "planned_stem_clearance_m": .008,
                        "opposite_cube": opposite_id, "lateral_seating_travel_m": float(seating_travel),
                    })
                    if seating_travel < .008:
                        raise RuntimeError(
                            f"Cube_{current_task['cube_id']} side approach is blocked by {opposite_id}: "
                            f"clearance requires landing X={landing_x * 1000:.1f} mm, "
                            f"target X={goal[0] * 1000:.1f} mm, remaining push {seating_travel * 1000:.1f} mm (need 8 mm)")
                CONFIG["task"]["phase_B1"]["target_center_m"][0] = landing_x
                CONFIG["task"]["phase_A"]["landing_center_m"][0] = landing_x
            record("supply_arrival", "READY_FOR_RGB_LOCALIZATION",
                   marker_id=marker_id, target_center_m=goal.tolist())
            run_cube()
            final_center, final_rotation = observe_resting_cube(
                "slot_verification", vision_camera, vision_estimator, CONFIG["floor_top_m"])
            if np.linalg.norm(final_center - goal) > .01:
                raise RuntimeError(f"Cube_{cube_index+1}: camera observed more than 10 mm final slot error")
            report["completed_cubes"].append({
                "cube_id": current_task["cube_id"], "target_center_m": goal.tolist(),
                "camera_final_center_m": final_center.tolist(), "rotation_world": final_rotation.tolist(),
            })
        if args.five_cube_row:
            record("packing_verification_park", "RUNNING")
            move({side: np.asarray(CONFIG["fr3"]["official_start_q_rad"]) for side in robots})
            step(120)
            final_row = observe_placed_cubes(count=len(sequence))
            report["final_row_observations"] = {
                "time_s": float(world.current_time), "source": "rear_RGB_marker_support_plane",
                "cubes": [{"cube_id": index + 1, "position_m": center.tolist(),
                           "rotation_world": rotation.tolist()}
                          for index, (_, center, rotation) in enumerate(final_row)],
            }
            from fr3_row_acceptance import final_row_checks
            issues = final_row_checks(report)
            if issues:
                raise RuntimeError("Final row camera verification: " + "; ".join(issues))
        report["status"] = "PASS_FIVE_CUBE_SENSOR_SEQUENCE" if args.five_cube_row else "PASS_A_B1_B2_SENSOR_CONTROL"
        report["ab_status"] = "PASS_SENSOR_TARGETED_PHYSX"
        record("packing_complete", report["status"], completed_cubes=len(report["completed_cubes"]))
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
        if parallel_validator is not None:
            parallel_validator.close()
        if "applied_velocities" in locals() and app.is_running():
            for side, robot in robots.items():
                robot.get_articulation_controller().apply_action(ArticulationAction(
                    joint_velocities=np.zeros(7), joint_indices=joints[side]))
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
        if contact_experiment is not None:
            report["contact_state_recognition"] = contact_experiment.close(report.get("status", "UNVERIFIED"))
        if force_recorder is not None:
            report["wrist_force_sensors"] = force_recorder.close()
            force_recorder.metadata["task_status"] = report["status"]
            (output_directory / "metadata.json").write_text(
                json.dumps(force_recorder.metadata, indent=2, allow_nan=False) + "\n")
        if contact_monitor is not None:
            report["physx_contacts"] = contact_monitor.report()
            contact_monitor.close()
        trajectory_cache["status"] = (
            "COMPLETE" if report.get("status") in ("PASS_FIVE_CUBE_SENSOR_SEQUENCE", "PASS_A_B1_B2_SENSOR_CONTROL") else "INCOMPLETE"
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
    return 0 if report["status"] in ("PASS_FIVE_CUBE_SENSOR_SEQUENCE", "PASS_A_B1_B2_SENSOR_CONTROL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
