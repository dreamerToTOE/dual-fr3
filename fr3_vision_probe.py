"""Validate D455 ArUco visibility through the Isaac Sim viewport render product."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "fr3_four_wall_scene_config.json").read_text())
REPORT = ROOT / "validation" / "idea_one_d455_aruco_probe.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
    try:
        import cv2
        import omni.client
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Usd, UsdGeom, UsdLux
        from fr3_ab_visual_runner import _add_aruco_marker, _add_camera_mount, _box
        from fr3_vision import ArucoCubeEstimator

        if args.headless:
            raise RuntimeError("Run the vision probe without --headless so the viewport render product is active")
        from omni.kit.viewport.utility import get_active_viewport

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")

        _box(stage, "/World/Floor", (0.0, 0.8, 0.706), (0.9, 1.8, 0.012))
        _box(stage, "/World/BackWall", (0.0, 1.157, 0.787), (0.64, 0.012, 0.15))
        cube = UsdGeom.Cube.Define(stage, "/World/Cube")
        cube.CreateSizeAttr(1.0)
        cube_xf = UsdGeom.Xformable(cube.GetPrim())
        cube_xf.AddTranslateOp().Set(Gf.Vec3d(*CONFIG["task"]["phase_B1"]["target_center_m"]))
        cube_xf.AddScaleOp().Set(Gf.Vec3f(*([CONFIG["cube_size_m"]] * 3)))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.35, 0.05)])
        _add_aruco_marker(stage, CONFIG["cube_size_m"], CONFIG["vision"]["aruco_marker_size_m"])
        light = UsdLux.DomeLight.Define(stage, "/World/Lighting")
        light.CreateIntensityAttr(1500.0)

        assets_root = get_assets_root_path()
        d455_url = f"{assets_root.rstrip('/')}/{CONFIG['vision']['camera_asset']}"
        d455_status, _ = omni.client.stat(d455_url)
        d455_available = d455_status == omni.client.Result.OK
        if d455_available:
            root = stage.DefinePrim(CONFIG["vision"]["camera_root_path"], "Xform")
            root.GetReferences().AddReference(d455_url)
            root_xf = UsdGeom.Xformable(root)
            root_xf.ClearXformOpOrder()
            root_xf.AddTransformOp().Set(Gf.Matrix4d(1.0))
            for _ in range(30):
                app.update()
            rsd_root = stage.GetPrimAtPath(f"{CONFIG['vision']['camera_root_path']}/RSD455")
            rsd_xf = UsdGeom.Xformable(rsd_root)
            rsd_xf.ClearXformOpOrder()
            rsd_xf.AddTransformOp().Set(Gf.Matrix4d(1.0))

        _add_camera_mount(stage, CONFIG["vision"])

        viewport = get_active_viewport()
        capture_path = "/World/D455/CalibratedColorCamera" if d455_available else "/World/VisionViewportCamera"
        capture_prim = UsdGeom.Camera.Define(stage, capture_path)
        capture_prim.AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*CONFIG["vision"]["camera_position_m"]),
                Gf.Vec3d(*CONFIG["vision"]["camera_look_at_m"]),
                Gf.Vec3d(0, 0, 1),
            ).GetInverse()
        )
        capture_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        capture_prim.CreateFocalLengthAttr(1.93 if d455_available else 24.0)
        capture_prim.CreateHorizontalApertureAttr(3.896 if d455_available else 36.0)
        capture_prim.CreateVerticalApertureAttr(2.453 if d455_available else 20.25)
        camera = Camera(
            prim_path=capture_path,
            name="vision_viewport_camera",
            resolution=tuple(CONFIG["vision"]["resolution"]),
            render_product_path=viewport.get_render_product_path(),
        )
        world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 60)
        world.scene.add(camera)
        camera.set_projection_type("pinhole")
        world.reset()
        camera.initialize()
        viewport.set_active_camera(capture_path)
        camera_matrix = np.asarray(camera.get_intrinsics_matrix(), dtype=float)
        camera_to_world = np.linalg.inv(np.asarray(camera.get_view_matrix_ros(), dtype=float))
        estimator = ArucoCubeEstimator(
            camera_matrix, camera_to_world,
            marker_id=CONFIG["vision"]["aruco_marker_id"],
            marker_size_m=CONFIG["vision"]["aruco_marker_size_m"],
            cube_size_m=CONFIG["cube_size_m"],
            marker_surface_offset_m=0.0085,
            dictionary_name=CONFIG["vision"]["aruco_dictionary"],
        )
        for _ in range(60):
            app.update()
            world.step(render=True)

        async def _render_frame():
            await omni.syntheticdata.sensors.next_render_simulation_async(
                camera.get_render_product_path(), 2
            )

        future = asyncio.ensure_future(_render_frame())
        loop = asyncio.get_event_loop()
        while not future.done():
            app.update()
            loop.run_until_complete(asyncio.sleep(0))
        future.result()

        frame = camera.get_current_frame(clone=True)
        rgba = np.asarray(frame["rgba"])
        expected_shape = (CONFIG["vision"]["resolution"][1], CONFIG["vision"]["resolution"][0], 4)
        capture_frame = int(frame["rendering_frame"])
        capture_time = float(frame["rendering_time"])
        if (rgba.shape != expected_shape or capture_frame <= 0
                or not np.isfinite(capture_time) or capture_time <= 0):
            raise RuntimeError(
                f"No valid sensor frame: shape={rgba.shape}, frame={capture_frame}, time={capture_time}")
        rgb = rgba[..., :3].astype(np.uint8, copy=False)
        observation = estimator.estimate(rgb)
        detections = int(observation["status"] == "detected")
        cube_estimate = observation["cube_position_m"]
        # Reference state is read only after the camera-only estimate is complete.
        actual_center = np.asarray(UsdGeom.Xformable(cube.GetPrim()).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()).ExtractTranslation(), dtype=float)
        position_error = (None if cube_estimate is None else
                          float(np.linalg.norm(np.asarray(cube_estimate) - actual_center)))
        image_path = "/tmp/idea_one_viewport_d455_probe.png"
        cv2.imwrite(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        result = {
            "snapshot_id": "idea_one_d455_viewport_probe_2026-11-17",
            "scene_id": CONFIG["scene_id"],
            "d455_asset_available": d455_available,
            "camera_path": capture_path,
            "capture_mode": "d455_calibrated_pinhole_fallback" if d455_available else "generic_pinhole",
            "d455_color_prim": CONFIG["vision"]["camera_path"] if d455_available else None,
            "d455_color_prim_renderable": False if d455_available else None,
            "fallback_reason": (
                "Isaac Sim 4.5 D455 USD color prim carries unsupported cameraProjectionType metadata; "
                "capture uses a pinhole camera with the D455 color intrinsics"
                if d455_available else None
            ),
            "camera_position_m": CONFIG["vision"]["camera_position_m"],
            "camera_look_at_m": CONFIG["vision"]["camera_look_at_m"],
            "aruco_dictionary": CONFIG["vision"]["aruco_dictionary"],
            "aruco_marker_id": CONFIG["vision"]["aruco_marker_id"],
            "frame_shape": list(rgba.shape),
            "capture_frame": capture_frame,
            "capture_time_s": capture_time,
            "camera_matrix": camera_matrix.tolist(),
            "camera_to_world_ros_optical": camera_to_world.tolist(),
            "estimator_only_uses_rgb_and_calibration": True,
            "estimator_status": observation["status"],
            "reprojection_error_px": observation["reprojection_error_px"],
            "detections": detections,
            "cube_position_estimate_m": cube_estimate,
            "cube_position_actual_m": actual_center.tolist(),
            "position_error_m": position_error,
            "status": "PASS_ARUCO_DETECTED" if detections else "NO_ARUCO_DETECTION",
            "image": image_path,
        }
        REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if detections else 2
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
