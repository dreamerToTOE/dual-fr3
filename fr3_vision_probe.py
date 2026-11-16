"""Probe D455 RGB rendering and ArUco Cube localization independently."""
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
        import omni.syntheticdata
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, UsdGeom, UsdLux
        from fr3_ab_visual_runner import _add_aruco_marker

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")

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
            rsd = stage.GetPrimAtPath(f"{CONFIG['vision']['camera_root_path']}/RSD455")
            if rsd.IsValid():
                rsd_xf = UsdGeom.Xformable(rsd)
                rsd_xf.ClearXformOpOrder()
                rsd_xf.AddTransformOp().Set(Gf.Matrix4d(1.0))

        capture_path = "/World/VisionCaptureCamera"
        capture_prim = UsdGeom.Camera.Define(stage, capture_path)
        capture_prim.AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*CONFIG["vision"]["camera_position_m"]),
                Gf.Vec3d(*CONFIG["vision"]["camera_look_at_m"]),
                Gf.Vec3d(0, 0, 1),
            ).GetInverse()
        )
        capture_prim.CreateFocalLengthAttr().Set(24.0)
        capture_prim.CreateHorizontalApertureAttr().Set(36.0)
        world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 60)
        fallback_camera = world.scene.add(Camera(
            prim_path=capture_path,
            name="vision_capture_camera",
            frequency=CONFIG["vision"]["frequency_hz"],
            resolution=tuple(CONFIG["vision"]["resolution"]),
        ))
        d455_camera = None
        if d455_available:
            d455_camera = world.scene.add(Camera(
                prim_path=CONFIG["vision"]["camera_path"],
                name="d455_color_camera",
                frequency=CONFIG["vision"]["frequency_hz"],
                resolution=tuple(CONFIG["vision"]["resolution"]),
            ))
            d455_camera.set_projection_type("pinhole")
        world.reset()
        fallback_camera.resume()
        if d455_camera is not None:
            d455_camera.resume()

        # The probe owns this render-driven simulation, so its extra physics
        # steps do not interfere with the A/B control clock.
        loop = asyncio.get_event_loop()
        for camera in (fallback_camera, d455_camera):
            if camera is None:
                continue
            future = asyncio.ensure_future(
                omni.syntheticdata.sensors.next_render_simulation_async(
                    camera.get_render_product_path(), 8
                )
            )
            while not future.done():
                app.update()
                loop.run_until_complete(asyncio.sleep(0))
            future.result()

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        results = {}
        for name, camera in (("fallback", fallback_camera), ("d455", d455_camera)):
            if camera is None:
                continue
            rgba = np.asarray(camera.get_rgba())
            if rgba.ndim != 3 or rgba.shape[2] < 3 or rgba.size == 0:
                results[name] = {"shape": list(rgba.shape), "detections": 0}
                continue
            rgb = rgba[..., :3].astype(np.uint8, copy=False)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            results[name] = {
                "shape": list(rgba.shape),
                "detections": 0 if ids is None else int(np.sum(ids.reshape(-1) == CONFIG["vision"]["aruco_marker_id"])),
            }
            cv2.imwrite(f"/tmp/idea_one_{name}_d455_probe.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        result = {
            "snapshot_id": "idea_one_d455_async_probe_2026-11-17",
            "scene_id": CONFIG["scene_id"],
            "d455_asset_available": d455_available,
            "camera_position_m": CONFIG["vision"]["camera_position_m"],
            "camera_look_at_m": CONFIG["vision"]["camera_look_at_m"],
            "aruco_dictionary": CONFIG["vision"]["aruco_dictionary"],
            "aruco_marker_id": CONFIG["vision"]["aruco_marker_id"],
            "results": results,
            "status": "PASS_ARUCO_DETECTED" if any(item.get("detections", 0) > 0 for item in results.values()) else "NO_ARUCO_DETECTION",
            "images": {name: f"/tmp/idea_one_{name}_d455_probe.png" for name in results},
        }
        REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["status"] == "PASS_ARUCO_DETECTED" else 2
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
