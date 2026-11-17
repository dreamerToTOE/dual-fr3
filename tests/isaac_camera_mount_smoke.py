"""Visible Isaac Sim check: D455 stand does not obstruct the RGB sensor."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "width": 1400, "height": 900})
try:
    import cv2
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
    from fr3_ab_visual_runner import CONFIG, _add_aruco_marker, _add_camera_mount, _box

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    for name, center, size in CONFIG["static_boxes"]:
        prim = _box(stage, f"/World/Environment/{name}", center, size)
        color = (0.18, 0.45, 0.48) if name == "floor" else (0.70, 0.75, 0.80)
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    cube = _box(stage, "/World/Cube", CONFIG["task"]["phase_B1"]["target_center_m"], [CONFIG["cube_size_m"]] * 3)
    UsdGeom.Gprim(cube).CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.35, 0.05)])
    _add_aruco_marker(stage, CONFIG["cube_size_m"], CONFIG["vision"]["aruco_marker_size_m"])
    _add_camera_mount(stage, CONFIG["vision"])
    assert not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/D455/Mount")))
    light = UsdLux.DomeLight.Define(stage, "/World/Lighting")
    light.CreateIntensityAttr(1500.0)

    for path, eye, target, focal in (
        ("/World/MainCamera", [1.65, -1.7, 2.05], [0, 0.6, 1.0], 14.0),
        ("/World/D455/CalibratedColorCamera", CONFIG["vision"]["camera_position_m"], CONFIG["vision"]["camera_look_at_m"], 1.93),
    ):
        camera_prim = UsdGeom.Camera.Define(stage, path)
        camera_prim.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
        camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        camera_prim.CreateFocalLengthAttr(focal)
        if path == "/World/D455/CalibratedColorCamera":
            camera_prim.CreateHorizontalApertureAttr(3.896)
            camera_prim.CreateVerticalApertureAttr(2.453)
    world = World(stage_units_in_meters=1.0, physics_dt=1/120, rendering_dt=1/60)
    sensor = world.scene.add(Camera(prim_path="/World/D455/CalibratedColorCamera", name="sensor", resolution=(640, 480)))
    viewport = get_active_viewport()
    overview = world.scene.add(Camera(prim_path="/World/MainCamera", name="overview", resolution=(1400, 900), render_product_path=viewport.get_render_product_path()))
    world.reset()
    sensor.initialize()
    overview.initialize()
    sensor.set_projection_type("pinhole")
    viewport.set_active_camera("/World/MainCamera")
    for _ in range(120):
        world.step(render=True)
    sensor_rgba = np.asarray(sensor.get_rgba())
    overview_rgba = np.asarray(overview.get_rgba())
    assert sensor_rgba.shape == (480, 640, 4), sensor_rgba.shape
    assert overview_rgba.shape == (900, 1400, 4), overview_rgba.shape
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    gray = cv2.cvtColor(sensor_rgba[..., :3], cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    assert ids is not None and CONFIG["vision"]["aruco_marker_id"] in ids, ids
    output = ROOT / "validation"
    sensor_png = output / "idea_one_camera_mount_sensor.png"
    overview_png = output / "idea_one_camera_mount_overview.png"
    cv2.imwrite(str(sensor_png), cv2.cvtColor(sensor_rgba[..., :3], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overview_png), cv2.cvtColor(overview_rgba[..., :3], cv2.COLOR_RGB2BGR))
    report = {"status": "PASS", "sensor_shape": list(sensor_rgba.shape), "overview_shape": list(overview_rgba.shape), "marker_ids": ids.reshape(-1).tolist(), "mount_collisions": False, "overview_camera_eye_m": [1.65, -1.7, 2.05], "overview_camera_target_m": [0, 0.6, 1.0], "overview_focal_length_mm": 14.0, "overview_horizontal_aperture_mm": overview.get_horizontal_aperture() * 10.0, "overview_vertical_aperture_mm": overview.get_vertical_aperture() * 10.0, "sensor_image": sensor_png.name, "overview_image": overview_png.name}
    (output / "idea_one_camera_mount_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CAMERA_MOUNT_SMOKE", json.dumps(report), flush=True)
finally:
    app.close()
