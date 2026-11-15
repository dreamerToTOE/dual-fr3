"""Static FR3 preview for the current 100 mm, four-wall scene.

This is deliberately geometry-only: it keeps the current scene dimensions and
only swaps the robot/tool visual model to FR3 for layout review.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
import traceback

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parent
# The FR3-only scene config defines the 100 mm / five-goal, four-wall geometry.
CONFIG = json.loads((ROOT / "fr3_four_wall_scene_config.json").read_text())
OUT = ROOT / "preview"
OUT.mkdir(parents=True, exist_ok=True)
print("FR3_PREVIEW_SCRIPT_START", flush=True)

app = SimulationApp({"headless": False, "width": 1280, "height": 800})
print("FR3_PREVIEW_APP_READY", flush=True)

try:
    import omni.client
    import omni.kit.app
    import omni.usd
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
    print("FR3_PREVIEW_IMPORTS_READY", flush=True)

    omni.usd.get_context().new_stage()
    app.update()
    stage = omni.usd.get_context().get_stage()
    assert stage is not None
    print("FR3_PREVIEW_STAGE_READY", flush=True)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    print("FR3_PREVIEW_STAGE_CONFIGURED", flush=True)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    def _material(path: str, color):
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.48)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    mats = {
        "table": _material("/World/Looks/Table", (0.12, 0.16, 0.20)),
        "floor": _material("/World/Looks/Floor", (0.14, 0.45, 0.48)),
        "wall": _material("/World/Looks/Wall", (0.70, 0.76, 0.82)),
        "cube": _material("/World/Looks/Cube", (0.96, 0.48, 0.12)),
        "goal": _material("/World/Looks/Goal", (0.98, 0.76, 0.18)),
        "tool": _material("/World/Looks/Tool", (0.10, 0.12, 0.16)),
        "rail": _material("/World/Looks/Rail", (0.22, 0.25, 0.30)),
    }
    print("FR3_PREVIEW_MATERIALS_READY", flush=True)

    def _bind(prim, mat):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    def _box(path, center, size, mat, collision=False):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*center))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        _bind(cube.GetPrim(), mat)
        if collision:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return cube.GetPrim()

    # Keep the FR3 package's four walls, including the front wall.
    for name, center, size in CONFIG["static_boxes"]:
        if name in {"table", "floor", "wall_front", "wall_back", "wall_left", "wall_right"}:
            mat = mats["table"] if name == "table" else mats["floor"] if name == "floor" else mats["wall"]
            _box("/World/Environment/" + name, center, size, mat, collision=True)
    print("FR3_PREVIEW_ENV_READY", flush=True)
    cube_size = CONFIG["cube_size_m"]
    _box("/World/Cube", CONFIG["supply_cube_center_m"], [cube_size] * 3, mats["cube"], collision=True)

    # The five marks are targets only; they do not add collision geometry.
    for index, center in enumerate(CONFIG["goal_cube_centers_m"]):
        _box(
            f"/World/Targets/Goal_{index:02d}",
            (center[0], center[1], CONFIG["floor_top_m"] + 0.001),
            (cube_size + 0.012, cube_size + 0.012, 0.002),
            mats["goal"],
        )
    print("FR3_PREVIEW_GOALS_READY", flush=True)

    def _asset_url():
        root = get_assets_root_path()
        if not root:
            raise RuntimeError("Isaac asset root is unavailable")
        configured = CONFIG["fr3"]["asset"]
        for suffix in (configured, "Isaac/" + configured):
            url = f"{root.rstrip('/')}/{suffix}"
            result, _ = omni.client.stat(url)
            if result == omni.client.Result.OK:
                return url
        raise RuntimeError("FR3 USD was not found in Isaac assets")

    FR3_START = tuple(CONFIG["fr3"]["official_start_q_rad"])

    def _set_start_targets(root_path: str):
        for i, value in enumerate(FR3_START, start=1):
            prim = stage.GetPrimAtPath(f"{root_path}/fr3_link{i - 1}/fr3_joint{i}")
            if not prim.IsValid():
                continue
            attr = prim.GetAttribute("drive:angular:physics:targetPosition")
            if attr.IsValid():
                attr.Set(math.degrees(value))

    def _add_tool(root_path: str, sign: float):
        tool = UsdGeom.Xform.Define(stage, f"{root_path}/fr3_link8/PreviewSideSuction")
        # L support: down from the flange, then toward the cube side face.
        _box(f"{tool.GetPath()}/vertical", (0.0, 0.0, 0.04), (0.028, 0.028, 0.08), mats["tool"])
        _box(f"{tool.GetPath()}/lateral", (0.0, sign * 0.065, 0.08), (0.018, 0.13, 0.018), mats["tool"])
        _box(f"{tool.GetPath()}/manifold", (0.0, sign * 0.138, 0.08), (0.07, 0.014, 0.07), mats["tool"])
        for row, zoff in enumerate((-0.024, 0.024)):
            for col, xoff in enumerate((-0.024, 0.024)):
                cup = UsdGeom.Cylinder.Define(stage, f"{tool.GetPath()}/cup_{row}_{col}")
                cup.CreateAxisAttr(UsdGeom.Tokens.y)
                cup.CreateRadiusAttr(0.010)
                cup.CreateHeightAttr(0.010)
                UsdGeom.Xformable(cup.GetPrim()).AddTranslateOp().Set(
                    Gf.Vec3d(xoff, sign * 0.150, 0.08 + zoff)
                )
                cup.CreateDisplayColorAttr([Gf.Vec3f(0.03, 0.03, 0.03)])

    bases = CONFIG["fr3"]["bases"]
    for side, base in bases.items():
        root = f"/World/{side}_fr3"
        robot = stage.DefinePrim(root, "Xform")
        robot.GetReferences().AddReference(_asset_url())
        xf = UsdGeom.Xformable(robot)
        xf.AddTranslateOp(opSuffix="preview_base").Set(Gf.Vec3d(base[0], base[1], base[2]))
        xf.AddRotateZOp(opSuffix="preview_yaw").Set(math.degrees(base[3]))
        _set_start_targets(root)
        _add_tool(root, -1.0 if side == "left" else 1.0)

        rail = f"/World/{side}_rail"
        beam_local = CONFIG["rails"].get("beam_local_center_m", [0.0, 0.0, 0.0])
        beam_size = CONFIG["rails"].get("beam_size_m", [0.15, 1.5, 0.025])
        _box(f"{rail}/beam", (base[0] + beam_local[0], base[1] + beam_local[1], base[2] + beam_local[2]), beam_size, mats["rail"])
        _box(f"{rail}/carriage", (base[0], base[1], base[2] - 0.01), CONFIG["rails"].get("carriage_size_m", [0.17, 0.16, 0.02]), mats["rail"])

    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Dome")
    dome.CreateIntensityAttr(700.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.96, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/Key")
    sun.CreateIntensityAttr(1500.0)
    sun.CreateAngleAttr(0.4)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, -35.0))

    def _camera(path, eye, target, ortho=False, scale=2.2):
        cam = UsdGeom.Camera.Define(stage, path)
        # Z-up cannot be used as the camera up vector for a straight-down view.
        up = Gf.Vec3d(0, 1, 0) if abs(eye[2] - target[2]) > 1.0 else Gf.Vec3d(0, 0, 1)
        cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up).GetInverse())
        cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 20.0))
        if ortho:
            cam.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
            cam.CreateHorizontalApertureAttr().Set(11.0 * scale / 2.2)
            cam.CreateVerticalApertureAttr().Set(8.25 * scale / 2.2)
        else:
            cam.CreateFocalLengthAttr().Set(25.0)

    _camera("/World/Cameras/Perspective", (1.45, -1.25, 1.55), (0.0, 0.70, 0.76))
    # Perspective top view is more reliable than orthographic capture in Isaac Sim 4.5.
    _camera("/World/Cameras/Top", (0.0, 0.70, 3.2), (0.0, 0.70, 0.72), False)

    # Let referenced FR3 meshes and materials resolve before capture.
    for _ in range(900):
        app.update()

    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    vp = get_active_viewport()
    for label, camera, filename in (
        ("perspective", "/World/Cameras/Perspective", OUT / "fr3_four_wall_perspective.png"),
        ("top", "/World/Cameras/Top", OUT / "fr3_four_wall_top.png"),
    ):
        vp.camera_path = camera
        for _ in range(20):
            app.update()
        request = capture_viewport_to_file(vp, str(filename))
        future = asyncio.ensure_future(request.wait_for_result(completion_frames=30))
        for _ in range(240):
            app.update()
            if future.done():
                break
        future.result()
        deadline = time.monotonic() + 5.0
        while (not filename.exists() or filename.stat().st_size <= 1000) and time.monotonic() < deadline:
            app.update()
            time.sleep(0.01)
        assert filename.exists() and filename.stat().st_size > 1000
        print("FR3_FOUR_WALL_CAPTURE", label, filename, flush=True)

    stage.GetRootLayer().Export(str(OUT / "fr3_four_wall_preview.usda"))
    print("FR3_FOUR_WALL_PREVIEW_READY", flush=True)
except BaseException as exc:
    print("FR3_PREVIEW_ERROR", repr(exc), flush=True)
    traceback.print_exc()
    raise
finally:
    app.close()
