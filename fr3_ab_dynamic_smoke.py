"""Load the new four-wall FR3 scene with a real PhysX cube.

This intentionally stops after dynamic settling.  It proves the runtime scene
contract can load without pretending that A/B manipulation already passed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import traceback

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "fr3_four_wall_scene_config.json").read_text())
REPORT = ROOT / "validation" / "fr3_ab_dynamic_smoke.json"
app = SimulationApp({"headless": True})


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
    from pxr import Gf, UsdGeom

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


def main():
    import omni.client
    import omni.timeline
    import omni.usd
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, UsdGeom, UsdPhysics

    result = {
        "scene_id": CONFIG["scene_id"],
        "scope": "dynamic_scene_load_and_settle_only",
        "ab_status": "BLOCKED",
        "ab_blocker": "joint trajectory, suction attach/release, and B1/B2 controller are not implemented",
    }
    omni.usd.get_context().new_stage()
    app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac stage creation failed")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    for name, center, size in CONFIG["static_boxes"]:
        _box(stage, f"/World/Environment/{name}", center, size, collision=True)
    _box(
        stage,
        "/World/Cube",
        CONFIG["supply_cube_center_m"],
        [CONFIG["cube_size_m"]] * 3,
        collision=True,
        dynamic=True,
        mass=CONFIG["physics"]["cube_mass_kg"],
    )
    for side in CONFIG["fr3"]["bases"]:
        rail = f"/World/{side}_rail"
        base = CONFIG["fr3"]["bases"][side]
        beam = CONFIG["rails"]["beam_size_m"]
        local = CONFIG["rails"]["beam_local_center_m"]
        _box(stage, f"{rail}/beam", [base[0] + local[0], base[1] + local[1], base[2] + local[2]], beam)
        _box(stage, f"{rail}/carriage", base[:3], CONFIG["rails"]["carriage_size_m"])

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac asset root unavailable")
    asset_url = None
    for suffix in (CONFIG["fr3"]["asset"], "Isaac/" + CONFIG["fr3"]["asset"]):
        candidate = f"{assets_root.rstrip('/')}/{suffix}"
        status, _ = omni.client.stat(candidate)
        if status == omni.client.Result.OK:
            asset_url = candidate
            break
    if asset_url is None:
        raise RuntimeError(f"FR3 asset unavailable below {assets_root}")
    _add_robot(stage, "left", asset_url)
    _add_robot(stage, "right", asset_url)
    app.update()
    for _ in range(240):
        app.update()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(120):
        app.update()
    articulations = {}
    for side in ("left", "right"):
        articulation = SingleArticulation(f"/World/{side}_fr3")
        articulation.initialize()
        if not articulation.handles_initialized:
            raise RuntimeError(f"{side} FR3 articulation did not initialize")
        articulations[side] = {"num_dof": int(articulation.num_dof)}

    required_paths = [
        "/physicsScene",
        "/World/Cube",
        "/World/Environment/wall_front",
        "/World/Environment/wall_back",
        "/World/Environment/wall_left",
        "/World/Environment/wall_right",
        "/World/left_fr3/fr3_link8/ABSideTool",
        "/World/right_fr3/fr3_link8/ABSideTool",
    ]

    cube = SingleRigidPrim("/World/Cube", name="ab_smoke_cube")
    cube.initialize()
    for _ in range(240):
        app.update()
    position, orientation = cube.get_world_pose()
    linear_velocity = cube.get_linear_velocity()
    speed = float(sum(float(value) ** 2 for value in linear_velocity) ** 0.5)
    expected = CONFIG["supply_cube_center_m"]
    position_error = float(sum((float(position[i]) - expected[i]) ** 2 for i in range(3)) ** 0.5)
    result.update({
        "status": "PASS" if speed <= CONFIG["physics"]["settle_speed_threshold_m_s"] else "BLOCKED",
        "physics_scene": "/physicsScene",
        "dynamic_cube": "/World/Cube",
        "required_prim_presence": {
            path: bool(stage.GetPrimAtPath(path).IsValid()) for path in required_paths
        },
        "articulations": articulations,
        "cube_position_m": [float(value) for value in position],
        "cube_orientation_wxyz": [float(value) for value in orientation],
        "cube_linear_velocity_m_s": [float(value) for value in linear_velocity],
        "cube_speed_m_s": speed,
        "cube_position_error_m": position_error,
    })
    timeline.stop()
    return result


try:
    report = main()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report.get("status") != "PASS":
        raise RuntimeError("dynamic scene did not settle")
except BaseException as exc:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "status": "FAIL",
        "scope": "dynamic_scene_load_and_settle_only",
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }, indent=2, ensure_ascii=False) + "\n")
    traceback.print_exc()
    raise
finally:
    app.close()
