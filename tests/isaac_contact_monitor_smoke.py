"""Run with Isaac Sim's python.sh; exercise enabled PhysX colliders and guard."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import json
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
from isaacsim.core.api import World
from fr3_contact_monitor import PhysxContactMonitor

def box(stage, path, xyz, dynamic=False):
    prim = UsdGeom.Cube.Define(stage, path)
    prim.CreateSizeAttr(.1)
    prim.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(prim.GetPrim()).CreateRestOffsetAttr(0.0)
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
        UsdPhysics.MassAPI.Apply(prim.GetPrim()).CreateMassAttr(1.)
    return prim

cases = [
 ('link_wall', '/World/left_fr3/fr3_link3/collider', '/World/Environment/wall_back', set(), True, .08),
 ('cup_cube', '/World/left_fr3/fr3_link8/ABSideTool/cup_r1_c1', '/World/Cube', {'left'}, False, .098),
 ('cup_cube_deep', '/World/left_fr3/fr3_link8/ABSideTool/cup_r1_c1', '/World/Cube', {'left'}, True, .08),
 ('manifold_cube', '/World/left_fr3/fr3_link8/ABSideTool/manifold', '/World/Cube', {'left'}, True, .08),
 ('cup_wall', '/World/left_fr3/fr3_link8/ABSideTool/cup_r1_c1', '/World/Environment/wall_back', {'left'}, True, .08),
 ('interarm', '/World/left_fr3/fr3_link3/collider', '/World/right_fr3/fr3_link3/collider', set(), True, .08),
]
results=[]
for name, path0, path1, allowed, expect_collision, x1 in cases:
    omni.usd.get_context().new_stage()
    app.update()
    stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world=World(stage_units_in_meters=1., physics_dt=1./120., rendering_dt=1./60.)
    UsdPhysics.Scene.Get(stage, '/physicsScene').CreateGravityMagnitudeAttr(0.)
    # Every collider stays enabled; authorized cup contact is only 2 mm deep.
    box(stage,path0,(0,0,1),True)
    box(stage,path1,(x1,0,1),path1.startswith('/World/right_fr3'))
    monitor=PhysxContactMonitor(stage)
    monitor.allowed_cube_contacts=allowed
    monitor.phase=name
    world.reset()
    for _ in range(3):
        world.step(render=False)
    result=monitor.report()
    assert result['contact_count']>0, (name,'no contacts observed',result)
    assert bool(result['first_collision'])==expect_collision,(name,result)
    assert bool(result['unexpected_contact_count'])==expect_collision,(name,result)
    caught=False
    try:
        monitor.raise_if_collision()
    except RuntimeError:
        caught=True
    assert caught==expect_collision,(name,'guard did not raise')
    result['case']=name
    result['passed']=True
    results.append(result)
    monitor.close()
    world.stop()
    World.clear_instance()
print('FR3_CONTACT_MONITOR_SMOKE=' + json.dumps(results),flush=True)
(ROOT / 'validation/fr3_contact_monitor_smoke.json').write_text(json.dumps(results,indent=2)+'\n')
app.close()
