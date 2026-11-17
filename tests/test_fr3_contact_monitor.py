import sys
from types import SimpleNamespace

import pytest

from fr3_contact_monitor import PhysxContactMonitor, is_unexpected_robot_contact


@pytest.mark.parametrize("tool", ["ABSideTool", "fr3_link8/ABSideTool"])
def test_contact_policy_allows_only_phase_authorized_cups(tool):
    cup = f"/World/left_fr3/{tool}/cup_r1_c1"
    cube = "/World/Cube"
    assert is_unexpected_robot_contact(cup, cube)
    assert not is_unexpected_robot_contact(cup, cube, {"left"})
    assert not is_unexpected_robot_contact(cube, cup, {"left"})
    assert is_unexpected_robot_contact(cup, cube, {"right"})
    assert not is_unexpected_robot_contact(cup, cube, {"left"}, -0.003)
    assert is_unexpected_robot_contact(cup, cube, {"left"}, -0.0031)
    assert is_unexpected_robot_contact(cube, cup, {"left"}, -0.004)
    assert is_unexpected_robot_contact(cup.replace("cup_r1_c1", "manifold"), cube, {"left"})
    assert is_unexpected_robot_contact(cup, "/World/Environment/wall_back", {"left"})


def test_contact_policy_keeps_environment_and_interarm_contacts_distinct():
    assert not is_unexpected_robot_contact("/World/Cube", "/World/Environment/floor")
    assert is_unexpected_robot_contact("/World/left_fr3/fr3_link3/mesh", "/World/Environment/floor")
    assert is_unexpected_robot_contact("/World/left_fr3/fr3_link3/mesh", "/World/right_fr3/fr3_link3/mesh")
    assert is_unexpected_robot_contact("/World/left_fr3/fr3_link3/mesh", "/World/left_fr3/fr3_link6/mesh")


def test_contact_report_keeps_each_phase_minimum_and_its_physics_step(monkeypatch):
    reporter = SimpleNamespace(CreateThresholdAttr=lambda value: None)
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        PhysicsSchemaTools=SimpleNamespace(intToSdfPath=lambda path: path),
        PhysxSchema=SimpleNamespace(PhysxContactReportAPI=SimpleNamespace(Apply=lambda prim: reporter)),
        UsdPhysics=SimpleNamespace(RigidBodyAPI=object()),
    ))
    physics_interface = SimpleNamespace(subscribe_contact_report_events=lambda callback: callback)
    monkeypatch.setitem(sys.modules, "omni.physx", SimpleNamespace(
        get_physx_simulation_interface=lambda: physics_interface,
    ))
    prim = SimpleNamespace(GetPath=lambda: "/World/right_fr3/fr3_link8", HasAPI=lambda api: True)
    monitor = PhysxContactMonitor(SimpleNamespace(Traverse=lambda: [prim]))
    monitor.allowed_cube_contacts = {"right"}
    cube = "/World/Cube"
    cup = "/World/right_fr3/fr3_link8/ABSideTool/cup_r1_c1"
    header = SimpleNamespace(collider0=cup, collider1=cube, contact_data_offset=1, num_contact_data=2)
    monitor.phase, monitor.physics_step_index = "B1", 10
    monitor._on_contacts([header], [SimpleNamespace(separation=value) for value in (-9.0, -.0007, -.0004)])
    monitor.phase = "B2_push"
    header.collider0, header.collider1 = cube, cup
    for physics_step, separation in ((20, -.0001), (21, -.0003), (22, -.0002)):
        monitor.physics_step_index = physics_step
        monitor._on_contacts([header], [SimpleNamespace(separation=value) for value in (-9.0, separation, 0.0)])

    report = monitor.report()
    pair = " <-> ".join(sorted((cube, cup)))
    assert report["minimum_separation_m"] == -.0007
    assert report["minimum_separation_by_pair_m"][pair] == -.0007
    assert report["minimum_separation_by_phase_pair"]["B1"][pair] == {
        "separation_m": -.0007, "physics_step_index": 10,
    }
    assert report["minimum_separation_by_phase_pair"]["B2_push"][pair] == {
        "separation_m": -.0003, "physics_step_index": 21,
    }
    assert report["contact_count"] == 4
    assert report["unexpected_contact_count"] == 0
