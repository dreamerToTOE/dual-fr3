import sys
from types import SimpleNamespace

import pytest

from fr3_contact_monitor import CubeContactTracker, PhysxContactMonitor, is_unexpected_robot_contact


def test_cube_contact_lifecycle_distinguishes_live_contacts_and_historical_minimum():
    tracker = CubeContactTracker()
    cube, wall = "/World/Cube", "/World/Environment/wall_back"
    tracker.update(cube, wall, event_type=0, physics_step=10, separation_m=-1e-5, normal_impulse_Ns=.01)
    assert tracker.snapshot(10)["contacts"]["wall_back"]
    assert tracker.snapshot(10)["loaded_contacts"]["wall_back"]
    tracker.advance(11)
    sleeping = tracker.snapshot(11)
    assert sleeping["contacts"]["wall_back"]
    assert not sleeping["pairs"][0]["current_step_observed"]
    assert sleeping["pairs"][0]["last_observed_physics_step"] == 10
    # A positive-separation manifold is not geometric touch, even after penetration.
    tracker.update(wall, cube, event_type=2, physics_step=12, separation_m=2.4e-5, normal_impulse_Ns=0)
    assert not tracker.snapshot(12)["contacts"]["wall_back"]
    assert not tracker.snapshot(12)["loaded_contacts"]["wall_back"]
    # Force-bearing contact offsets use independent PhysX impulse, not wrist force.
    tracker.update(cube, wall, event_type=2, physics_step=13, separation_m=2.4e-5, normal_impulse_Ns=.02)
    assert not tracker.snapshot(13)["contacts"]["wall_back"]
    assert tracker.snapshot(13)["loaded_contacts"]["wall_back"]
    tracker.advance(14)
    assert not tracker.snapshot(14)["loaded_contacts"]["wall_back"]
    # LOST headers have no contact data; they must still clear the lifecycle.
    tracker.update(cube, wall, event_type=1, physics_step=14)
    assert not tracker.snapshot(14)["pairs"]
    tracker.update(cube, wall, event_type=2, physics_step=12, separation_m=-.01)
    assert not tracker.snapshot(14)["pairs"]  # late event cannot resurrect a contact
    assert not tracker.snapshot(12)["valid"]
    assert tracker.snapshot(12)["contacts"]["wall_back"] is None


def test_cube_contact_multi_cup_pairs_and_step_reset():
    tracker = CubeContactTracker()
    cube = "/World/Cube"
    cups = [f"/World/left_fr3/ABSideTool/cup_r1_c{i}" for i in (1, 2)]
    for cup in cups:
        tracker.update(cup, cube, event_type=0, physics_step=3, separation_m=0)
    tracker.update(cube, cups[0], event_type=1, physics_step=4)
    assert tracker.snapshot(4)["contacts"]["left_cup"]
    tracker.update(cube, cups[1], event_type=1, physics_step=4)
    assert not tracker.snapshot(4)["contacts"]["left_cup"]
    tracker.update(cube, "/World/Environment/floor", event_type=0, physics_step=5,
                   separation_m=1e-5, normal_impulse_Ns=.1)
    assert tracker.snapshot(5)["loaded_contacts"]["floor"]
    tracker.advance(1)
    assert not tracker.snapshot(1)["pairs"]


def test_monitor_routes_cube_wall_impulses_and_zero_data_lost_without_changing_robot_guard(monkeypatch):
    applied = []
    reporter = SimpleNamespace(CreateThresholdAttr=lambda value: None)
    def apply(prim):
        applied.append(str(prim.GetPath()))
        return reporter
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        PhysicsSchemaTools=SimpleNamespace(intToSdfPath=lambda path: path),
        PhysxSchema=SimpleNamespace(PhysxContactReportAPI=SimpleNamespace(Apply=apply)),
        UsdPhysics=SimpleNamespace(RigidBodyAPI=object()),
    ))
    monkeypatch.setitem(sys.modules, "omni.physx", SimpleNamespace(
        get_physx_simulation_interface=lambda: SimpleNamespace(
            subscribe_contact_report_events=lambda callback: callback)))
    prims = [SimpleNamespace(GetPath=lambda path=path: path, HasAPI=lambda api: True)
             for path in ("/World/Cube", "/World/left_fr3/fr3_link8")]
    monitor = PhysxContactMonitor(SimpleNamespace(Traverse=lambda: prims))
    assert applied == ["/World/Cube", "/World/left_fr3/fr3_link8"]
    assert monitor.reporter_count == monitor.cube_reporter_count == 1
    header = SimpleNamespace(collider0="/World/Cube", collider1="/World/Environment/wall_left",
                             contact_data_offset=0, num_contact_data=2, type=0)
    points = [SimpleNamespace(separation=2e-5, impulse=(2., 3., 4.), normal=(0., 0., 2.)),
              SimpleNamespace(separation=3e-5, impulse=(-1., 4., 0.), normal=(-2., 0., 0.))]
    monitor.physics_step_index = 7
    monitor._on_contacts([header], points)
    snapshot = monitor.cube_contact_snapshot()
    assert not snapshot["contacts"]["wall_left"]
    assert snapshot["loaded_contacts"]["wall_left"]
    assert snapshot["pairs"][0]["normal_impulse_Ns"] == 5.
    assert snapshot["pairs"][0]["normal_axis_impulse_abs_Ns"] == [1., 0., 4.]
    assert monitor.contact_count == monitor.unexpected_contact_count == 0
    monitor.raise_if_collision()
    monitor.physics_step_index = 8
    assert not monitor.cube_contact_snapshot()["loaded_contacts"]["wall_left"]
    header.type, header.num_contact_data = 1, 0
    monitor._on_contacts([header], [])
    assert monitor.cube_contact_snapshot()["pairs"] == []


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
