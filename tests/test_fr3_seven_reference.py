import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fr3_contact_monitor import _contact_normal_impulses

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from seven_reference import SevenReference


def test_absolute_axis_projection_is_independent_of_collider_order():
    for sign in (-1, 1):
        point = SimpleNamespace(impulse=(0., 9., -.02 * sign), normal=(0., 0., sign))
        total, axes = _contact_normal_impulses([point])
        assert total == pytest.approx(.02)
        assert axes == pytest.approx([0., 0., .02])


def test_current_box_contacts_and_load_transfer():
    config = {"cube_size_m": .15, "floor_top_m": .7, "static_boxes": [["floor", [0, 0, .69], [.9, .9, .02]]],
              "physics": {"cube_mass_kg": 1.2, "gravity_m_s2": [0, 0, -10.]}}
    model = SevenReference(config, object_positions={1: [0, 0, .775], 8: [-.15, 0, .925]})
    cube = "/World/Cube_10"

    def hit(other, label, axis, impulse):
        axes = [0., 0., 0.]
        axes[axis] = impulse
        return {"collider0": cube, "collider1": other, "label": label, "current_step_observed": True,
                "normal_impulse_Ns": impulse, "normal_axis_impulse_abs_Ns": axes}

    def frame(step, contacts, sealed=False, z=.925):
        return {"active_cube_id": 10, "physics_step": step, "cube_position_m": [0, 0, z],
                "cube_rotation_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "attachments": {"left": sealed},
                "cube_contacts": {"valid": True, "pairs": contacts}}

    bottom = hit("/World/Cube_1", "cube_neighbor", 2, .1)
    cup = hit("/World/left_fr3/cup", "left_cup", 0, .008)
    side = hit("/World/Cube_8", "cube_neighbor", 0, .02)
    rear = hit("/World/Environment/wall_back", "wall_back", 1, .03)
    for step in range(1, 17):
        assert model.update(frame(step, [bottom]))["state"] == "S0"
    assert model.update(frame(17, [bottom, cup]))["state"] == "S1"
    assert model.update(frame(18, [bottom, cup], True))["state"] == "S2"
    for step in range(19, 33):
        assert model.update(frame(step, [bottom, cup], True))["state"] == "S2"
    partial = hit("/World/Cube_1", "cube_neighbor", 2, .04)
    for step in range(33, 37):
        transfer = model.update(frame(step, [partial, cup], True))
    assert transfer["state"] == "S3" and transfer["reason"] == "unloading_bottom"
    assert model.update(frame(37, [cup], True, z=1.))["state"] == "S4"
    assert model.update(frame(38, [bottom, side, cup]))["state"] == "S5"
    out = model.update(frame(39, [bottom, side, rear, cup]))
    assert out["state"] == "S6" and out["horizontal_objects"] == sorted([side["collider1"], rear["collider1"]])
    assert model.update(frame(40, [{**bottom, "current_step_observed": False}]))["state"] == "UNKNOWN"
    assert model.update(frame(41, []))["state"] == "UNKNOWN"
    old = {k: v for k, v in partial.items() if k != "normal_axis_impulse_abs_Ns"}
    out = model.update(frame(42, [old], True))
    assert out["support_load_fraction"] is None and out["state"] == "UNKNOWN"
    unoriented = hit("/World/Cube_99", "cube_neighbor", 0, .02)
    assert model.update(frame(43, [bottom, unoriented]))["state"] == "S0"
    assert model.update(frame(44, [bottom, cup, side, rear, unoriented]))["state"] == "S6"
    assert model.update(frame(45, [bottom, cup, side, unoriented]))["state"] == "UNKNOWN"
