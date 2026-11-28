import sys
from pathlib import Path

import numpy as np

SCENES = Path(__file__).resolve().parents[1] / "scenes"
sys.path.insert(0, str(SCENES / "w760_300mm_two_layer"))

from markerless_rows import build, rear_depth_only
from dynamic_withdraw import install as dynamic_withdraw
from front_row_a import floor_guide
from markerless_tracking import install as markerless
from observed_b1 import install as observed_b1
from observed_front_bearing import contact_evidence, install as observed_front_bearing
from observed_rear import install as observed_rear
from observed_rear import rear_landing
from observed_transport import front_placement_goal, install as observed_transport, register_transport


def test_registered_carry_connects_the_observed_endpoints():
    guide = floor_guide(2)
    pickup = np.array([.004, -.306, .787])
    landing = np.asarray(guide["landing_m"]) + [.004, .010, 0]
    live = register_transport(guide, pickup, landing)
    phases = [phase for phase in live["phases"] if phase["carrying"]]
    np.testing.assert_allclose(phases[0]["cube_centers_m"][0], pickup)
    np.testing.assert_allclose(phases[-1]["cube_centers_m"][-1], landing)
    for first, second in zip(phases, phases[1:]):
        np.testing.assert_allclose(first["cube_centers_m"][-1], second["cube_centers_m"][0])
        for arm, rail in first["rail_paths_m"].items():
            assert np.isclose(rail[-1], second["rail_paths_m"][arm][0])
    assert guide["landing_m"] != live["landing_m"]


def test_rear_neighbor_landing_follows_the_fitted_box_and_calibrated_wall():
    wall = [0., 1.2, .86]
    config = {"cube_size_m": .15, "floor_top_m": .712,
              "static_boxes": [("wall_back", wall, [.8, .1, .3])]}
    left_box = np.array([-.305, 1.075, .787])
    poses = {"Cube_1": (left_box, np.eye(3))}
    before = rear_landing(config, 3, poses)
    np.testing.assert_allclose(before, [-.02915, .735, .787])
    wall[1] += .025
    left_box[0] += .006
    np.testing.assert_allclose(rear_landing(config, 3, poses) - before, [.006, .025, 0.])


def test_front_landing_uses_center_of_a_positive_sub_two_mm_slot():
    config = {"cube_size_m": .15, "floor_top_m": .712}
    poses = [(name, np.array(center), np.eye(3)) for name, center in (
        ("Cube_5", [0., 1.075, .787]),
        ("Cube_13", [-.1505, .925, .787]),
        ("Cube_14", [.15047, .925, .787]),
    )]
    np.testing.assert_allclose(front_placement_goal(config, 5, poses), [-.000015, .925, .787])


def test_full_runner_assembles_with_observed_skills():
    generated = markerless(rear_depth_only(build()))
    generated = observed_rear(dynamic_withdraw(observed_b1(observed_transport(generated))))
    code = compile(generated, "<observed skills>", "exec")
    assert {"START_REAR", "STOP_AFTER_REAR"} <= set(code.co_names) | set(
        name for fn in code.co_consts if hasattr(fn, "co_names") for name in fn.co_names)
    assert "def move_to_observed_precontact(" in generated
    assert "def dynamic_withdraw(" in generated
    assert "NormalForceController = partial(NormalForceController, max_force_N=None)" in generated
    for removed_abort in ("declared acceptance envelope", "exceeded its wrist wrench limit",
                          "dual-contact force exceeds 10 N", "B wrist envelope exceeded",
                          "side seal unloading overload", "seam transport exceeded its wrench limits"):
        assert removed_abort not in generated
    assert "contact_monitor.raise_if_collision()" in generated
    assert "Surface Gripper seal lost while carrying the Cube" in generated
    assert "move_independent_pair(targets, phase, planning_margin_m=planning_margin_m)" in generated
    assert "move_independent_pair({arm: path[-1] for arm, path in samples.items()}, phase, routes=samples)" in generated
    assert 'execute_joint_path(samples, motion_phase, check_step=monitor,' in generated
    pickup_standoff = generated.index('paired_cartesian_move(approach_targets, 2., "A_contact_approach")')
    confirm_arrival = generated.index("wait_for_precontact(", pickup_standoff)
    allow_touch = generated.index("allowed_contacts.update(robots)", pickup_standoff)
    assert pickup_standoff < confirm_arrival < allow_touch
    assert "distance = FACE_STANDOFF_M if slot" not in generated
    assert 'hint = None if slot == 1 else role["rail_hold_push_q"]' in generated
    assert "continuation_issues=validation" in generated
    assert "use_cache=not continuation" in generated
    lateral_precontact = generated.index('cartesian_move(push_side, final_precontact')
    lateral_arrival = generated.index("wait_for_precontact(push_side, final_precontact", lateral_precontact)
    lateral_permission = generated.index("allowed_contacts.add(push_side)", lateral_precontact)
    assert lateral_precontact < lateral_arrival < lateral_permission


def test_front_bearing_requires_motion_before_the_loaded_stall():
    travel = np.arange(15) * 3.75e-6
    noise = 5.e-6 * np.sin(np.arange(15))
    force = np.full(30, 6.)
    contact = np.column_stack((np.r_[travel, travel[-1] + noise], force))
    assert contact_evidence(contact) is not None
    assert contact_evidence(np.column_stack((np.r_[noise, noise], force))) is None
    assert contact_evidence(np.column_stack((np.arange(30) * 3.75e-6, force))) is None
    assert contact_evidence(np.column_stack((contact[:, 0], force / 2))) is None


def test_front_bearing_experiment_has_a_separate_physx_audit():
    generated = markerless(rear_depth_only(build()))
    generated = observed_rear(dynamic_withdraw(observed_b1(observed_transport(generated))))
    generated = observed_front_bearing(generated)
    compile(generated, "<observed front bearing>", "exec")
    start = generated.index("            def observe_front_rear_bearing(")
    end = generated.index("            def verify_contact(", start)
    transition = generated[start:end]
    assert "contact_monitor.cube_contact_snapshot()" not in transition
    assert "observe_front_rear_bearing(support_side, rear_name)" in generated
    assert 'verify_contact("rear_seat", [rear_path], support_side, 1, 1, audit_only=True)' in generated
