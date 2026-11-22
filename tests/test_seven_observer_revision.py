import numpy as np

from test_seven_observer import CONFIG, sample
from seven_observer import SevenObserver


BACK = "/World/Environment/wall_back"
NEIGHBOR = "/World/Cube_8"


def observer():
    return SevenObserver(CONFIG, "fusion_compensated", revised=True)


def slow_push(estimator, *, follows_command, side_evidence="none"):
    neighbors = {NEIGHBOR: {"position_m": [.15, .9198, .075],
                            "rotation_world": np.eye(3),
                            "capture_time_s": 0., "received_time_s": 0.}}
    for step in range(73):
        commanded = min(step, 30) * 1.5e-6 + max(step - 30, 0) * .5e-6
        actual = commanded if follows_command else 0.
        force = .7 + step * .025
        shear = -force if side_evidence == "shear" else 0.
        side_force = -force if side_evidence == "aligned" else 0.
        row = sample(step / 120, center=(0., .9198 + actual, .075),
                     left=(shear, -force, 0.), right=(side_force, 0., 0.))
        row["camera"]["frame_id"] = step
        left_command = row["tcp_positions_m"]["left"] + [0., commanded - actual, 0.]
        row["tcp_command_positions_m"] = {"left": left_command}
        if side_evidence == "aligned":
            right_tcp = np.array([-.075, .9198, .075])
            row["cup_normals_world"]["right"] = [1., 0., 0.]
            row["tcp_positions_m"]["right"] = right_tcp
            row["tcp_command_positions_m"]["right"] = right_tcp + [commanded, 0., 0.]
        estimate = estimator.update(row, neighbors=neighbors)
    return estimate


def test_slow_normal_push_does_not_confirm_a_wall_from_low_absolute_speed():
    old = SevenObserver(CONFIG, "fusion_compensated")
    assert BACK in slow_push(old, follows_command=True)["boundaries"]
    result = slow_push(observer(), follows_command=True)
    assert result["state"] == "S1"
    assert result["boundaries"] == []


def test_wall_contact_does_not_borrow_wrist_shear_for_the_side_neighbor():
    result = slow_push(observer(), follows_command=False, side_evidence="shear")
    assert (result["state"], result["boundaries"]) == ("S5", [BACK])


def test_wall_contact_is_held_with_load_and_cleared_when_push_follows_again():
    estimator = observer()
    assert slow_push(estimator, follows_command=False)["boundaries"] == [BACK]
    for step in range(73, 205):
        advance = max(step - 140, 0) * 1e-6
        row = sample(step / 120, center=(0., .9198 + advance, .075), left=(0., -2.5, 0.))
        row["camera"]["frame_id"] = step
        row["tcp_command_positions_m"] = {
            "left": row["tcp_positions_m"]["left"] + [0., .000066, 0.],
        }
        result = estimator.update(row)
        if step == 140:
            assert (result["state"], result["boundaries"]) == ("S5", [BACK])
    assert (result["state"], result["boundaries"]) == ("S1", [])


def test_independent_evidence_on_two_signed_normals_can_confirm_s6():
    result = slow_push(observer(), follows_command=False, side_evidence="aligned")
    assert result["state"] == "S6"
    assert set(result["boundaries"]) == {BACK, NEIGHBOR}


def test_s2_support_height_can_be_checked_with_inferred_opposed_grip():
    estimator = observer()
    start = sample(0., left=(0., 1., -.5), right=(0., -1., -.5))
    assert estimator.update(start)["state"] == "S2"
    row = sample(.1, left=(0., 1., -.5), right=(0., -1., -.5))
    row["camera"]["position_m"] = [0., 0., .125]
    assert estimator.update(row)["state"] == "S2"


def test_s3_possible_support_transfer_precedes_the_high_wrist_load_check():
    estimator = observer()
    start = sample(0., left=(0., 1., -1.), right=(0., -1., -1.))
    assert estimator.update(start)["state"] == "S2"
    transfer = sample(.12, center=(0., 0., .077), left=(0., 1., -4.6),
                      right=(0., -1., -4.6))
    assert estimator.update(transfer)["state"] == "S3"
    suspended = sample(.24, center=(0., 0., .39), left=(0., 1., -6.),
                       right=(0., -1., -6.))
    assert estimator.update(suspended)["state"] == "S4"


def test_s0_separation_from_the_box_face_precedes_full_retreat():
    estimator = observer()
    for step in range(16):
        row = sample(step / 120)
        row["camera"]["frame_id"] = step
        estimate = estimator.update(row)
    assert estimate["state"] != "S0"
    states = []
    for step in range(16, 41):
        gap = (step - 15) * .010 / 25
        row = sample(step / 120)
        row["camera"]["frame_id"] = step
        row["tcp_positions_m"]["left"] -= [0., gap, 0.]
        row["tcp_positions_m"]["right"] += [0., gap, 0.]
        states.append(estimator.update(row))
    detached = next(estimate for estimate in states if estimate["state"] == "S0")
    assert detached["tools"] == {"left": "separated", "right": "separated"}
    assert detached["tool_retreat_complete"] == {"left": False, "right": False}
    retreated = estimator.update(sample(.5, separated=True))
    assert retreated["state"] == "S0"
    assert retreated["tool_retreat_complete"] == {"left": True, "right": True}
