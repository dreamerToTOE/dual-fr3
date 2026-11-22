import numpy as np

from test_seven_observer import CONFIG, sample
from u4_contact_observer import REAR, SIDE, U4ContactObserver


def front_sample(tick, command_increment=.00010, lateral_force=.0):
    t = tick / 120.
    row = sample(t, center=(0., .920, .075))
    row["cup_normals_world"]["right"] = [0., 1., 0.]
    row["tcp_positions_m"]["right"] = [0., .845, .075]
    row["right_wrench_world"] = [lateral_force, -3. - 1.5*t, 0., 0., 0., 0.]
    row["right_command_y_m"] = .845 + t * command_increment
    return row


def test_independent_axes_require_current_wrist_evidence():
    neighbor = {SIDE: {"position_m": [.15,.92,.075], "rotation_world": np.eye(3),
                       "capture_time_s": 0., "received_time_s": .2}}
    observer = U4ContactObserver(CONFIG)
    for tick in range(1, 90):
        result = observer.update(front_sample(tick, lateral_force=-.5), neighbor)
        if tick < 24:
            assert result["side_loaded"] is None
        if tick < 61:
            assert result["rear_loaded"] is None
    assert result["state"] == "S6" and set(result["boundaries"]) == {SIDE, REAR}
    # Losing lateral response invalidates that axis even when the box remains optically close.
    result = observer.update(front_sample(90), neighbor)
    assert result["side_loaded"] is None and result["rear_loaded"] is True
    assert result["state"] == "UNKNOWN"
    # A fresh observer must not convert near-wall appearance and constant command into contact.
    observer = U4ContactObserver(CONFIG)
    for tick in range(1, 90):
        result = observer.update(front_sample(tick, command_increment=0, lateral_force=-.5), neighbor)
    assert result["rear_loaded"] is None and result["boundaries"] == [SIDE]


def test_handover_recovers_side_bearing_with_biased_optics_and_updates_on_loss():
    observer = U4ContactObserver(CONFIG, handover=True)
    untreated = U4ContactObserver(CONFIG, handover=True)
    neighbor = {SIDE: {"position_m": [.142, .920, .075], "rotation_world": np.eye(3),
                       "capture_time_s": 0., "received_time_s": 0.}}
    for tick in range(90):
        row = front_sample(tick, lateral_force=-.5)
        row["tcp_command_positions_m"] = {"right": [0., row["right_command_y_m"], .075]}
        if tick < 6:
            row["cup_normals_world"]["left"] = [1., 0., 0.]
            row["tcp_positions_m"]["left"] = [-.075, .920, .075]
            row["left_wrench_world"] = [-4., 0., 0., 0., 0., 0.]
        result = observer.update(row, neighbor)
        if tick >= 6:
            untouched = untreated.update(row, neighbor)
            assert SIDE not in untouched["boundaries"]
    assert result["state"] == "S6" and result["rear_loaded"] is True
    assert untreated.update(front_sample(90), neighbor)["side_loaded"] is None

    for tick, shear, expected in ((90, .5, True), (91, .5, True), (92, .5, False),
                                 (93, .0, None), (94, -.5, None), (95, -.5, None), (96, -.5, True)):
        row = front_sample(tick, lateral_force=shear)
        row["tcp_command_positions_m"] = {"right": [0., row["right_command_y_m"], .075]}
        result = observer.update(row, neighbor)
        assert result["side_loaded"] is expected
        assert result["state"] == ("S6" if expected else "S5")


def test_projected_marker_pose_uses_its_own_exposure_and_delivery():
    observer = U4ContactObserver(CONFIG, handover=True)
    packet = front_sample(24)["camera"]
    packet.update(supported_capture_time_s=.10, supported_position_m=[.002, .920, .075],
                  supported_rotation_world=np.eye(3))

    assert observer.supported_pose(packet, .19) is None
    assert observer.supported_pose(packet, .20) == ([.002, .920, .075], 0.)
    assert observer.supported_pose(None, .36) is None


def test_handover_bridges_only_a_bounded_missing_side_sample():
    observer = U4ContactObserver(CONFIG, handover=True)
    neighbor = {SIDE: {"position_m": [.142, .920, .075], "rotation_world": np.eye(3),
                       "capture_time_s": 0., "received_time_s": 0.}}
    for tick in range(90):
        row = front_sample(tick, lateral_force=-.5)
        row["tcp_command_positions_m"] = {"right": [0., row["right_command_y_m"], .075]}
        if tick < 6:
            row["cup_normals_world"]["left"] = [1., 0., 0.]
            row["tcp_positions_m"]["left"] = [-.075, .920, .075]
            row["left_wrench_world"] = [-4., 0., 0., 0., 0., 0.]
        result = observer.update(row, neighbor)
    assert result["side_loaded"] is True

    row = front_sample(90, lateral_force=0.)
    row["tcp_command_positions_m"] = {"right": [0., row["right_command_y_m"], .075]}
    bridged = observer.update(row, neighbor)
    assert bridged["side_loaded"] is True and bridged["side_hold_used"]

    for tick in range(91, 116):
        row = front_sample(tick, lateral_force=0.)
        row["tcp_command_positions_m"] = {"right": [0., row["right_command_y_m"], .075]}
        result = observer.update(row, neighbor)
    assert result["side_loaded"] is None and not result["side_hold_used"]
