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
