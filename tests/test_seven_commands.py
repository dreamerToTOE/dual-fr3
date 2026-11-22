import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "scenes/w760_300mm_two_layer/experiments"))
from seven_commands import targets


def test_measured_world_tcp_anchors_rotated_command_and_omits_missing_arm():
    def pose(q, link):
        assert link == "fr3_link8"
        quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rotation = quarter_turn if q[3] else np.eye(3)
        return SimpleNamespace(translation=q[:3], rotation=SimpleNamespace(matrix=lambda: rotation))

    kinematics = SimpleNamespace(pose=pose)
    sensor = {"tcp_positions_m": {"left": [0.2, 0.8, 1.0]}}
    row = {"physics_step": 20, "left": {"q_rad": [0., 0., 0., 0.], "tcp_position_m": [0.2, 0.8, 1.0]},
           "control": {"position_targets_rad": {"left": [0.01, 0., 0., 1.]}}}
    arguments = (sensor, kinematics, {"left": np.pi/2}, {"left": np.array([0.01, 0., 0.])})
    result = targets(row, *arguments)
    assert result == {"left": pytest.approx([0.19, 0.8, 1.0])}
    row["left"]["tcp_position_m"][0] = 0.21
    with pytest.raises(ValueError, match="paired clip at 20"):
        targets(row, *arguments)
