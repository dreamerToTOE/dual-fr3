import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fr3_wrist_ft import WristForceRecorder, wrench_feedback


@pytest.mark.parametrize("use_fk", [False, True])
def test_actual_sampler_logs_both_origins_at_the_same_physics_step(monkeypatch, tmp_path, use_fk):
    config = json.loads((Path(__file__).resolve().parents[1] / "fr3_four_wall_scene_config.json").read_text())
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    position = np.array([.4, .8, 1.0])
    orientation = np.array([2**-.5, 0., 0., 2**-.5])
    def tool_pose():
        assert not use_fk, "The runtime sensor must use encoder FK"
        return position, orientation

    monkeypatch.setitem(sys.modules, "isaacsim.core.prims", SimpleNamespace(
        SingleXFormPrim=lambda path: SimpleNamespace(get_world_pose=tool_pose)))
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.rotations", SimpleNamespace(
        quat_to_rot_matrix=lambda quat: rotation, rot_matrix_to_quat=lambda matrix: orientation))
    robots, offsets, expected_sensor = {}, {}, {}
    force, couple = np.array([3., -4., 2.]), np.array([.2, .1, -.05])
    for side, sign in (("left", -1), ("right", 1)):
        offset = np.array([0., sign * .155, .08])
        offsets[side] = offset
        cfg = config["wrist_force_sensors"][side]
        gravity_force = rotation.T @ (cfg["mass_kg"] * np.array([0., 0., -9.81]))
        gravity = np.r_[gravity_force, np.cross(cfg["center_of_mass_m"], gravity_force)]
        sensor = np.r_[force, couple + np.cross(offset, force)]
        expected_sensor[side] = sensor
        raw = -sensor - gravity
        robots[side] = SimpleNamespace(
            _articulation_view=SimpleNamespace(get_link_index=lambda name: 1,
                                             _metadata=SimpleNamespace(joint_names=["wrist_ft_joint"])),
            get_measured_joint_forces=lambda raw=raw: np.vstack([np.zeros(6), raw]),
            get_joint_positions=lambda joint_indices: np.zeros(7),
            get_joint_velocities=lambda joint_indices: np.zeros(7),
        )
    def forward_kinematics(frame, q):
        assert frame == "fr3_link8"
        np.testing.assert_array_equal(q, np.zeros(7))
        return position.reshape(3, 1), rotation

    solvers = {side: SimpleNamespace(compute_forward_kinematics=forward_kinematics) for side in robots}
    recorder = WristForceRecorder(robots, {side: np.arange(7) for side in robots}, config, tmp_path,
                                  solvers=solvers if use_fk else None)
    returned = recorder.sample(1.25, 150, "B2_push")
    recorder.close()
    with gzip.open(tmp_path / "wrench.jsonl.gz", "rt") as stream:
        saved = json.loads(stream.readline())
        assert not stream.readline()
    assert saved == returned
    assert saved["physics_step"] == 150 and saved["time_s"] == 1.25
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["schema_version"] == 2
    assert ("joint encoders" in metadata["tcp_position_source"]) == use_fk
    for side in robots:
        record = saved[side]
        np.testing.assert_allclose(record["gravity_corrected_local"], expected_sensor[side], atol=1e-12)
        np.testing.assert_allclose(record["tcp_gravity_corrected_local"], np.r_[force, couple], atol=1e-12)
        np.testing.assert_allclose(record["tcp_gravity_corrected_world"],
                                   np.r_[rotation @ force, rotation @ couple], atol=1e-12)
        np.testing.assert_allclose(record["tcp_position_m"], position + rotation @ offsets[side])
        np.testing.assert_allclose(metadata["tcp_offset_sensor_m"][side], offsets[side])
        assert record["gravity_corrected_world"][:3] == record["tcp_gravity_corrected_world"][:3]


@pytest.mark.parametrize("raw,offset", [([0.] * 5, [0., .155, .08]),
                                        ([0., 0., np.nan, 0., 0., 0.], [0., .155, .08]),
                                        ([0.] * 6, [0., np.inf, .08])])
def test_feedback_rejects_invalid_input(raw, offset):
    with pytest.raises(ValueError):
        wrench_feedback(raw, np.eye(3), .3, [0., .06, .055], [0., 0., -9.81], offset)
