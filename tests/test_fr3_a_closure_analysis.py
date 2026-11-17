import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fr3_a_closure_analysis import closure_metrics


def test_rigid_translation_passes_but_interpolated_grasp_drift_fails():
    poses = np.tile(np.eye(4), (5, 2, 1, 1))
    poses[:, :, :3, :3] = Rotation.from_euler("xyz", [.2, -.4, .6]).as_matrix()
    poses[:, :, :3, 3] = np.linspace([0, 0, .8], [0, .7, 1.1], 5)[:, None, :]
    poses[:, 1, 0, 3] += .15
    assert closure_metrics(poses)["status"] == "PASS"
    # The old trajectory reaches valid endpoints but drifts between them.
    poses[2, 1, 0, 3] += .0005
    poses[2, 1, :3, :3] = poses[2, 1, :3, :3] @ Rotation.from_rotvec([0, 0, .001]).as_matrix()
    result = closure_metrics(poses)
    assert result["status"] == "FAIL"
    assert result["maximum_translation_drift_sample"] == 2
    assert result["maximum_rotation_drift_sample"] == 2
    assert np.isclose(result["maximum_translation_drift_m"], .0005)
    assert np.isclose(result["maximum_rotation_drift_rad"], .001)
    poses[2, 1, :3, :3] = 0
    with pytest.raises(ValueError, match="proper rigid"):
        closure_metrics(poses)
