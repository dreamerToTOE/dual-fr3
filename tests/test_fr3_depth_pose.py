from pathlib import Path
import sys

import cv2
import numpy as np

from fr3_vision import ArucoCubeEstimator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from s0_optical_capture import depth_marker_pose


def test_aligned_optical_depth_recovers_the_raised_marker_plane():
    intrinsics = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
    calibration = np.eye(4)
    calibration[:3, :3] = cv2.Rodrigues(np.array([.2, -.4, .1]))[0]
    calibration[:3, 3] = [.5, 1.2, 1.8]
    estimator = ArucoCubeEstimator(intrinsics, calibration)
    marker_rotvec = np.array([2.8, .2, -.1])
    marker_rotation = cv2.Rodrigues(marker_rotvec)[0]
    marker_origin = np.array([.03, -.02, .8])
    pixels = cv2.projectPoints(estimator.object_points, marker_rotvec, marker_origin,
                               intrinsics, estimator.distortion)[0].reshape(4, 2)
    grid_y, grid_x = np.indices((480, 640))
    normal = marker_rotation[:, 2]
    ray_normal = (normal[0] * (grid_x - 320) / 700 + normal[1] * (grid_y - 240) / 700 + normal[2])
    depth = ((marker_origin @ normal) / ray_normal)[..., None]
    expected_rotation = calibration[:3, :3] @ marker_rotation
    expected_center = (calibration[:3, :3] @ marker_origin + calibration[:3, 3]
                       - expected_rotation[:, 2] * estimator.marker_to_cube_offset_m)

    pose = depth_marker_pose(estimator, pixels, depth)
    assert pose["status"] == "detected"
    np.testing.assert_allclose(pose["position_m"], expected_center, atol=1e-9)
    np.testing.assert_allclose(pose["cube_rotation_world"], expected_rotation, atol=1e-9)
    assert depth_marker_pose(estimator, pixels, np.full_like(depth, np.inf))["status"] == "too_few_valid_depth_samples"
