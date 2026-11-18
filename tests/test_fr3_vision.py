import inspect

import cv2
import numpy as np
import pytest

from fr3_vision import ArucoCubeEstimator, PlanarPoseMedian


def marker_image(camera_matrix, rvec, tvec, size=0.105, marker_id=23):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = (cv2.aruco.generateImageMarker(dictionary, marker_id, 400)
              if hasattr(cv2.aruco, "generateImageMarker")
              else cv2.aruco.drawMarker(dictionary, marker_id, 400))
    h = size / 2
    corners = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=float)
    pixels, _ = cv2.projectPoints(corners, rvec, tvec, camera_matrix, np.zeros(5))
    homography = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [399, 0], [399, 399], [0, 399]]),
        pixels.reshape(4, 2).astype(np.float32))
    gray = cv2.warpPerspective(marker, homography, (640, 480), borderValue=255)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def test_camera_estimate_recovers_tilted_cube_with_nonidentity_calibration():
    k = np.array([[700., 0, 320], [0, 700., 240], [0, 0, 1.]])
    calibration = np.eye(4)
    calibration[:3, :3] = cv2.Rodrigues(np.array([0.1, -0.35, 0.2]))[0]
    calibration[:3, 3] = [0.5, 1.2, 1.8]
    rvec = np.array([2.8, 0.2, 0.1])
    tvec = np.array([0.03, -0.02, 0.8])
    rgb = marker_image(k, rvec, tvec)
    estimate = ArucoCubeEstimator(k, calibration).estimate(rgb)
    marker_rotation = calibration[:3, :3] @ cv2.Rodrigues(rvec)[0]
    expected = calibration[:3, :3] @ tvec + calibration[:3, 3] - marker_rotation[:, 2] * (0.075 + 0.0085)
    assert estimate["status"] == "detected"
    assert np.linalg.norm(np.array(estimate["cube_position_m"]) - expected) < 0.008
    assert estimate["reprojection_error_px"] < 0.5
    # A constant world-Z subtraction is wrong for this tilted cube.
    naive = np.array(estimate["marker_position_m"]) - [0, 0, 0.0835]
    assert np.linalg.norm(naive - expected) > 0.1
    assert "truth" not in inspect.signature(ArucoCubeEstimator).parameters
    assert list(inspect.signature(ArucoCubeEstimator.estimate).parameters) == ["self", "rgb"]


def test_explicit_missing_marker_and_invalid_calibration():
    k = np.array([[700., 0, 320], [0, 700., 240], [0, 0, 1.]])
    estimator = ArucoCubeEstimator(k, np.eye(4))
    missing = estimator.estimate(np.full((480, 640, 4), 255, dtype=np.uint8))
    assert missing["status"] == "marker_not_visible"
    assert missing["cube_position_m"] is None
    wrong_id = estimator.estimate(marker_image(k, np.array([2.8, .2, .1]), np.array([0., 0., .8]), marker_id=7))
    assert wrong_id["status"] == "marker_not_visible"
    assert wrong_id["observed_ids"] == [7]
    assert wrong_id["cube_position_m"] is None
    with pytest.raises(ValueError, match="rigid"):
        ArucoCubeEstimator(k, np.diag([2., 1., 1., 1.]))
    with pytest.raises(ValueError, match="RGB"):
        estimator.estimate(np.zeros((480, 640), dtype=np.uint8))


def test_resting_cube_pose_uses_image_and_support_plane_for_nonzero_yaw():
    k = np.array([[700., 0, 320], [0, 700., 240], [0, 0, 1.]])
    calibration = np.eye(4)
    calibration[:3, :3] = cv2.Rodrigues(np.array([2.9, .1, -.13]))[0]
    calibration[:3, 3] = [.15, .92, 1.65]
    support_z = .712
    center = np.array([.07, .77, support_z + .075])
    yaw = np.radians(17.)
    cube_rotation = cv2.Rodrigues(np.array([0., 0., yaw]))[0]
    marker = center + [0., 0., .0835]
    world_to_camera = calibration[:3, :3].T
    rvec = cv2.Rodrigues(world_to_camera @ cube_rotation)[0]
    tvec = world_to_camera @ (marker - calibration[:3, 3])
    estimator = ArucoCubeEstimator(k, calibration)
    detection = estimator.estimate(marker_image(k, rvec, tvec))
    assert detection["status"] == "detected"
    measured_center, measured_rotation = estimator.resting_pose(detection, support_z)
    measured_yaw = np.arctan2(measured_rotation[1, 0], measured_rotation[0, 0])
    assert np.linalg.norm(measured_center - center) < .001
    assert abs(measured_yaw - yaw) < np.radians(.5)
    assert estimator.resting_pose(detection, support_z - .35) is None
    assert estimator.resting_pose({"corners_px": detection["corners_px"][::-1]}, support_z) is None
    assert estimator.resting_pose({"corners_px": None}, support_z) is None


def test_pose_median_rejects_one_frame_outlier_and_duplicate_exposures():
    model = PlanarPoseMedian()
    for frame in (1, 2, 3):
        value = model.update(frame, frame / 15, np.zeros(3), np.eye(3), np.eye(3))
    assert value is not None
    outlier = cv2.Rodrigues(np.array([0., 0., np.radians(15.)]))[0]
    center, rotation, stamp = model.update(4, 4 / 15, [10., -5., 4.], outlier, np.eye(3))
    np.testing.assert_allclose(center, np.zeros(3))
    np.testing.assert_allclose(rotation, np.eye(3))
    assert stamp == 3 / 15
    assert model.update(4, 5 / 15, np.zeros(3), np.eye(3), np.eye(3)) is None
    model.reset()
    assert model.update(5, 5 / 15, np.zeros(3), np.eye(3), np.eye(3)) is None


def test_pose_median_keeps_persistent_rotation_fault_and_wraps_at_pi():
    model = PlanarPoseMedian()
    for frame in (1, 2, 3):
        model.update(frame, frame / 15, np.zeros(3), np.eye(3), np.eye(3))
    changed = cv2.Rodrigues(np.array([0., 0., np.radians(4.)]))[0]
    fault_samples = []
    for frame in (4, 5, 6):
        _, rotation, _ = model.update(frame, frame / 15, np.zeros(3), changed, np.eye(3))
        fault_samples.append(rotation[0, 0] < np.cos(np.radians(3.)))
    assert not fault_samples[0]
    assert any(fault_samples)
    model.reset()
    reference = cv2.Rodrigues(np.array([0., 0., np.radians(179.)]))[0]
    for frame, angle in enumerate((179.5, -179.5, 178.5), 1):
        rotation = cv2.Rodrigues(np.array([0., 0., np.radians(angle)]))[0]
        value = model.update(frame, frame / 15, np.zeros(3), rotation, reference)
    _, rotation, _ = value
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    error = yaw - np.radians(179.5)
    assert abs(np.arctan2(np.sin(error), np.cos(error))) < 1e-10
