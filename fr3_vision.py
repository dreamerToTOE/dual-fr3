"""ArUco Cube observations from RGB and fixed camera calibration only."""
from __future__ import annotations

from collections import deque
import cv2
import numpy as np


class PlanarPoseMedian:
    """Causal median of three distinct optical frames, timestamped at the middle exposure."""

    def __init__(self):
        self._samples = deque(maxlen=3)

    def reset(self):
        self._samples.clear()

    def update(self, frame_id, capture_time_s, position, rotation, reference_rotation=None):
        position, rotation = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
        if (not isinstance(frame_id, (int, np.integer)) or frame_id <= 0
                or not np.isfinite(capture_time_s) or position.shape != (3,) or rotation.shape != (3, 3)
                or not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation))):
            raise ValueError("Pose filtering requires a finite pose and optical exposure identity")
        if self._samples and (frame_id <= self._samples[-1][0] or capture_time_s <= self._samples[-1][1]):
            return None
        self._samples.append((frame_id, float(capture_time_s), position.copy(), rotation.copy()))
        if len(self._samples) < 3:
            return None
        reference = self._samples[0][3] if reference_rotation is None else np.asarray(reference_rotation)
        reference_yaw = np.arctan2(reference[1, 0], reference[0, 0])
        yaws = np.array([np.arctan2(sample[3][1, 0], sample[3][0, 0]) for sample in self._samples])
        residuals = np.arctan2(np.sin(yaws - reference_yaw), np.cos(yaws - reference_yaw))
        yaw = reference_yaw + np.median(residuals)
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                             [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        return np.median([sample[2] for sample in self._samples], axis=0), rotation, self._samples[1][1]


class ArucoCubeEstimator:
    def __init__(self, camera_matrix, camera_to_world, marker_id=23,
                 marker_size_m=0.105, cube_size_m=0.15,
                 marker_surface_offset_m=0.0085, dictionary_name="DICT_4X4_50"):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).copy()
        self.camera_to_world = np.asarray(camera_to_world, dtype=np.float64).copy()
        if (self.camera_matrix.shape != (3, 3)
                or not np.all(np.isfinite(self.camera_matrix))
                or self.camera_matrix[0, 0] <= 0 or self.camera_matrix[1, 1] <= 0):
            raise ValueError("camera_matrix must be a finite 3x3 pinhole matrix")
        if (self.camera_to_world.shape != (4, 4)
                or not np.all(np.isfinite(self.camera_to_world))
                or not np.allclose(self.camera_to_world[3], [0, 0, 0, 1])
                or not np.allclose(self.camera_to_world[:3, :3].T @ self.camera_to_world[:3, :3],
                                   np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(self.camera_to_world[:3, :3]), 1.0, atol=1e-3)):
            raise ValueError("camera_to_world must be a rigid ROS optical-to-world transform")
        if (not np.isfinite([marker_size_m, cube_size_m, marker_surface_offset_m]).all()
                or marker_size_m <= 0 or cube_size_m <= 0 or marker_surface_offset_m < 0):
            raise ValueError("Cube and marker dimensions must be positive and finite")
        self.marker_id = int(marker_id)
        self.cube_half_m = cube_size_m / 2.0
        self.marker_to_cube_offset_m = self.cube_half_m + marker_surface_offset_m
        half = marker_size_m / 2.0
        # IPPE_SQUARE expects top-left, top-right, bottom-right, bottom-left.
        self.object_points = np.array([[-half, half, 0], [half, half, 0],
                                       [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
        # shortcut: the renderer is ideal pinhole; use calibrated distortion before hardware deployment.
        self.distortion = np.zeros((5, 1), dtype=np.float64)
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if not 0 <= self.marker_id < len(self.dictionary.bytesList):
            raise ValueError("marker_id is outside the selected ArUco dictionary")
        self.parameters = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters")
                           else cv2.aruco.DetectorParameters_create())
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 3
        self.detector = (cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
                         if hasattr(cv2.aruco, "ArucoDetector") else None)

    def estimate(self, rgb):
        """Return a detection or an explicit miss; never read simulation state."""
        result = {"status": "empty_frame", "cube_position_m": None,
                  "marker_position_m": None, "cube_rotation_world": None,
                  "rvec_camera": None, "tvec_camera_m": None,
                  "reprojection_error_px": None, "pixel_center": None, "corners_px": None,
                  "observed_ids": []}
        rgb = np.asarray(rgb)
        if rgb.size == 0:
            return result
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or rgb.dtype != np.uint8:
            raise ValueError("Expected a uint8 RGB or RGBA image")
        gray = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        result["status"] = "marker_not_visible"
        if ids is None:
            return result
        ids = np.asarray(ids).reshape(-1)
        result["observed_ids"] = ids.tolist()
        matches = np.flatnonzero(ids == self.marker_id)
        if not len(matches):
            return result
        image_points = np.asarray(corners[int(matches[0])], dtype=np.float64).reshape(4, 2)
        result["corners_px"] = image_points.tolist()
        solved, rvec, tvec = cv2.solvePnP(
            self.object_points, image_points, self.camera_matrix, self.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        result["status"] = "pose_unsolved"
        if (not solved or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec))
                or float(tvec[2, 0]) <= 0):
            return result
        marker_rotation_camera, _ = cv2.Rodrigues(rvec)
        marker_rotation_world = self.camera_to_world[:3, :3] @ marker_rotation_camera
        marker_position_world = (self.camera_to_world[:3, :3] @ tvec.reshape(3)
                                 + self.camera_to_world[:3, 3])
        cube_position_world = (marker_position_world
                               - marker_rotation_world[:, 2] * self.marker_to_cube_offset_m)
        projected, _ = cv2.projectPoints(self.object_points, rvec, tvec,
                                         self.camera_matrix, self.distortion)
        reprojection_error = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(4, 2) - image_points) ** 2, axis=1))))
        result.update(status="detected", cube_position_m=cube_position_world.tolist(),
                      marker_position_m=marker_position_world.tolist(),
                      cube_rotation_world=marker_rotation_world.tolist(),
                      rvec_camera=rvec.reshape(3).tolist(), tvec_camera_m=tvec.reshape(3).tolist(),
                      reprojection_error_px=reprojection_error,
                      pixel_center=image_points.mean(axis=0).tolist())
        return result

    def resting_pose(self, result, support_z_m):
        """Recover an upright Cube from marker corners and a calibrated support plane."""
        pixels = np.asarray(result.get("corners_px"), dtype=float)
        if pixels.shape != (4, 2) or not np.all(np.isfinite(pixels)) or not np.isfinite(support_z_m):
            return None
        rays_xy = cv2.undistortPoints(pixels, self.camera_matrix, self.distortion).reshape(4, 2)
        rays_world = np.column_stack((rays_xy, np.ones(4))) @ self.camera_to_world[:3, :3].T
        ray_z = rays_world[:, 2]
        if np.any(np.abs(ray_z) < 1e-6):
            return None
        camera_origin = self.camera_to_world[:3, 3]
        center_z = support_z_m + self.cube_half_m
        distance = (center_z + self.marker_to_cube_offset_m - camera_origin[2]) / ray_z
        if np.any(distance <= 0):
            return None
        corners_world = camera_origin + distance[:, None] * rays_world
        marker_center = corners_world.mean(axis=0)
        edge_x = ((corners_world[1] - corners_world[0]) + (corners_world[2] - corners_world[3])) / 2
        marker_size = self.object_points[1, 0] - self.object_points[0, 0]
        if np.linalg.norm(edge_x[:2]) < marker_size / 2:
            return None
        ux, uy = edge_x[:2] / np.linalg.norm(edge_x[:2])
        rotation = np.array([[ux, -uy, 0.], [uy, ux, 0.], [0., 0., 1.]])
        expected = marker_center + self.object_points @ rotation.T
        if np.max(np.linalg.norm(expected - corners_world, axis=1)) > 0.1 * marker_size:
            return None
        return np.array([marker_center[0], marker_center[1], center_z]), rotation
