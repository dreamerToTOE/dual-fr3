"""Hold the floor push while confirming a D455 yaw alarm with fresh stationary exposures."""
import math

import numpy as np

from source_hooks import replace_once


def marker_pose_on_floor(pixel_corners, estimator, floor, pose):
    """Reject distorted ArUco corners; fit planar orientation from both square axes."""
    import cv2

    pixels = np.asarray(pixel_corners, dtype=float).reshape(4, 2)
    uv = cv2.undistortPoints(pixels, estimator.camera_matrix, estimator.distortion).reshape(4, 2)
    rays = np.column_stack((uv, np.ones(4))) @ estimator.camera_to_world[:3, :3].T
    origin = estimator.camera_to_world[:3, 3]
    scale = (floor + estimator.cube_half_m + estimator.marker_to_cube_offset_m - origin[2]) / rays[:, 2]
    coords = (origin + scale[:, None] * rays)[:, :2]
    local = estimator.object_points[:, :2]
    centered = coords - coords.mean(axis=0)
    yaw = math.atan2(np.sum(local[:, 0]*centered[:, 1] - local[:, 1]*centered[:, 0]),
                     np.sum(local*centered))
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    residual = float(np.max(np.linalg.norm(centered - local @ rotation[:2, :2].T, axis=1)))
    # A 3 mm residual is already comparable to a 3 degree corner displacement on this marker.
    return (pose[0], rotation) if residual <= .003 else None


def inject(source):
    source = replace_once(source, '        async def sample_vision_async():', '''        front_yaw_gate = {"active": False}

        async def sample_vision_async():''')
    key = '                    filtered_pose = supported_pose_filter.update(\n                        frame_id'
    source = replace_once(source, key, '''                    if "front_slot" in current_task:
                        from front_yaw_check import marker_pose_on_floor
                        supported_pose = marker_pose_on_floor(observation["corners_px"], vision_estimator, SUPPORT_TOP, supported_pose)
                if supported_pose is not None:
                    filtered_pose = supported_pose_filter.update(
                        frame_id''')
    source = replace_once(source, '                    pose = estimator.resting_pose(observation, support_z)', """                    pose = estimator.resting_pose(observation, support_z)
                    if pose is not None and camera is vision_camera and "front_slot" in current_task:
                        from front_yaw_check import marker_pose_on_floor
                        pose = marker_pose_on_floor(observation["corners_px"], estimator, support_z, pose)""")
    before = '''                            raise RuntimeError("RGB observed more than 3 degrees of Cube rotation; contact replan required")
                        control_rotation = supported_rotation'''
    after = '''                            if "front_slot" not in current_task:
                                raise RuntimeError("RGB observed more than 3 degrees of Cube rotation; contact replan required")
                            if not front_yaw_gate["active"]:
                                front_yaw_gate.update(active=True, start=tick, exposures=[], good=0)
                                report.setdefault("front_yaw_holds", []).append({"cube_id": current_task["cube_id"],
                                    "start_step": tick, "phase": contact_monitor.phase, "exposures": front_yaw_gate["exposures"]})
                        if front_yaw_gate["active"]:
                            signed_yaw = math.degrees(math.atan2(float(np.cross(contact_plan_rotation[:, 0],
                                supported_rotation[:, 0])[2]), float(contact_plan_rotation[:, 0] @ supported_rotation[:, 0])))
                            exposures = front_yaw_gate["exposures"]
                            exposures.append([frame_id, round(signed_yaw, 3)])
                            front_yaw_gate["good"] = front_yaw_gate["good"] + 1 if abs(signed_yaw) <= 3. else 0
                            if front_yaw_gate["good"] >= 3:
                                front_yaw_gate["active"] = False
                            elif len(exposures) >= 7 and (all(y > 3. for _, y in exposures[-5:])
                                    or all(y < -3. for _, y in exposures[-5:])):
                                raise RuntimeError(f"Front box yaw exceeds 3 degrees while held: {exposures[-5:]}")
                            else:
                                return
                        control_rotation = supported_rotation'''
    source = replace_once(source, before, after)
    return replace_once(source, '''            for _ in range(count):
                if not app.is_running():''', '''            remaining = count
            while remaining > 0 or front_yaw_gate["active"]:
                if front_yaw_gate["active"] and tick - front_yaw_gate["start"] > 120:
                    raise RuntimeError("Stopped front push: D455 could not confirm box yaw during a one-second hold")
                if not front_yaw_gate["active"]:
                    remaining -= 1
                if not app.is_running():''')
