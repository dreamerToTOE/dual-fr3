"""Collect aligned D455 depth and supply RGB observations during the normal A approach."""

import json

import cv2
import numpy as np


def depth_marker_pose(estimator, corners_px, depth_m):
    """Fit the visible marker plane using optical Z, then intersect its corner rays."""
    corners = np.asarray(corners_px, dtype=float)
    depth = np.asarray(depth_m).squeeze()
    if corners.shape != (4, 2):
        return {"status": "marker_not_visible"}
    if depth.ndim != 2 or depth.size == 0:
        return {"status": "depth_unavailable"}
    grid = np.linspace(.16, .84, 7)
    u, v = np.meshgrid(grid, grid)
    u, v = u.reshape(-1, 1), v.reshape(-1, 1)
    points_px = ((1 - v) * ((1 - u) * corners[0] + u * corners[1])
                 + v * ((1 - u) * corners[3] + u * corners[2]))
    px = np.rint(points_px).astype(int)
    width, height = depth.shape[1], depth.shape[0]
    inside = np.all((px >= 0) & (px < [width, height]), axis=1)
    px = px[inside]
    z = depth[px[:, 1], px[:, 0]]
    good = np.isfinite(z) & (z > 0)
    px, z = px[good], z[good]
    if len(z) < 25:
        return {"status": "too_few_valid_depth_samples", "depth_samples": len(z)}

    def rays(pixels):
        normalized = cv2.undistortPoints(pixels.astype(float), estimator.camera_matrix,
                                         estimator.distortion).reshape(-1, 2)
        return np.column_stack((normalized, np.ones(len(normalized))))

    points = rays(px) * z[:, None]
    origin = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - origin, full_matrices=False)
    normal = axes[-1]
    residual = (points - origin) @ normal
    error = float(np.sqrt(np.mean(residual ** 2)))
    quality = {"depth_samples": len(z), "plane_rms_m": error,
               "median_optical_depth_m": float(np.median(z))}
    corner_rays = rays(corners)
    divisor = corner_rays @ normal
    if error > .004 or np.any(np.abs(divisor) < .10):
        return {"status": "bad_marker_plane", **quality}
    lengths = (origin @ normal) / divisor
    if np.any(lengths <= 0):
        return {"status": "marker_behind_camera", **quality}
    corners_world = (corner_rays * lengths[:, None]) @ estimator.camera_to_world[:3, :3].T
    corners_world += estimator.camera_to_world[:3, 3]
    x = (corners_world[1] - corners_world[0] + corners_world[2] - corners_world[3]) / 2
    y = (corners_world[0] - corners_world[3] + corners_world[1] - corners_world[2]) / 2
    size = estimator.object_points[1, 0] - estimator.object_points[0, 0]
    if np.max(np.abs(np.linalg.norm([x, y], axis=1) / size - 1)) > .15:
        return {"status": "inconsistent_marker_size", **quality}
    x /= np.linalg.norm(x)
    z_axis = np.cross(x, y)
    z_axis /= np.linalg.norm(z_axis)
    rotation = np.column_stack((x, np.cross(z_axis, x), z_axis))
    center = corners_world.mean(axis=0) - estimator.marker_to_cube_offset_m * z_axis
    return {"status": "detected", "position_m": center.tolist(),
            "cube_rotation_world": rotation.tolist(), **quality}


class OpticalCapture:
    def __init__(self, path):
        self.stream = path.open("w", buffering=1)
        self.supply_frame = 0

    def record(self, source, frame_id, captured, received, pose):
        row = {"source": source, "frame_id": frame_id, "capture_time_s": captured,
               "received_time_s": received, **pose}
        self.stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    def record_supply(self, camera, estimator, height, received):
        frame = camera.get_current_frame()
        frame_id = int(frame["rendering_frame"])
        captured = float(frame["rendering_time"])
        rgba = np.asarray(frame["rgba"])
        if (frame_id <= self.supply_frame or not 0 <= received - captured <= .2
                or rgba.ndim != 3 or rgba.shape[2] != 4):
            return
        self.supply_frame = frame_id
        detection = estimator.estimate(rgba.astype(np.uint8, copy=False))
        pose = estimator.resting_pose(detection, height)
        row = {"status": "no_planar_pose"}
        if pose is not None:
            position, rotation = pose
            row = {"status": "detected", "position_m": position.tolist(),
                   "cube_rotation_world": rotation.tolist()}
        self.record("supply_rgb", frame_id, captured, received, row)

    def record_ranges(self, query, poses, step, stamp):
        from pxr import Gf

        gaps = {}
        for side, (position, rotation) in poses.items():
            inward = rotation[:, 1] * (-1 if side == "left" else 1)
            # The emitter sits between the four cups, 2 mm behind their calibrated plane.
            emitter = position - .002 * inward
            hit = query.raycast_closest(Gf.Vec3f(*emitter.tolist()), Gf.Vec3f(*inward.tolist()), .052)
            distance = hit["distance"] - .002 if hit["hit"] else None
            gaps[side] = max(0., round(distance / .0005) * .0005) if distance is not None else None
        self.record("short_range", step, stamp, stamp, {"status": "sampled", "gaps_m": gaps})


def inject(source, yaw_deg, short_range=False, light_touch=False):
    from source_hooks import replace_once

    source = replace_once(source, "        supply_camera.pause()\n", "        supply_camera.resume()\n")
    source = replace_once(source, "                if camera is supply_camera:\n                    camera.pause()\n",
                          "                pass  # The observation capture owns the active supply camera.\n")
    source = replace_once(source, "        def sample_vision(render=True):", f"""        from experiments.s0_optical_capture import OpticalCapture, depth_marker_pose
        s0_capture = OpticalCapture(output_directory / "s0_camera.jsonl")
        s0_support = next(center[2] + size[2] / 2 for name, center, size in CONFIG["static_boxes"]
                          if name == "supply")
        vision_camera.add_distance_to_image_plane_to_frame()
        report["s0_optical_capture"] = {{
            "depth_source": "distance_to_image_plane",
            "render_camera": vision_runtime_path,
            "aligned_rgb_and_depth": True,
            "supply_yaw_deg": {float(yaw_deg)!r},
            "light_touch": {light_touch!r},
        }}

        def sample_vision(render=True):""")
    needle = "                    observation = vision_estimator.estimate(rgba.astype(np.uint8, copy=False))"
    source = replace_once(source, needle, needle + """
                    if 0 <= receive_time - capture_time <= .2:
                        depth_pose = depth_marker_pose(vision_estimator, observation["corners_px"],
                                                       frame.get("distance_to_image_plane"))
                        s0_capture.record("d455_depth", frame_id, capture_time, receive_time, depth_pose)
                    s0_capture.record_supply(supply_camera, supply_estimator, s0_support, receive_time)""")
    # Capture only the approach and its terminal contact.
    start = source.index("            try_attach_pair()\n", source.index("        def run_cube():"))
    end = source.index("        placed_observations = []", start)
    source = source[:start] + "            try_attach_pair()\n            step(24)\n" + source[end:]
    needle = '                run_cube()\n            observed, _ = observe_resting_cube("upper_landing_pose"'
    status = "LIGHT_TOUCH_BEFORE_SEAL" if light_touch else "SEALED_BEFORE_LIFT"
    source = replace_once(source, needle, f'''                run_cube()
                record("s0_approach_capture", "{status}", cube_id=task_id)
                break
            observed, _ = observe_resting_cube("upper_landing_pose"''')
    if light_touch:
        begin = source.index("        def try_attach_pair():")
        loop = source.index("                stable = stable + 1 if all(.5 <= force", begin)
        commands = source.index("                for side, (position, rotation) in targets.items():", loop)
        source = source[:loop] + '''                stable = stable + 1 if all(.03 <= force <= .12 for force in forces.values()) else 0
                if stable >= 120:
                    record("s0_light_touch", "STABLE_LOW_COMPRESSION", compression_N=forces, held_ticks=stable)
                    return
''' + source[commands:]
        end = source.index("        def follow_side_guide", begin)
        approach = source[begin:end].replace("forces[side] > .1:", "forces[side] > .025:")
        approach = approach.replace("target_force_N=1., gain_m_per_Ns=2e-5,",
                                    "target_force_N=.075, gain_m_per_Ns=8e-5, max_speed_m_s=.0001,")
        source = source[:begin] + approach + source[end:]
    if yaw_deg:
        needle = '        lower_bodies = {}'
        half_yaw = float(yaw_deg) * np.pi / 360
        source = replace_once(source, needle, f'''        UsdGeom.Xformable(prim).AddOrientOp().Set(
            Gf.Quatf({float(np.cos(half_yaw))!r}, Gf.Vec3f(0., 0., {float(np.sin(half_yaw))!r})))
{needle}''')
    if short_range:
        source = replace_once(source, "        def sample_vision(render=True):", '''        from omni.physx import get_physx_scene_query_interface
        short_range_query = get_physx_scene_query_interface()
        report["s0_optical_capture"]["short_range"] = {
            "source": "single-beam simulated range", "range_from_lip_m": .05,
            "quantization_m": .0005, "timestamp": "encoder sampling step",
        }

        def sample_vision(render=True):''')
        anchor = "                reference_contacts = contact_monitor.cube_contact_snapshot()"
        source = replace_once(source, anchor, '''                range_poses = {side: (np.asarray(force_sample[side]["tcp_position_m"]), rotation)
                               for side, rotation in sensor_rotations.items()}
                s0_capture.record_ranges(short_range_query, range_poses, tick, float(world.current_time))
''' + anchor)
    return replace_once(source, '        if vision_comparison is not None:\n',
                        '        if "s0_capture" in locals():\n'
                        '            s0_capture.stream.close()\n'
                        '        if vision_comparison is not None:\n')
