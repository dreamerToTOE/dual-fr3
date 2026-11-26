"""Test an unmarked 150 mm Cube with the high-wall scene's rear camera."""

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from continuous_side_row import GUIDE_NAMES, HERE, REPO, patch_a, patch_b
from source_hooks import replace_once


class SquareTop:
    """Recover a supported Cube from the rear camera's optical-axis depth."""

    def __init__(self, intrinsics, to_world, resolution, config, support_z=None):
        self.intrinsics = intrinsics
        self.to_world = to_world
        self.size = config["cube_size_m"]
        supports = [(np.asarray(center), np.asarray(extent)) for name, center, extent in config["static_boxes"]
                    if name in ("supply", "floor")]
        support_heights = [center[2] + extent[2] / 2 for center, extent in supports]
        assert np.allclose(support_heights, support_heights[0])
        self.top_z = (support_heights[0] if support_z is None else support_z) + self.size

        width, height = resolution
        v, u = np.indices((height, width), dtype=np.float32)
        # Isaac's intrinsics place the principal point at width/2, height/2.
        rays = np.stack((u + .5, v + .5, np.ones_like(u)), axis=-1) @ (
            to_world[:3, :3] @ np.linalg.inv(intrinsics)).T
        origin = to_world[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ranges = (self.top_z - origin[2]) / rays[..., 2]
            xy = origin[:2] + rays[..., :2] * ranges[..., None]
        inside = np.zeros((height, width), dtype=bool)
        for center, extent in supports:
            inside |= (np.abs(xy - center[:2]) <= extent[:2] / 2).all(axis=-1)
        for name, center, extent in config["static_boxes"]:
            if name.startswith("wall_"):
                near_wall = np.abs(xy - center[:2]) <= np.array(extent[:2]) / 2 + .005
                inside &= ~near_wall.all(axis=-1)
        inside &= ranges > 0
        rows, columns = np.where(inside)
        self.roi = (slice(rows.min(), rows.max() + 1), slice(columns.min(), columns.max() + 1))
        self.xy = xy[self.roi].astype(np.float32)
        self.expected_depth = ranges[self.roi].astype(np.float32)
        self.ray_z = rays[self.roi][..., 2].astype(np.float32)
        self.inside = inside[self.roi]

    def fit_visible_edges(self, contour, top, clear, center, yaw):
        """Recover the square center from transitions to visible free space."""
        component = np.zeros(top.shape, dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 1, cv2.FILLED)
        component = component.astype(bool) & top
        points, directions = [], []
        for axis in (0, 1):
            preceding = [slice(None), slice(None)]
            following = preceding.copy()
            preceding[axis], following[axis] = slice(None, -1), slice(1, None)
            for source, neighbor in ((tuple(preceding), tuple(following)),
                                     (tuple(following), tuple(preceding))):
                boundary = component[source] & clear[neighbor]
                a, b = self.xy[source][boundary], self.xy[neighbor][boundary]
                points.append((a + b) / 2)
                directions.append(b - a)
        points, directions = np.concatenate(points), np.concatenate(directions)
        c, s = math.cos(yaw), math.sin(yaw)
        axes = np.array([[c, -s], [s, c]])
        local = (points - center) @ axes
        normals = (directions / np.linalg.norm(directions, axis=1, keepdims=True)) @ axes
        outline = (self.xy[component] - center) @ axes
        coordinates = []
        for axis in (0, 1):
            x = local[:, axis]
            y = local[:, 1 - axis]
            support = []
            for sign in (-1, 1):
                edge = sign * np.max(sign * outline[:, axis])
                along_edge = ((normals[:, axis] * sign > .55) & (abs(x - edge) < .013)
                              & (abs(y) < self.size * .36))
                if along_edge.sum() >= 6 and np.ptp(y[along_edge]) >= .033:
                    support.append(np.median(x[along_edge]) - sign * self.size / 2)
            if not support:
                return None
            coordinates.append(np.mean(support))
        return center + axes @ coordinates

    def estimate(self, frame, region_center=None):
        missing = {"status": "no_square_top", "cube_position_m": None}
        depth = np.asarray(frame.get("distance_to_image_plane", [])).squeeze()
        if depth.ndim != 2:
            return {**missing, "status": "no_depth_frame"}
        depth = depth[self.roi]
        with np.errstate(invalid="ignore"):
            height_error = (depth - self.expected_depth) * self.ray_z
        valid = self.inside & np.isfinite(depth) & (depth > 0)
        if region_center is not None:
            valid &= (np.abs(self.xy - region_center[:2]) < self.size * .82).all(axis=-1)
        top = valid & (abs(height_error) < .004)
        clear = valid & (height_error <= -.004)
        contours, _ = cv2.findContours(top.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if len(contour) < 12:
                break
            points = contour[:, 0]
            outline = self.xy[points[:, 1], points[:, 0]]
            center, sides, angle = cv2.minAreaRect(outline)
            if min(sides) < .5 * self.size or max(sides) > self.size + .017:
                continue
            if cv2.contourArea(outline) < .45 * self.size ** 2:
                continue
            # A square's physical orientation is defined modulo 90 degrees.
            yaw = math.radians((angle + 45) % 90 - 45)
            center = self.fit_visible_edges(contour, top, clear, center, yaw)
            if center is None:
                continue
            c, s = math.cos(yaw), math.sin(yaw)
            rotation = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
            pixels = points + [self.roi[1].start, self.roi[0].start]
            return {
                "status": "detected", "cube_position_m": [*center, self.top_z - self.size / 2],
                "cube_rotation_world": rotation, "pixel_center": pixels.mean(axis=0).tolist(),
                "reprojection_error_px": None, "observed_top_sides_m": list(sides),
                "outline_px": pixels.tolist(),
            }
        return missing

    def resting_pose(self, observation, support_z):
        center = observation["cube_position_m"]
        if center is not None and abs(center[2] - support_z - self.size / 2) < 1e-6:
            return np.asarray(center), np.asarray(observation["cube_rotation_world"])
        return None

    def save_view(self, frame, fit, path):
        image = cv2.cvtColor(np.asarray(frame["rgba"]), cv2.COLOR_RGBA2BGR)
        center = np.asarray(fit["cube_position_m"]) + [0, 0, self.size / 2]
        signs = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
        corners = center + (signs * self.size / 2) @ np.asarray(fit["cube_rotation_world"]).T
        optical = (corners - self.to_world[:3, 3]) @ self.to_world[:3, :3]
        projected = optical @ self.intrinsics.T
        polygon = np.rint(projected[:, :2] / projected[:, 2:] - .5).astype(np.int32)
        cv2.polylines(image, [polygon], True, (50, 255, 80), 2, cv2.LINE_AA)
        cv2.imwrite(str(path), image)


def capture_snapshots(camera, fitter, cube, step, period, output):
    """Move the same physical Cube between test poses, then fit only its measured images."""
    cases = [
        ("supply", (0., -.300), 0.),
        ("supply_plus", (.012, -.302), 3.),
        ("supply_minus", (-.013, -.305), -4.),
        ("landing", (0., .736), 0.),
        ("landing_offset", (.023, .748), 5.),
        ("seated", (-.3045, 1.0755), 0.),
    ]
    frames = []
    for label, (x, y), yaw_deg in cases:
        yaw = math.radians(yaw_deg)
        cube.set_world_pose(np.array([x, y, fitter.top_z - fitter.size / 2]),
                            np.array([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]))
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        step(4 * period)
        frame = camera.get_current_frame(clone=True)
        fit = fitter.estimate(frame)
        truth, quaternion = cube.get_world_pose()
        w, qx, qy, qz = quaternion
        truth_yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        row = {"case": label, "frame_id": int(frame.get("rendering_frame", 0)),
               "truth_center_m": truth.tolist(), "truth_yaw_deg": truth_yaw, **fit}
        if fit["status"] == "detected":
            rotation = fit["cube_rotation_world"]
            measured_yaw = math.degrees(math.atan2(rotation[1][0], rotation[0][0]))
            row.update(error_xy_mm=float(1000 * np.linalg.norm(np.asarray(fit["cube_position_m"][:2]) - truth[:2])),
                       yaw_error_deg=abs((measured_yaw - truth_yaw + 45) % 90 - 45))
            fitter.save_view(frame, fit, output / f"{label}.png")
        np.savez_compressed(output / f"{label}.npz", depth_m=frame["distance_to_image_plane"], rgba=frame["rgba"])
        frames.append(row)
        print("FR3_SQUARE_SNAPSHOT", label, fit["status"], row.get("error_xy_mm"), row.get("yaw_error_deg"), flush=True)
    (output / "square_snapshots.json").write_text(json.dumps(frames, indent=2, allow_nan=False) + "\n")
    return frames


def replace_block(source, start, end, replacement=""):
    first = source.index(start)
    last = source.index(end, first + len(start))
    return source[:first] + replacement + source[last:]


def patch_square(source):
    source = replace_once(source, '        camera["resolution"] = [1280, 960]',
                          '''        camera["resolution"] = [1280, 960]
        CONFIG["cube_sequence"] = CONFIG["cube_sequence"][:1]''')
    source = replace_once(source, 'W760 300mm wall continuous five-cube row; opposing vertical face suction only',
                          'W760 300mm wall single Cube; rear depth square and vertical face suction')
    source = source.replace('READY_FOR_RGB_LOCALIZATION', 'READY_FOR_DEPTH_LOCALIZATION')
    source = source.replace('fresh rear RGB', 'fresh rear depth square')
    source = replace_block(source, '        if args.five_cube_row:\n            from fr3_cube_supply import add_cube_marker',
                           '        for side, base in CONFIG["fr3"]["bases"].items():')
    source = replace_block(source, '        # The main rear view is blocked', '        if not args.headless:\n            import omni.ui')
    source = replace_once(source, '        supply_camera.pause()\n        side_rear_camera.pause()\n',
                          '        vision_camera.add_distance_to_image_plane_to_frame()\n')
    source = replace_block(source, '        def make_vision_estimator(', '        comparison_directory = ',
                           '''        vision_estimator = SquareTop(camera_matrix, camera_to_world,
                                         vision_cfg["resolution"], CONFIG)
        report["vision"]["observation"] = "markerless rear optical-axis depth and 150 mm square"
        report["control_input_boundary"]["target_sources"][0] = "rear depth square on calibrated support plane"
''')
    source = replace_once(source, 'from fr3_vision import ArucoCubeEstimator, PlanarPoseMedian',
                          'from fr3_vision import PlanarPoseMedian')
    source = source.replace('.estimate(rgba.astype(np.uint8, copy=False))', '.estimate(frame)')
    source = source.replace('supply_camera, supply_estimator, supply_top', 'vision_camera, vision_estimator, supply_top')
    source = replace_once(source, '            nonlocal vision_camera, vision_estimator, last_supported_capture, withdrawing_corner_tools',
                          '            nonlocal last_supported_capture, withdrawing_corner_tools')
    source = replace_block(source, '            if slot == 1:\n                # Use the centered RGB frame',
                           '            allowed_contacts.add(support_side)')
    source = replace_block(source, '            if slot == 1:\n                vision_camera.pause()',
                           '            max_speed = settle_cube("B2")')
    source = replace_block(source, '            if current_task["cube_id"] == 1 and phase in ("B2_rear_pose"',
                           '            return observe_resting_cube(phase, vision_camera,')
    source = replace_block(source, '            camera.resume()\n            diagnostics = []', '            if len(estimates) < 3:',
                           '''            diagnostics = []
            last_detection = None
            for _ in range(30):
                step(vision_period)
                render_only()
                frame = camera.get_current_frame(clone=True)
                frame_id = int(frame.get("rendering_frame", 0))
                stamp = float(frame.get("rendering_time", -1.))
                age = float(world.current_time) - stamp
                if frame_id <= 0 or frame_id in frames or stamp < started or not 0 <= age <= .20:
                    continue
                frames.add(frame_id)
                observation = estimator.estimate(frame)
                diagnostics.append({"id": frame_id, "age": age, "status": observation["status"]})
                pose = estimator.resting_pose(observation, support_z)
                if pose is not None:
                    estimates.append(pose)
                    exposure_ids.append((frame_id, stamp))
                    last_detection = frame, observation
                if len(estimates) >= 3:
                    break
            if last_detection is not None:
                estimator.save_view(*last_detection, output_directory / f"{phase}.png")
''')
    source = replace_block(source, '                "phase": phase, "source": ("supply_rgb"', '                "frame_count": len(estimates)',
                           '                "phase": phase, "source": "rear_depth_square",\n')
    source = replace_once(source, '                    "optical_view": "left_bay" if vision_camera is side_rear_camera else "rear_center",',
                          '                    "optical_view": "rear_center",')
    source = replace_once(source, '            supply_estimator.marker_id = marker_id\n', '')
    source = replace_once(source, '            "camera_source": "RGB + ArUco + solvePnP + fixed camera calibration only",',
                          '            "camera_source": "rear_depth_square",\n'
                          '            "camera_stream_source": "rear_depth_square",')
    source = source.replace('"RGB_support_plane"', '"rear_depth_square"')
    source = replace_once(source, 'no stable fresh ArUco observation', 'no stable fresh square-top observation')
    source = replace_once(source, '                report["failed_observation"] = diagnostics\n', '''                failed_frame = camera.get_current_frame(clone=True)
                np.savez_compressed(output_directory / f"{phase}_unobserved.npz",
                                    rgba=failed_frame.get("rgba", []),
                                    depth_m=failed_frame.get("distance_to_image_plane", []))
                report["failed_observation"] = diagnostics
''')
    source = replace_once(source, '        sequence = CONFIG.get("cube_sequence", [current_task])', '''        if SNAPSHOTS:
            frames = capture_snapshots(vision_camera, vision_estimator, cube, step, vision_period, output_directory)
            passed = all(row.get("error_xy_mm", float("inf")) < 10. for row in frames)
            report["status"] = "PASS_SINGLE_CUBE_SNAPSHOTS" if passed else "BLOCKED_SINGLE_CUBE_SNAPSHOTS"
            return 0 if passed else 1

        sequence = CONFIG.get("cube_sequence", [current_task])''')
    source = replace_block(source, '        if args.five_cube_row:\n            record("packing_verification_park"',
                           '        report["ab_status"] = "PASS_SENSOR_TARGETED_PHYSX"',
                           '        report["status"] = "PASS_A_B1_B2_SENSOR_CONTROL"\n')
    source = replace_once(source, '        import numpy as np\n', '        import numpy as np\n        np.random.seed(0)\n')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--snapshots", action="store_true", help="Fit six observed poses of the same Cube")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    production = REPO / "fr3_ab_visual_runner.py"
    source = patch_square(patch_b(patch_a(production.read_text())))
    output = args.output.resolve()
    runner = compile(source, str(output / "runner_snapshot.py"), "exec")
    if args.check_only:
        return
    output.mkdir(parents=True)
    (output / "runner_snapshot.py").write_text(source)
    sys.path.insert(0, str(REPO))
    sys.argv = [str(production), "--five-cube-row", "--output-dir", str(output), "--close-on-finish",
                "--collision-workers", "3"]
    if args.headless:
        sys.argv.append("--headless")
    guides = {1: json.loads((HERE / GUIDE_NAMES[0]).read_text())}
    exec(runner, {"__file__": str(production), "__name__": "__main__", "SIDE_GUIDES": guides,
                  "SIDE_GUIDE_NAMES": [GUIDE_NAMES[0]], "SNAPSHOTS": args.snapshots,
                  "SquareTop": SquareTop, "capture_snapshots": capture_snapshots})


if __name__ == "__main__":
    main()
