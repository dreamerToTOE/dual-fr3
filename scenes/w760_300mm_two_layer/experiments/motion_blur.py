"""Collect a causal exposure stress test alongside the native D455 observations."""

from collections import deque
import json
import math

import cv2
import numpy as np


class MotionBlurObserver:
    def __init__(self, shutter_ms, frequency_hz):
        from fr3_vision import PlanarPoseMedian

        self.duration = shutter_ms / 1000
        self.max_gap = 1.8 / frequency_hz
        self.frames = deque()
        self.pose_filter = PlanarPoseMedian()
        self.rows = []

    def blend(self, rgba, end):
        if self.frames and end <= self.frames[-1][0]:
            return None
        if self.frames and end - self.frames[-1][0] > self.max_gap:
            self.frames.clear()
            self.pose_filter.reset()
        self.frames.append((end, cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)))
        start = end - self.duration
        while len(self.frames) > 1 and self.frames[1][0] <= start + 1e-9:
            self.frames.popleft()
        if len(self.frames) < 2 or self.frames[0][0] > start + 1e-9:
            return None
        weights = [0.] * len(self.frames)
        for idx, ((left, _), (right, _)) in enumerate(zip(self.frames, list(self.frames)[1:])):
            # Integrate a linear interpolation of the two sampled images over this segment.
            begin = max(start, left)
            fraction = ((begin + right) / 2 - left) / (right - left)
            width = (right - begin) / self.duration
            weights[idx] += width * (1 - fraction)
            weights[idx + 1] += width * fraction
        gray = np.zeros(self.frames[0][1].shape, dtype=np.float32)
        for (_, image), weight in zip(self.frames, weights):
            cv2.addWeighted(gray, 1., image, weight, 0., dst=gray, dtype=cv2.CV_32F)
        return cv2.cvtColor(cv2.convertScaleAbs(gray), cv2.COLOR_GRAY2RGB)

    def record(self, rgba, frame_id, exposure_end, receive_time, sharp, estimator, support_z, use_planar):
        blurred = self.blend(rgba, exposure_end)
        center_time = exposure_end - self.duration / 2
        row = {"frame_id": frame_id, "capture_time_s": center_time, "received_time_s": receive_time,
               "exposure_end_s": exposure_end, "native_status": sharp["status"], "status": "insufficient_frames"}
        if blurred is not None:
            result = estimator.estimate(blurred)
            row.update(status=result["status"], position_m=result["cube_position_m"],
                       cube_rotation_world=result["cube_rotation_world"],
                       reprojection_error_px=result["reprojection_error_px"])
            raw_rotation = np.asarray(result["cube_rotation_world"], dtype=float)
            if use_planar and raw_rotation.shape == (3, 3) and raw_rotation[2, 2] >= math.cos(math.radians(35)):
                pose = estimator.resting_pose(result, support_z)
                if pose is not None:
                    filtered = self.pose_filter.update(frame_id, center_time, *pose)
                    if filtered is not None:
                        position, rotation, stamp = filtered
                        row.update(supported_position_m=position.tolist(), supported_rotation_world=rotation.tolist(),
                                   supported_capture_time_s=stamp)
        self.rows.append(row)

    def write(self, path):
        with path.open("w") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        paired = [row for row in self.rows if row["status"] != "insufficient_frames"]
        return {"shutter_ms": self.duration * 1000, "method": "time-weighted causal integration of sampled intensity",
                "paired_frames": len(paired),
                "sharp_detections": sum(row["native_status"] == "detected" for row in paired),
                "blurred_detections": sum(row["status"] == "detected" for row in paired),
                "blurred_planar_updates": sum("supported_position_m" in row for row in paired)}


def inject(source, shutter_ms):
    from source_hooks import replace_once

    source = replace_once(source, "        def sample_vision(render=True):",
                          f"""        from experiments.motion_blur import MotionBlurObserver
        blur_observer = MotionBlurObserver({shutter_ms!r}, vision_cfg["frequency_hz"])

        def sample_vision(render=True):""")
    needle = "                    observation = vision_estimator.estimate(rgba.astype(np.uint8, copy=False))"
    source = replace_once(source, needle, needle + """
                    if 0 <= receive_time - capture_time <= .20:
                        blur_observer.record(rgba.astype(np.uint8, copy=False), frame_id, capture_time,
                                             receive_time, observation, vision_estimator,
                                             CONFIG["floor_top_m"] + CONFIG["cube_size_m"], bool(push_faces))
""".rstrip())
    return replace_once(source, '        if vision_comparison is not None:\n',
                        '        if "blur_observer" in locals():\n'
                        '            report["vision"]["motion_blur_observer"] = blur_observer.write(\n'
                        '                output_directory / "vision" / "blurred_camera.jsonl")\n'
                        '        if vision_comparison is not None:\n')
