"""Replay upper-row slot 5 contact estimation from causal sensor logs."""

import argparse
from collections import Counter, deque
import gzip
import json
import math
from pathlib import Path

import numpy as np


MODES = ("vision_fk", "wrench_fk", "fusion")
STATES = ("NORMAL_ADVANCE", "NEIGHBOR_CONTACT_MOVING", "BLOCKED_EARLY", "AT_BACK_TARGET")
CAMERA_MAX_AGE_S = .25
CAMERA_HOLD_S = 1.
MOTION_WINDOW_S = .25
CAMERA_MOVE_MIN_M = .00045
TCP_MOVE_MIN_M = .00002
FRONT_LOAD_N = .30
BLOCKING_LOAD_N = 3.
CAMERA_REAR_BAND_M = .002
TCP_REAR_BAND_M = .0005
NEIGHBOR_BAND_M = .001
AXIAL_VISION_WEIGHT = .15


def records(path):
    open_stream = gzip.open if path.suffix == ".gz" else open
    with open_stream(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def yaw_of(rotation):
    return math.atan2(rotation[1][0], rotation[0][0])


def axes_for(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def neighbor_clearance(xy, yaw, neighbors, half):
    own = axes_for(yaw)
    signed = []
    for position, axes in neighbors.values():
        candidates = np.column_stack((own, axes))
        spans = np.abs(candidates.T @ own).sum(axis=1) + np.abs(candidates.T @ axes).sum(axis=1)
        signed.append(float(np.max(np.abs((position - xy) @ candidates) - half * spans)))
    return min(signed, default=math.inf)


class Estimator:
    def __init__(self, mode, half, rear_plane, delay_s):
        self.mode = mode
        self.half, self.rear_plane = half, rear_plane
        self.delay_s = delay_s
        self.camera_queue = deque()
        self.last_frame = None
        self.camera = None
        self.loaded_history = deque()
        self.motion_history = deque()

    def observe_camera(self, camera, now):
        if camera and camera["frame_id"] != self.last_frame:
            self.last_frame = camera["frame_id"]
            if camera.get("supported_position_m") is not None:
                capture = camera["supported_capture_time_s"]
                received = camera["received_time_s"]
                if capture <= received <= now + 1e-8:
                    xy = np.asarray(camera["supported_position_m"][:2])
                    yaw = yaw_of(camera["supported_rotation_world"])
                    self.camera_queue.append((received + self.delay_s, capture, xy, yaw))
        while self.camera_queue and self.camera_queue[0][0] <= now + 1e-8:
            self.camera = self.camera_queue.popleft()
        return self.camera

    def update(self, sample, neighbors):
        now = sample["time_s"]
        xy, yaw, y = None, None, None
        force_enabled = self.mode != "vision_fk"
        normal = np.asarray(sample["cup_normals_world"]["left"])
        tcp = np.asarray(sample["tcp_positions_m"]["left"])
        front_facing = normal[1] > .95
        compression, lateral = 0., 0.
        if force_enabled and front_facing:
            force = np.asarray(sample["left_wrench_world"][:3])
            compression = float(-force @ normal)
            lateral = abs(float(force[0]))
        loaded = compression >= FRONT_LOAD_N
        if loaded:
            self.loaded_history.append((now, float(tcp[1])))
            while self.loaded_history[0][0] < now - CAMERA_HOLD_S - .05:
                self.loaded_history.popleft()
            y = float(tcp[1] + self.half * normal[1])
        else:
            self.loaded_history.clear()
        camera_fresh = False
        if self.mode != "wrench_fk":
            camera = self.observe_camera(sample["camera"], now)
            if camera is not None:
                _, exposure, camera_xy, camera_yaw = camera
                age = now - exposure
                camera_fresh = 0 <= age <= CAMERA_MAX_AGE_S
                if camera_fresh or loaded and age <= CAMERA_HOLD_S:
                    xy, yaw = camera_xy.copy(), camera_yaw
                    if not loaded:
                        y = float(xy[1])
                    elif camera_fresh and self.loaded_history[0][0] <= exposure:
                        times, tcp_y = zip(*self.loaded_history)
                        extrapolated = xy[1] + tcp[1] - np.interp(exposure, times, tcp_y)
                        # Loaded FK is the direct axial reading; vision corrects drift and provides X/yaw.
                        y = float(AXIAL_VISION_WEIGHT * extrapolated + (1. - AXIAL_VISION_WEIGHT) * y)
                    xy[1] = y
        known_front = loaded
        if not force_enabled and xy is not None:
            front_y = y - self.half * (abs(math.sin(yaw)) + abs(math.cos(yaw)))
            known_front = front_facing and abs(tcp[1] - front_y) <= .012 and abs(tcp[0] - xy[0]) <= self.half
        if y is None or not known_front:
            self.motion_history.clear()
            return {"state": "UNKNOWN", "x_m": None if xy is None else float(xy[0]),
                    "y_m": y, "yaw_deg": None if yaw is None else math.degrees(yaw),
                    "camera_fresh": camera_fresh}
        motion_y = float(tcp[1] + self.half * normal[1]) if loaded else y
        self.motion_history.append((now, motion_y, compression))
        while len(self.motion_history) > 2 and self.motion_history[1][0] <= now - MOTION_WINDOW_S:
            self.motion_history.popleft()
        past, past_y, _ = self.motion_history[0]
        enough_history = now - past >= MOTION_WINDOW_S - .01
        moving = motion_y - past_y >= (TCP_MOVE_MIN_M if loaded else CAMERA_MOVE_MIN_M)
        rear_span = self.half if loaded else self.half * (abs(math.sin(yaw)) + abs(math.cos(yaw)))
        rear_gap = self.rear_plane - motion_y - rear_span
        near_neighbor = xy is not None and neighbor_clearance(xy, yaw, neighbors, self.half) <= NEIGHBOR_BAND_M
        side_load = lateral >= max(.45, .17 * compression)
        if self.mode == "vision_fk":
            side_contact = near_neighbor
        elif self.mode == "wrench_fk" or xy is None or not neighbors:
            side_contact = side_load
        else:
            side_contact = near_neighbor and side_load
        sustained_load = min(force for _, _, force in self.motion_history)
        state = "UNKNOWN"
        if enough_history:
            rear_band = TCP_REAR_BAND_M if loaded else CAMERA_REAR_BAND_M
            if moving:
                state = "NEIGHBOR_CONTACT_MOVING" if side_contact else "NORMAL_ADVANCE"
            elif abs(rear_gap) <= rear_band and (not force_enabled or compression >= .6):
                state = "AT_BACK_TARGET"
            elif rear_gap > rear_band and side_contact and (not force_enabled or sustained_load >= BLOCKING_LOAD_N):
                state = "BLOCKED_EARLY"
        return {"state": state, "x_m": None if xy is None else float(xy[0]),
                "y_m": y, "yaw_deg": None if yaw is None else math.degrees(yaw),
                "camera_fresh": camera_fresh}


def camera_neighbors(report):
    attempts = [row for row in report.get("neighbor_observation_attempts", []) if row["cube_id"] == 11
                and {"Cube_9", "Cube_10"} <= set(row["required_names"])]
    layouts = []
    for report_row in report.get("neighbor_observations", []):
        if report_row["cube_id"] != 11:
            continue
        for neighbor in report_row["neighbors"]:
            if neighbor["name"] not in {"Cube_9", "Cube_10"}:
                continue
            time = neighbor["capture_time_s"]
            eligible = [attempt["ended_s"] for attempt in attempts
                        if attempt["started_s"] <= time <= attempt["ended_s"]]
            if eligible:
                position = np.asarray(neighbor["position_m"][:2])
                axes = axes_for(yaw_of(neighbor["rotation_world"]))
                layouts.append((min(eligible), neighbor["name"], (position, axes)))
    return sorted(layouts, key=lambda row: row[0])


def evaluate(labels, estimated):
    pairs = [(row, predicted) for row, predicted in zip(labels, estimated) if row["state"] != "UNKNOWN"]
    confusion = {}
    for reference, output in pairs:
        confusion.setdefault(reference["state"], Counter())[output["state"]] += 1
    scores = {}
    for state, counts in confusion.items():
        reference_count = sum(counts.values())
        prediction_count = sum(other.get(state, 0) for other in confusion.values())
        correct = counts[state]
        scores[state] = {"samples": reference_count, "recall": correct / reference_count,
                         "f1": 2 * correct / (reference_count + prediction_count)}
    counts = len(pairs)
    pose = {}
    for axis, ref_value, unit in (
        ("x_m", lambda row: row["cube_position_m"][0], 1000.),
        ("y_m", lambda row: row["cube_position_m"][1], 1000.),
        ("yaw_deg", lambda row: row["yaw_deg"], 1.),
    ):
        for view in ("all", "camera_stale"):
            errors = [output[axis] - ref_value(row) for row, output in zip(labels, estimated)
                      if output[axis] is not None and (view == "all" or not output["camera_fresh"])]
            if axis == "yaw_deg":
                errors = [(error + 180.) % 360. - 180. for error in errors]
            pose.setdefault(view, {})[axis] = {
                "count": len(errors), "rmse_mm_or_deg": float(np.sqrt(np.mean(np.square(errors)))) * unit
                if errors else None,
            }
    events = []
    start = 0
    while start < len(labels):
        state = labels[start]["state"]
        end = start + 1
        while (end < len(labels) and labels[end]["state"] == state
               and labels[end]["physics_step"] == labels[end - 1]["physics_step"] + 1):
            end += 1
        if state != "UNKNOWN" and labels[end - 1]["time_s"] - labels[start]["time_s"] >= .24:
            detection = next((labels[i]["time_s"] for i in range(start, end) if estimated[i]["state"] == state), None)
            events.append({"state": state, "time_s": labels[start]["time_s"],
                           "duration_s": labels[end - 1]["time_s"] - labels[start]["time_s"],
                           "delay_s": None if detection is None else detection - labels[start]["time_s"]})
        start = end
    return {"reference_samples": counts, "reference_unknown_samples": len(labels) - counts,
            "accuracy": sum(row["state"] == output["state"] for row, output in pairs) / counts if counts else None,
            "coverage": sum(output["state"] != "UNKNOWN" for _, output in pairs) / counts if counts else None,
            "macro_f1_supported_classes": float(np.mean([entry["f1"] for entry in scores.values()])) if scores else None,
            "absent_reference_classes": [state for state in STATES if state not in confusion],
            "by_state": scores, "confusion": confusion, "pose": pose, "events": events}


def replay(directory, labels, delay_s, blurred_camera=False):
    report = json.loads((directory / "runner_report.json").read_text())
    cfg = report["scene_parameters"]
    half = cfg["cube_size_m"] / 2.
    _, center, size = next(row for row in cfg["static_boxes"] if row[0] == "wall_back")
    models = {mode: Estimator(mode, half, center[1] - size[1] / 2., delay_s) for mode in MODES}
    replay_labels = {row["physics_step"]: row for row in labels}
    samples = records(directory / "contact_states" / "sensor_inputs.jsonl.gz")
    blurred = {row["frame_id"]: row for row in records(directory / "vision" / "blurred_camera.jsonl")} if blurred_camera else None
    observations = deque(camera_neighbors(report))
    neighbors, ordered_labels = {}, []
    predictions = {mode: [] for mode in MODES}
    for row in samples:
        while observations and observations[0][0] + delay_s <= row["time_s"]:
            _, name, geometry = observations.popleft()
            neighbors[name] = geometry
        if blurred is not None:
            native_camera = row.get("camera")
            row = {**row, "camera": blurred.get(native_camera["frame_id"]) if native_camera else None}
        outputs = {mode: model.update(row, {} if mode == "wrench_fk" else neighbors)
                   for mode, model in models.items()}
        label = replay_labels.get(row["physics_step"])
        if label is not None:
            ordered_labels.append(label)
            for mode, output in outputs.items():
                predictions[mode].append(output)
    if len(ordered_labels) != len(labels):
        raise ValueError(f"{directory}: sensor and physical label clocks differ")
    return {mode: evaluate(ordered_labels, data) for mode, data in predictions.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", nargs=2, required=True, metavar=("RUN_DIR", "LABELS"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delay-ms", type=float, action="append", help="Extra RGB delivery delay, repeatable")
    parser.add_argument("--blurred-camera", action="store_true", help="Use the recorded blurred ArUco stream")
    args = parser.parse_args()
    delays = args.delay_ms or [0.]
    if not all(math.isfinite(delay) and delay >= 0 for delay in delays):
        parser.error("delays must be finite and non-negative")
    results = {}
    for run, label_path in args.case:
        labels = records(Path(label_path))
        results[run] = {str(delay): replay(Path(run), labels, delay / 1000., args.blurred_camera) for delay in delays}
    args.out.write_text(json.dumps({"method": "fixed causal rules applied independently per run",
                                   "camera_model": "blurred support-plane ArUco with optional receive delay"
                                   if args.blurred_camera else "native support-plane ArUco with optional receive delay",
                                   "threshold_provenance": "Motion and rear proximity were calibrated on the archived normal stroke. Confirmed anomaly classes have not been calibrated.",
                                   "thresholds": {
                                       "camera_max_age_s": CAMERA_MAX_AGE_S, "loaded_camera_hold_s": CAMERA_HOLD_S,
                                       "motion_window_s": MOTION_WINDOW_S, "camera_move_min_m": CAMERA_MOVE_MIN_M,
                                       "tcp_move_min_m": TCP_MOVE_MIN_M, "front_load_N": FRONT_LOAD_N,
                                       "blocking_load_N": BLOCKING_LOAD_N, "neighbor_band_m": NEIGHBOR_BAND_M,
                                       "camera_rear_band_m": CAMERA_REAR_BAND_M, "tcp_rear_band_m": TCP_REAR_BAND_M,
                                       "axial_vision_weight": AXIAL_VISION_WEIGHT},
                                   "pose_error_unit": {"x_m": "mm", "y_m": "mm", "yaw_deg": "deg"},
                                   "delay_ms_by_run": results}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
