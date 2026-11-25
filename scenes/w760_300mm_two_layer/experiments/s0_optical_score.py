"""Compare S0 optical inputs on the same timestamped physical approach."""

import argparse
from collections import Counter
import json
from math import dist, hypot
from pathlib import Path

import numpy as np

from seven_observer import SevenObserver
from seven_reference import MIN_IMPULSE_NS, SevenReference
from thesis_metrics import confusion_metrics, load_clip
from u4_handover_score import events


def camera_error(rows, packets, hz):
    times = np.array([physical["time_s"] for _, physical in rows])
    errors = []
    for packet in packets:
        if packet["status"] != "detected":
            continue
        sample_time = packet["capture_time_s"]
        index = int(np.searchsorted(times, sample_time))
        candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
        step = min(candidates, key=lambda i: abs(times[i] - sample_time))
        if abs(times[step] - sample_time) <= .5 / hz + 1e-7:
            center = packet["position_m"]
            truth = rows[step][1]["cube_position_m"]
            errors.append((1000 * dist(center[:2], truth[:2]), 1000 * (center[2] - truth[2])))
    result = {"paired_frames": len(errors), "status_counts": dict(Counter(p["status"] for p in packets))}
    if errors:
        xy, z = np.asarray(errors).T
        result.update(median_xy_error_mm=float(np.median(xy)), median_signed_z_error_mm=float(np.median(z)),
                      median_abs_z_error_mm=float(np.median(np.abs(z))),
                      p95_abs_z_error_mm=float(np.percentile(np.abs(z), 95)))
    return result


def observation(packet):
    return {**packet, "frame_id": f"{packet['source']}:{packet['frame_id']}"}


def range_confirms_separation(observer, result, sensor, gaps):
    if (result["state"] != "UNKNOWN" or not any(gap is not None for gap in gaps.values())
            or result["camera_age_s"] is None or not 0 <= result["camera_age_s"] <= .40
            or result["position_m"] is None or observer._support_level(result["position_m"]) > .04):
        return False
    return all(
        hypot(*sensor[side + "_wrench_world"][:3]) < .30
        and (gap >= .001 if gap is not None else result["tools"][side] == "separated")
        for side, gap in gaps.items()
    )


def score(path, alternate, separation_margin_mm=5.):
    header, rows = load_clip(path)
    if header["case"] != "s0":
        raise ValueError("Expected a paired S0 approach capture")
    hz = header["hz"]
    config = header["config"]
    stamps = [(p["capture_time_s"], p["received_time_s"]) for p in header["camera_packets"]]
    if any(captured > received for captured, received in stamps):
        raise ValueError("Camera packet was received before its exposure")
    depth = {p["frame_id"]: p for p in header["camera_packets"] if p["source"] == "d455_depth"}
    supply = [p for p in header["camera_packets"] if p["source"] == "supply_rgb"]
    ranges = {p["frame_id"]: p for p in header["camera_packets"] if p["source"] == "short_range"}
    if alternate == "short_range" and len(ranges) != len(rows):
        raise ValueError("Short-range replay requires a same-step range signal throughout the capture")
    alternate_packets = sorted((p for p in supply if p["status"] == "detected"),
                               key=lambda p: p["received_time_s"])
    ref = SevenReference(config, hz, header["reference_positions"])
    observers = {"rear_rgb": SevenObserver(config, "fusion_compensated", revised=True),
                 alternate: SevenObserver(config, "fusion_compensated", revised=True,
                                          separation_margin_m=separation_margin_mm / 1000)}
    traces = {name: [] for name in observers}
    support_top = next(p[2] + d[2] / 2 for label, p, d in config["static_boxes"] if label == "supply")
    raw, latest_supply, index = {}, None, 0
    used_frames = set()
    used_steps = range_steps = range_confirmations = 0
    for sensor, physical in rows:
        step, now = sensor["physics_step"], sensor["time_s"]
        if (step, now) != (physical["physics_step"], physical["time_s"]):
            raise ValueError(f"Unpaired sample {step}")
        raw_camera = sensor.get("camera")
        if raw_camera:
            raw[raw_camera["frame_id"]] = raw_camera
        candidate = raw_camera
        if alternate == "d455_depth" and raw_camera:
            matched = depth.get(raw_camera["frame_id"])
            if (matched and matched["status"] == "detected"
                    and matched["capture_time_s"] == raw_camera["capture_time_s"]
                    and matched["received_time_s"] <= now + 1e-7):
                candidate = observation(matched)
        if alternate in ("supply_rgb", "short_range"):
            while index < len(alternate_packets) and alternate_packets[index]["received_time_s"] <= now + 1e-7:
                latest_supply = alternate_packets[index]
                index += 1
            if latest_supply is not None and 0 <= now - latest_supply["capture_time_s"] <= .40:
                candidate = observation(latest_supply)
        if candidate is not raw_camera:
            used_steps += 1
            used_frames.add(candidate["frame_id"])
        truth = ref.update(physical)["state"]
        valid = physical["cube_contacts"]["valid"]
        cups = [p for p in physical["cube_contacts"]["pairs"]
                if p["label"] in ("left_cup", "right_cup") and p["current_step_observed"]] if valid else []
        touch = [p for p in cups if ((p["normal_impulse_Ns"] or 0.) > MIN_IMPULSE_NS
                                     or (p["separation_m"] is not None and p["separation_m"] <= 0.))]
        min_positive = min((p["separation_m"] for p in cups
                            if p["separation_m"] is not None and p["separation_m"] > 0.),
                           default=float("inf"))
        base = {"step": step, "truth": truth, "near_s0": truth == "S0" and min_positive <= .005,
                "touch": bool(touch), "weak_touch": bool(touch) and
                sum(p["normal_impulse_Ns"] or 0. for p in touch) * hz < .3}
        for name, obs in observers.items():
            packet = raw_camera if name == "rear_rgb" else candidate
            sensors = {**sensor, "camera": packet}
            result = obs.update(sensors, neighbors=header["optical_neighbors"])
            if name == "short_range":
                measurement = ranges[step]
                if measurement["received_time_s"] != now:
                    raise ValueError(f"Unpaired range sample {step}")
                gaps = measurement["gaps_m"]
                range_steps += any(value is not None for value in gaps.values())
                if range_confirms_separation(obs, result, sensors, gaps):
                    result = {**result, "state": "S0"}
                    range_confirmations += 1
            traces[name].append({**base, "pred": result["state"],
                                 "tactile_hit": truth == result["state"] == "S1"})

    def subset(trace, key):
        samples = [row for row in trace if row[key]]
        return {"steps": len(samples), "recognized_as_s0": sum(r["pred"] == "S0" for r in samples)}

    summary = {}
    for name, trace in traces.items():
        native_s1 = events(trace, "truth", "S1")["first_3step_confirm"]
        first_s1 = events(trace, "tactile_hit")["first_3step_confirm"]
        by_state = confusion_metrics(trace)
        summary[name] = {"by_state": {s: by_state[s] for s in ("S0", "S1", "S2")},
                         "s0_near_cup": subset(trace, "near_s0"),
                         "native_touch": subset(trace, "touch"),
                         "native_weak_touch": subset(trace, "weak_touch"),
                         "first_s1_physx_step": native_s1, "first_s1_confirmed_step": first_s1,
                         "s1_confirmation_delay_s": ((first_s1 - native_s1) / hz
                                                     if first_s1 is not None else None)}
    camera_source = "d455_depth" if alternate == "d455_depth" else "supply_rgb"
    comparisons = {"rear_rgb": camera_error(rows, raw.values(), hz), camera_source: camera_error(
        rows, depth.values() if camera_source == "d455_depth" else supply, hz)}
    comparisons["supply_plane_height_m"] = support_top
    return {"source": header["source"], "steps": len(rows), "capture": header["optical_capture"],
            "incomplete_run": header.get("incomplete_run"),
            "weak_touch_longest_ticks": events(traces["rear_rgb"], "weak_touch")["longest_ticks"],
            "alternate_camera_steps": used_steps, "alternate_camera_frames": len(used_frames),
            "alternate_separation_margin_mm": separation_margin_mm,
            "short_range_steps_with_return": range_steps, "short_range_added_confirmations": range_confirmations,
            "paired_pose_error": comparisons, "scores": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired", type=Path)
    parser.add_argument("--observations", choices=("d455_depth", "supply_rgb", "short_range"), required=True)
    parser.add_argument("--separation-margin-mm", type=float, default=5.)
    args = parser.parse_args()
    print(json.dumps(score(args.paired, args.observations, args.separation_margin_mm), indent=2, allow_nan=False))
