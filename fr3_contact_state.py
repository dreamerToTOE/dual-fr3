"""Causal, explainable contact baseline using wrench, encoder FK and RGB/ArUco.

Task labels, simulator Cube poses and PhysX contact reports are not inputs.
Static scene geometry is a prior, and absence of wrist load is not proof of
absence of geometric contact. The returned skill IDs are recommendations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


STATES = (
    "NO_TOOL_CONTACT", "BILATERAL_SUPPORTED", "BILATERAL_SUSPENDED",
    "LEFT_FLOOR_CONTACT", "LEFT_REAR_CONTACT", "DUAL_REAR_CONTACT",
    "DUAL_CORNER_CONTACT", "RELEASED_CORNER", "UNKNOWN", "INCONSISTENT",
)


def _vector(value, length):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Expected {length} finite sensor coordinates")
    return result


class ContactStateEstimator:
    """Three-sample confirmation with immediate loss-of-observation/overload.

    `force_only` retains encoder geometry but ignores the camera entirely.
    `vision_only` retains encoders but does not inspect the wrench arguments.
    """

    def __init__(self, config, mode="fusion"):
        if mode not in ("fusion", "vision_only", "force_only"):
            raise ValueError("Unknown contact estimator mode")
        self.mode = mode
        self.half = float(config["cube_size_m"]) / 2
        boxes = {name: (np.asarray(center), np.asarray(size))
                 for name, center, size in config["static_boxes"]}
        back, back_size = boxes["wall_back"]
        left, left_size = boxes["wall_left"]
        self.rear_y = float(back[1] - back_size[1] / 2 - self.half)
        self.side_x = float(left[0] + left_size[0] / 2 + self.half)
        self.floor_z = float(config["floor_top_m"]) + self.half
        if not np.isfinite([self.half, self.rear_y, self.side_x, self.floor_z]).all() or self.half <= 0:
            raise ValueError("Invalid static contact geometry")
        library = json.loads(Path(__file__).with_name("fr3_contact_skills.json").read_text())
        self.skills = library["skills"]
        self.thresholds = library["observation_contract"]
        self.state = self.pending = "UNKNOWN"
        self.confirmations = 0
        self.last_time = None

    def update(self, *, time_s, left_wrench_world, right_wrench_world,
               tcp_positions_m, cup_normals_world, camera=None):
        """Consume only currently available observations, never future frames."""
        if not np.isfinite(time_s) or self.last_time is not None and time_s <= self.last_time:
            raise ValueError("Contact observations require strictly increasing finite time")
        self.last_time = float(time_s)
        sides = ("left", "right")
        positions = {side: _vector(tcp_positions_m[side], 3) for side in sides}
        normals = {side: _vector(cup_normals_world[side], 3) for side in sides}
        if any(abs(np.linalg.norm(normal) - 1) > 1e-3 for normal in normals.values()):
            raise ValueError("Cup normals must be unit world vectors from encoder FK")
        evidence, features = [], {}
        thresholds = self.thresholds
        visual = None
        if self.mode != "force_only" and camera is not None:
            try:
                capture, receive = float(camera["capture_time_s"]), float(camera["received_time_s"])
                if (camera["status"] == "detected" and np.isfinite([capture, receive]).all()
                        and capture <= receive + 1e-8 and receive <= time_s + 1e-8
                        and 0 <= time_s - capture <= thresholds["camera_max_age_s"]):
                    visual = _vector(camera["position_m"], 3)
                    features["camera_age_s"] = float(time_s - capture)
                    features["camera_frame_id"] = camera.get("frame_id")
                else:
                    evidence.append("camera_missing_stale_or_not_yet_received")
            except (KeyError, TypeError, ValueError):
                evidence.append("invalid_camera_observation")
        features["camera_fresh"] = visual is not None
        if visual is not None:
            features["camera_position_m"] = visual.tolist()
        forces, load, compression = {}, {}, {}
        if self.mode != "vision_only":
            for side, value in zip(sides, (left_wrench_world, right_wrench_world)):
                wrench = _vector(value, 6)
                forces[side] = wrench[:3]
                load[side] = float(np.linalg.norm(wrench[:3]))
                compression[side] = float(-wrench[:3] @ normals[side])
                features[side + "_force_norm_N"] = load[side]
                features[side + "_tcp_torque_norm_Nm"] = float(np.linalg.norm(wrench[3:]))
                features[side + "_compression_N"] = compression[side]
            if any(load[side] > thresholds["baseline_individual_force_max_N"]
                   or features[side + "_tcp_torque_norm_Nm"] > thresholds["baseline_TCP_torque_max_Nm"]
                   for side in sides):
                return self._result("INCONSISTENT", ["raw_wrench_exceeds_baseline_envelope"], features, False)
        raw = "UNKNOWN"
        centers = {}
        for side in sides:
            if compression.get(side, 0) > thresholds["contact_normal_threshold_N"]:
                centers[side] = positions[side] + self.half * normals[side]
        # Only a force-loaded contact licenses inferring Cube center from TCP.
        conflict = (len(centers) == 2 and np.linalg.norm(centers["left"] - centers["right"])
                    > thresholds["loaded_centers_consistency_band_m"])
        if visual is not None and centers:
            conflict |= any(np.linalg.norm(visual - center) > thresholds["visual_consistency_band_m"]
                            for center in centers.values())
        if conflict:
            return self._result("INCONSISTENT", ["camera_or_loaded_TCP_centers_disagree"], features, False)
        if forces:
            total_z = float(forces["left"][2] + forces["right"][2])
            features["combined_vertical_force_N"] = total_z
            if (total_z < thresholds["suspended_combined_vertical_force_max_N"]
                    and all(forces[side][2] < thresholds["suspended_each_vertical_force_max_N"]
                            for side in sides)):
                raw = "BILATERAL_SUSPENDED"
                evidence.append("both_wrists_support_vertical_payload")
            elif (len(centers) == 2 and normals["left"][0] > thresholds["cup_normal_axis_min_cosine"]
                  and normals["right"][0] < -thresholds["cup_normal_axis_min_cosine"]
                  and abs(np.mean([center[2] for center in centers.values()]) - self.floor_z)
                  < thresholds["support_height_band_m"]
                  and abs(total_z) < thresholds["supported_combined_vertical_force_abs_max_N"]):
                raw = "BILATERAL_SUPPORTED"
                evidence.append("opposed_side_loads_at_known_support_height")
            elif "left" in centers and normals["left"][1] > thresholds["cup_normal_axis_min_cosine"]:
                center = centers["left"]
                features["contact_center_from_FK_m"] = center.tolist()
                features["rear_gap_from_FK_m"] = float(self.rear_y - center[1])
                if abs(center[2] - self.floor_z) < thresholds["support_height_band_m"]:
                    at_rear = abs(center[1] - self.rear_y) <= thresholds["rear_side_geometry_band_m"]
                    right_contact = ("right" in centers and normals["right"][0]
                                     < -thresholds["cup_normal_axis_min_cosine"])
                    at_side = (right_contact and abs(centers["right"][0] - self.side_x)
                               <= thresholds["rear_side_geometry_band_m"])
                    raw = ("DUAL_CORNER_CONTACT" if at_rear and at_side else
                           "DUAL_REAR_CONTACT" if at_rear and right_contact else
                           "LEFT_REAR_CONTACT" if at_rear else "LEFT_FLOOR_CONTACT")
                    evidence.append("loaded_front_TCP_with_static_scene_geometry")
                    if right_contact:
                        evidence.append("right_side_normal_load")
        # Vision can establish separation; mere visual proximity never proves load.
        unloaded = not forces or all(load[side] < thresholds["unloaded_force_max_N"] for side in sides)
        if raw == "UNKNOWN" and visual is not None and unloaded:
            distances = {side: float(np.linalg.norm(np.maximum(
                np.abs(position - visual) - self.half, 0))) for side, position in positions.items()}
            features["camera_TCP_box_separation_m"] = distances
            if min(distances.values()) > thresholds["visual_TCP_separation_min_m"]:
                in_corner = (abs(visual[0] - self.side_x) < thresholds["visual_corner_zone_m"]
                             and abs(visual[1] - self.rear_y) < thresholds["visual_corner_zone_m"]
                             and abs(visual[2] - self.floor_z) < thresholds["visual_corner_zone_m"])
                raw = "RELEASED_CORNER" if in_corner else "NO_TOOL_CONTACT"
                evidence.append("both_TCPs_separated_from_visual_Cube_box")
                if not forces:
                    evidence.append("geometric_only_no_load_confirmation")
            else:
                evidence.append("low_force_does_not_exclude_geometric_contact")
        if raw == "UNKNOWN":
            evidence.append("insufficient_contact_evidence")
        return self._result(raw, evidence, features, visual is not None)

    def _result(self, raw, evidence, features, camera_fresh):
        if raw in ("UNKNOWN", "INCONSISTENT"):
            self.state, self.pending, self.confirmations = raw, raw, 0
        else:
            self.confirmations = self.confirmations + 1 if self.pending == raw else 1
            self.pending = raw
            if self.confirmations >= self.thresholds["confirmation_samples"]:
                self.state = raw
        observability = ("conflicting" if raw == "INCONSISTENT" else "unobserved" if raw == "UNKNOWN"
                         else "observed" if camera_fresh and self.mode == "fusion" else "partial")
        return {"state": self.state, "raw_state": raw, "evidence": evidence,
                "observability": observability, "features": features,
                "recommended_skills": [skill["id"] for skill in self.skills
                                       if self.state in skill["entry_states"]]}
