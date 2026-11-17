"""Check cached A commands against the rigid bilateral grasp using offline URDF FK.

This checks command geometry, not contact forces or collision safety. No Isaac
startup is needed; --urdf points to the official FR3 motion-policy URDF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


class FR3UrdfTCP:
    """The project's seven revolute joints plus fixed fr3_link8 tool mount."""

    def __init__(self, urdf, config):
        self.config = config
        joints = {joint.find("child").get("link"): joint
                  for joint in ET.parse(urdf).getroot().findall("joint")}
        self.chain = []
        for index in range(1, 9):
            joint = joints[f"fr3_link{index}"]
            expected = "revolute" if index <= 7 else "fixed"
            if joint.get("type") != expected:
                raise ValueError(f"Expected official FR3 {expected} joint at link {index}")
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            axis = (np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    if expected == "revolute" else np.zeros(3))
            self.chain.append((transform, axis))

    def pose(self, side, q):
        q = np.asarray(q, dtype=float)
        if side not in ("left", "right") or q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("TCP FK requires a named FR3 arm and seven finite joint positions")
        base = self.config["fr3"]["bases"][side]
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler("z", base[3]).as_matrix()
        result[:3, 3] = base[:3]
        for index, (origin, axis) in enumerate(self.chain):
            result = result @ origin
            if index < 7:
                turn = np.eye(4)
                turn[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
                result = result @ turn
        tool = self.config["side_suction_tool"]
        sign = -1 if side == "left" else 1
        result[:3, 3] += result[:3, :3] @ [
            0, sign * tool["tcp_side_offset_m"], tool["vertical_drop_m"]]
        return result


def closure_metrics(tcp_poses, translation_limit_m=20e-6, rotation_limit_rad=1e-4):
    """Relative right TCP pose in the left TCP frame must remain constant."""
    poses = np.asarray(tcp_poses, dtype=float)
    if (poses.ndim != 4 or poses.shape[1:] != (2, 4, 4) or len(poses) < 2
            or not np.all(np.isfinite(poses))):
        raise ValueError("Expected at least two finite pairs of 4x4 TCP poses")
    rotations = poses[:, :, :3, :3]
    if (not np.allclose(np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-6)
            or not np.allclose(poses[:, :, 3], [0, 0, 0, 1], atol=1e-9)):
        raise ValueError("TCP poses must be proper rigid transforms")
    limits = np.asarray([translation_limit_m, rotation_limit_rad])
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("Closure limits must be finite and positive")
    left, right = poses[:, 0], poses[:, 1]
    left_inverse = np.swapaxes(left[:, :3, :3], 1, 2)
    relative_position = np.einsum("nij,nj->ni", left_inverse,
                                  right[:, :3, 3] - left[:, :3, 3])
    relative_rotation = left_inverse @ right[:, :3, :3]
    translation = np.linalg.norm(relative_position - relative_position[0], axis=1)
    rotation = Rotation.from_matrix(relative_rotation[0].T @ relative_rotation).magnitude()
    return {
        "status": ("PASS" if translation.max() <= translation_limit_m
                   and rotation.max() <= rotation_limit_rad else "FAIL"),
        "samples": len(poses),
        "reference": "first sample; relative right TCP pose expressed in left TCP axes",
        "initial_relative_position_m": relative_position[0].tolist(),
        "initial_relative_rotation_matrix": relative_rotation[0].tolist(),
        "maximum_translation_drift_m": float(translation.max()),
        "maximum_translation_drift_sample": int(np.argmax(translation)),
        "maximum_rotation_drift_rad": float(rotation.max()),
        "maximum_rotation_drift_sample": int(np.argmax(rotation)),
        "limits": {"translation_m": translation_limit_m, "rotation_rad": rotation_limit_rad},
    }


def analyze_cache(cache, kinematics, translation_limit_m=20e-6, rotation_limit_rad=1e-4):
    phases = []
    for index, entry in enumerate(cache["phases"]):
        if entry["phase"] != "A_transfer":
            continue
        samples = entry["joint_samples_rad"]
        if len(samples["left"]) != len(samples["right"]):
            raise ValueError("A bilateral command sample counts differ")
        poses = [[kinematics.pose(side, samples[side][i]) for side in ("left", "right")]
                 for i in range(len(samples["left"]))]
        phases.append({"cache_phase_index": index, "phase": entry["phase"],
                       "duration_s": entry["duration_s"],
                       **closure_metrics(poses, translation_limit_m, rotation_limit_rad)})
    return {"status": ("PASS" if phases and all(p["status"] == "PASS" for p in phases) else "FAIL"),
            "scope": "A_transfer command closed-chain geometry only; no force or collision validation",
            "phases": phases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("fr3_four_wall_scene_config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translation-limit-m", type=float, default=20e-6)
    parser.add_argument("--rotation-limit-rad", type=float, default=1e-4)
    args = parser.parse_args()
    cache = json.loads(args.cache.read_text())
    config = cache.get("scene_config") or json.loads(args.config.read_text())
    result = analyze_cache(cache, FR3UrdfTCP(args.urdf, config),
                           args.translation_limit_m, args.rotation_limit_rad)
    result.update({"cache": str(args.cache), "urdf": str(args.urdf)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
