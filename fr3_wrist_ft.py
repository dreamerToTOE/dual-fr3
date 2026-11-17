"""Physical fixed-joint wrist sensors, sampled after each PhysX step.

The measured incoming wrench is parent-on-tool, at the child joint anchor.
Both joint frames coincide with the flange; the published residual reverses
that sign and removes only the explicitly modelled tool's gravity.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from fr3_wrench import gravity_wrench_local, transform_wrench, wrench_at_origin


def wrench_feedback(raw_parent_on_tool_local, R_world_from_sensor, mass_kg,
                    com_sensor_m, gravity_world, tcp_offset_sensor_m):
    """Return same-sample wrenches at sensor and TCP, with tool gravity removed.

    TCP local axes are parallel to sensor axes. World values rotate those axes
    while retaining their respective sensor/TCP origins, not the world origin.
    """
    raw = np.asarray(raw_parent_on_tool_local, dtype=float)
    if raw.shape != (6,) or not np.all(np.isfinite(raw)):
        raise ValueError("Raw sensor wrench must contain six finite values")
    gravity = gravity_wrench_local(R_world_from_sensor, mass_kg, com_sensor_m, gravity_world)
    sensor = -raw - gravity
    tcp = wrench_at_origin(sensor[:3], sensor[3:], tcp_offset_sensor_m)
    return {
        "raw_parent_on_tool_local": raw.tolist(), "tool_gravity_local": gravity.tolist(),
        "gravity_corrected_local": sensor.tolist(),
        "gravity_corrected_world": transform_wrench(sensor, R_world_from_sensor, np.zeros(3)).tolist(),
        "tcp_gravity_corrected_local": tcp.tolist(),
        "tcp_gravity_corrected_world": transform_wrench(tcp, R_world_from_sensor, np.zeros(3)).tolist(),
    }


def add_wrist_sensor(stage, side, cfg):
    """Create a separate tool body, retaining the existing flange/TCP geometry."""
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    root = f"/World/{side}_fr3"
    flange_path = f"{root}/fr3_link8"
    flange = stage.GetPrimAtPath(flange_path)
    if not flange.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"Sensor parent is not a rigid body: {flange_path}")
    mass = float(cfg["mass_kg"])
    com = np.asarray(cfg["center_of_mass_m"], dtype=float)
    inertia = np.asarray(cfg["diagonal_inertia_kg_m2"], dtype=float)
    if (not np.isfinite(mass) or mass <= 0 or com.shape != (3,)
            or inertia.shape != (3,) or not np.all(np.isfinite(com))
            or not np.all(np.isfinite(inertia)) or np.any(inertia <= 0)
            or 2 * np.max(inertia) > np.sum(inertia)):
        raise ValueError("Invalid tool mass, center of mass, or principal inertia")
    tool_path = f"{root}/ABSideTool"
    tool = UsdGeom.Xform.Define(stage, tool_path)
    relative, _ = UsdGeom.XformCache().ComputeRelativeTransform(
        flange, stage.GetPrimAtPath(root))
    tool.AddTransformOp().Set(relative)
    UsdPhysics.RigidBodyAPI.Apply(tool.GetPrim())
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(tool.GetPrim())
    body.CreateEnableCCDAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(tool.GetPrim())
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com.tolist()))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia.tolist()))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1))
    joint = UsdPhysics.FixedJoint.Define(stage, f"{root}/wrist_ft_joint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(flange_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(tool_path)])
    for create in (joint.CreateLocalPos0Attr, joint.CreateLocalPos1Attr):
        create(Gf.Vec3f(0))
    for create in (joint.CreateLocalRot0Attr, joint.CreateLocalRot1Attr):
        create(Gf.Quatf(1))
    joint.CreateExcludeFromArticulationAttr(False)
    joint.CreateCollisionEnabledAttr(False)
    return tool_path


class WristForceRecorder:
    def __init__(self, robots, joints, config, directory):
        from isaacsim.core.prims import SingleXFormPrim

        self.robots, self.joints = robots, joints
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cfg = config["wrist_force_sensors"]
        self.gravity = np.asarray(config["physics"]["gravity_m_s2"], dtype=float)
        geometry = config["side_suction_tool"]
        self.tcp_offsets = {side: np.array([
            0.0, (-1 if side == "left" else 1) * float(geometry["tcp_side_offset_m"]),
            float(geometry["vertical_drop_m"]),
        ]) for side in robots}
        if not all(np.all(np.isfinite(offset)) for offset in self.tcp_offsets.values()):
            raise ValueError("Sensor-to-TCP offsets must be finite")
        self.links, self.tools = {}, {}
        for side, robot in robots.items():
            view = robot._articulation_view
            self.links[side] = view.get_link_index("ABSideTool")
            if "wrist_ft_joint" not in view._metadata.joint_names:
                raise RuntimeError(f"{side} wrist sensor joint was merged or omitted")
            self.tools[side] = SingleXFormPrim(f"/World/{side}_fr3/ABSideTool")
        self.samples = 0
        self.metadata = {
            "schema_version": 2,
            "scene_id": config["scene_id"], "frequency_hz": 120,
            "channels": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "units": ["N", "N", "N", "N m", "N m", "N m"],
            "source": "PhysX articulation link incoming fixed-joint reaction",
            "local_frame": "child joint frame at flange origin; localPos1=0, localRot1=identity",
            "raw_sign": "parent-on-tool",
            "residual_sign": "tool-on-sensor",
            "gravity_correction": "residual = -raw - tool_gravity; no payload gravity subtraction",
            "tcp_offset_sensor_m": {side: offset.tolist() for side, offset in self.tcp_offsets.items()},
            "tcp_conversion": "F_TCP=F_sensor; torque_TCP=torque_sensor-cross(sensor_to_TCP,F_sensor)",
            "tcp_local_axes": "parallel to sensor axes; origin translated to nominal suction-face center",
            "world_wrench_origins": "sensor fields at sensor origin; tcp fields at TCP origin; neither at world origin",
            "tcp_position_source": "same-step sensor pose plus fixed tool geometry, no Cube truth",
            "dynamic_compensation": False,
            "sensor_noise_model": "ideal PhysX reaction, no added noise/latency",
            "phase_is_evaluation_context_only": True,
            "cube_truth_used_in_sensor": False,
            "tool_parameters": self.cfg,
            "cube_mass_kg_reference": config["physics"]["cube_mass_kg"],
            "gravity_world_m_s2": self.gravity.tolist(),
            "link_indices": self.links,
            "rail_transfer_note": "articulation base teleport; transients are not inertial calibration data",
        }
        (self.directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n")
        self.log = gzip.open(self.directory / "wrench.jsonl.gz", "wt", encoding="utf-8")

    def sample(self, time_s, physics_step, phase):
        from isaacsim.core.utils.rotations import quat_to_rot_matrix

        row = {"time_s": float(time_s), "physics_step": int(physics_step), "phase": phase}
        if hasattr(self, "command_context"):
            row["control"] = self.command_context
        for side, robot in self.robots.items():
            raw = np.asarray(robot.get_measured_joint_forces(), dtype=float)[self.links[side]]
            if raw.shape != (6,) or not np.all(np.isfinite(raw)):
                raise RuntimeError(f"{side} wrist sensor returned an invalid wrench")
            position, orientation = self.tools[side].get_world_pose()
            rotation = quat_to_rot_matrix(orientation)
            cfg = self.cfg[side]
            feedback = wrench_feedback(
                raw, rotation, cfg["mass_kg"], cfg["center_of_mass_m"],
                self.gravity, self.tcp_offsets[side])
            tcp_position = position + rotation @ self.tcp_offsets[side]
            q = np.asarray(robot.get_joint_positions(joint_indices=self.joints[side]), dtype=float)
            qdot = np.asarray(robot.get_joint_velocities(joint_indices=self.joints[side]), dtype=float)
            if not all(np.all(np.isfinite(v)) for v in (position, orientation, tcp_position, q, qdot)):
                raise RuntimeError(f"{side} wrist sensor state is not finite")
            row[side] = {
                **feedback,
                "sensor_position_m": position.tolist(), "sensor_orientation_wxyz": orientation.tolist(),
                "tcp_position_m": tcp_position.tolist(),
                "q_rad": q.tolist(), "qdot_rad_s": qdot.tolist(),
            }
        self.log.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        self.samples += 1
        return row

    def close(self):
        self.log.close()
        return {"samples": self.samples, "frequency_hz": 120,
                "directory": str(self.directory), "finite_wrenches_only": True,
                "wrench_origins": ["sensor", "TCP"],
                "dynamic_compensation": False, "sensor_noise_added": False}
