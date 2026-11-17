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

from fr3_wrench import gravity_wrench_local, transform_wrench


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
        self.links, self.tools = {}, {}
        for side, robot in robots.items():
            view = robot._articulation_view
            self.links[side] = view.get_link_index("ABSideTool")
            if "wrist_ft_joint" not in view._metadata.joint_names:
                raise RuntimeError(f"{side} wrist sensor joint was merged or omitted")
            self.tools[side] = SingleXFormPrim(f"/World/{side}_fr3/ABSideTool")
        self.samples = 0
        self.metadata = {
            "scene_id": config["scene_id"], "frequency_hz": 120,
            "channels": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "units": ["N", "N", "N", "N m", "N m", "N m"],
            "source": "PhysX articulation link incoming fixed-joint reaction",
            "local_frame": "child joint frame at flange origin; localPos1=0, localRot1=identity",
            "raw_sign": "parent-on-tool",
            "residual_sign": "tool-on-sensor",
            "gravity_correction": "residual = -raw - tool_gravity; no payload gravity subtraction",
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
        for side, robot in self.robots.items():
            raw = np.asarray(robot.get_measured_joint_forces(), dtype=float)[self.links[side]]
            if raw.shape != (6,) or not np.all(np.isfinite(raw)):
                raise RuntimeError(f"{side} wrist sensor returned an invalid wrench")
            position, orientation = self.tools[side].get_world_pose()
            rotation = quat_to_rot_matrix(orientation)
            cfg = self.cfg[side]
            gravity = gravity_wrench_local(
                rotation, cfg["mass_kg"], cfg["center_of_mass_m"], self.gravity)
            corrected = -raw - gravity
            corrected_world = transform_wrench(corrected, rotation, np.zeros(3))
            q = np.asarray(robot.get_joint_positions(joint_indices=self.joints[side]), dtype=float)
            qdot = np.asarray(robot.get_joint_velocities(joint_indices=self.joints[side]), dtype=float)
            if not all(np.all(np.isfinite(v)) for v in (position, orientation, corrected_world, q, qdot)):
                raise RuntimeError(f"{side} wrist sensor state is not finite")
            row[side] = {
                "raw_parent_on_tool_local": raw.tolist(), "tool_gravity_local": gravity.tolist(),
                "gravity_corrected_local": corrected.tolist(),
                "gravity_corrected_world": corrected_world.tolist(),
                "sensor_position_m": position.tolist(), "sensor_orientation_wxyz": orientation.tolist(),
                "q_rad": q.tolist(), "qdot_rad_s": qdot.tolist(),
            }
        self.log.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        self.samples += 1
        return row

    def close(self):
        self.log.close()
        return {"samples": self.samples, "frequency_hz": 120,
                "directory": str(self.directory), "finite_wrenches_only": True,
                "dynamic_compensation": False, "sensor_noise_added": False}
