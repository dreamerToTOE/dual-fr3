"""Breakable Isaac Sim vacuum constraints at the FR3 side-tool suction faces."""
from __future__ import annotations

import math


ATTACHMENT_PARAMETERS = {
    "ray_standoff_m": 0.002,
    "ray_range_m": 0.003,
    "simulated_pressure_difference_Pa": 80000.0,
    "cup_radius_m": 0.01,
    "cup_count": 4,
    "cup_center_half_span_m": 0.024,
    "bend_angle_deg": 5.0,
    "bend_stiffness_Nm_per_rad": 30.0,
    "bend_damping_Nms_per_rad": 3.0,
    "payload_gravity_enabled": True,
}
# A scalar D6 torque limit uses the weaker bending lever of the four-cup layout.
_force_per_cup_N = (ATTACHMENT_PARAMETERS["simulated_pressure_difference_Pa"]
                    * math.pi * ATTACHMENT_PARAMETERS["cup_radius_m"] ** 2)
ATTACHMENT_PARAMETERS["aggregate_normal_capacity_N"] = ATTACHMENT_PARAMETERS["cup_count"] * _force_per_cup_N
ATTACHMENT_PARAMETERS["force_limit_N"] = min(30.0, ATTACHMENT_PARAMETERS["aggregate_normal_capacity_N"])
ATTACHMENT_PARAMETERS["torque_limit_Nm"] = (
    ATTACHMENT_PARAMETERS["aggregate_normal_capacity_N"] * ATTACHMENT_PARAMETERS["cup_center_half_span_m"])
ATTACHMENT_PARAMETERS["model_scope"] = (
    "Four-cup pressure-area capacity; scalar torque uses the weak bending lever. "
    "Resultant force is additionally capped at the 30 N arm acceptance limit. "
    "Isotropic D6 shear and pressure difference are uncalibrated simulation assumptions.")


class FR3SurfaceGrippers:
    def __init__(self, tcp_offsets):
        import omni.physics.tensors
        from isaacsim.robot.surface_gripper._surface_gripper import (
            Surface_Gripper,
            Surface_Gripper_Properties,
        )

        p = ATTACHMENT_PARAMETERS
        self.grippers = {}
        for side, tcp in tcp_offsets.items():
            sign = -1.0 if side == "left" else 1.0
            properties = Surface_Gripper_Properties()
            properties.parentPath = f"/World/{side}_fr3/ABSideTool"
            properties.d6JointPath = f"{properties.parentPath}/A_suction_constraint"
            offset = omni.physics.tensors.Transform()
            offset.p = (float(tcp[0]), float(tcp[1]) - sign * p["ray_standoff_m"], float(tcp[2]))
            # The plugin casts its ray along offset +X. The cups face along tool ±Y.
            offset.r = (0, 0, sign * math.sqrt(0.5), math.sqrt(0.5))
            properties.offset = offset
            properties.gripThreshold = p["ray_range_m"]
            properties.forceLimit = p["force_limit_N"]
            properties.torqueLimit = p["torque_limit_Nm"]
            properties.bendAngle = math.radians(p["bend_angle_deg"])
            properties.stiffness = p["bend_stiffness_Nm_per_rad"]
            properties.damping = p["bend_damping_Nms_per_rad"]
            properties.disableGravity = False
            properties.retryClose = False
            gripper = Surface_Gripper()
            if not gripper.initialize(properties):
                raise RuntimeError(f"Could not initialize {side} Surface Gripper")
            self.grippers[side] = gripper

    def close(self, side):
        gripper = self.grippers[side]
        if gripper.close() and gripper.is_closed():
            return True
        gripper.open()
        return False

    def open(self, side):
        self.grippers[side].open()

    def update(self):
        for gripper in self.grippers.values():
            if gripper.is_closed():
                gripper.update()

    def is_closed(self, side):
        return self.grippers[side].is_closed()
