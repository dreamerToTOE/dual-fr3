"""Mirror official FR3 kinematics to seed alternating B2 packing directions."""
from __future__ import annotations

import math
import numpy as np


WORLD_REFLECTION = np.diag([-1., 1., 1.])
TOOL_REFLECTION = np.diag([1., -1., 1.])
JOINT_REFLECTION = np.array([-1., 1., -1., 1., -1., 1., -1.])


def mirror_joints(values):
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2) or values.shape[-1] != 7 or not np.all(np.isfinite(values)):
        raise ValueError("FR3 mirroring needs a finite seven-joint vector or path")
    return values * JOINT_REFLECTION


def b2_role_seeds(plan, push_side):
    """Return nominal seeds by role; the current scene still requires replanning."""
    if plan.get("status") != "PASS_COLLISION_CHECKED_B2_PLAN" or push_side not in ("left", "right"):
        raise ValueError("A valid original B2 seed plan and arm are required")
    mirrored = push_side == "left"

    def joints(values):
        return mirror_joints(values) if mirrored else np.asarray(values, dtype=float)

    def point(values):
        value = np.asarray(values, dtype=float)
        return WORLD_REFLECTION @ value if mirrored else value

    def base(values, yaw):
        value = list(values)
        if len(value) == 3:
            value.append(yaw)
        if mirrored:
            value[0] = -value[0]
            value[3] = math.pi - value[3]
        return value

    rotation = np.asarray(plan["right_rotation_world"], dtype=float)
    if mirrored:
        rotation = WORLD_REFLECTION @ rotation @ TOOL_REFLECTION
    return {
        "push_side": push_side,
        "support_side": "right" if mirrored else "left",
        "push_rotation_world": rotation,
        "push_base_initial_m": base(plan["right_base_initial_m"], math.pi),
        "push_base_contact_m": base(plan["right_base_contact_m"], math.pi),
        "support_base_m": base(plan["left_base_m"], 0.),
        "cube_initial_m": point(plan["cube_initial_m"]),
        "cube_final_m": point(plan["cube_final_m"]),
        "transit_z_m": plan["transit_z_m"],
        "rail_hold_push_q": joints(plan["rail"]["hold_right_q_rad"]),
        "high_transit": joints(plan["phases"]["high_transit"]["joint_path_rad"]),
        "descent": joints(plan["phases"]["descent"]["joint_path_rad"]),
        "approach": joints(plan["phases"]["approach"]["joint_path_rad"]),
        "push": joints(plan["phases"]["contact"]["right_joint_path_rad"]),
        "support": joints(plan["phases"]["contact"]["left_joint_path_rad"]),
        "collision_validation": "replan_and_validate_in_current_scene",
    }


def validate_mirrored_fk(source_solver, mirrored_solver, joint_paths, *, source_tool_offset):
    """Assert reflection against Lula FK with the two supplied mirrored base poses."""
    source_tool_offset = np.asarray(source_tool_offset, dtype=float)
    mirrored_tool_offset = TOOL_REFLECTION @ source_tool_offset
    maxima = {"flange_position_error_m": 0., "tcp_position_error_m": 0., "rotation_matrix_error": 0.}
    samples = 0
    for path in joint_paths:
        for q in np.atleast_2d(path):
            source_position, source_rotation = source_solver.compute_forward_kinematics("fr3_link8", q)
            position, rotation = mirrored_solver.compute_forward_kinematics("fr3_link8", mirror_joints(q))
            source_position, position = np.asarray(source_position).reshape(3), np.asarray(position).reshape(3)
            source_rotation, rotation = np.asarray(source_rotation), np.asarray(rotation)
            source_tcp = source_position + source_rotation @ source_tool_offset
            tcp = position + rotation @ mirrored_tool_offset
            errors = {
                "flange_position_error_m": np.linalg.norm(position - WORLD_REFLECTION @ source_position),
                "tcp_position_error_m": np.linalg.norm(tcp - WORLD_REFLECTION @ source_tcp),
                "rotation_matrix_error": np.linalg.norm(rotation - WORLD_REFLECTION @ source_rotation @ TOOL_REFLECTION),
            }
            maxima = {name: max(maxima[name], float(error)) for name, error in errors.items()}
            samples += 1
    if samples == 0 or max(maxima.values()) > 1e-9:
        raise RuntimeError(f"Official FR3 FK mirror validation failed {maxima}")
    return {"status": "PASS_OFFICIAL_LULA_FK_REFLECTION", "samples": samples, **maxima}
