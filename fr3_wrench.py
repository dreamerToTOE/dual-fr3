"""Six-axis wrench math; vectors are [Fx, Fy, Fz, Tx, Ty, Tz] in N and N m.

These functions do not assume the sign convention of any simulator sensor.
"""
from __future__ import annotations

import numpy as np


def _finite(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return value


def _rotation(value):
    rotation = _finite(value, (3, 3), "rotation")
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        raise ValueError("rotation must be a proper orthonormal matrix")
    return rotation


def gravity_wrench_local(rotation_world_from_sensor, mass, com_sensor, gravity_world):
    """External tool gravity about the sensor origin, expressed in sensor axes.

    ``com_sensor`` points from the sensor origin to the tool center of mass in
    meters. This removes gravity only; accelerating tools also have inertial load.
    """
    rotation = _rotation(rotation_world_from_sensor)
    com = _finite(com_sensor, (3,), "com_sensor")
    gravity = _finite(gravity_world, (3,), "gravity_world")
    if not np.isscalar(mass) or not np.isfinite(mass) or mass < 0:
        raise ValueError("mass must be a finite nonnegative scalar in kilograms")
    force = rotation.T @ (mass * gravity)
    return np.concatenate((force, np.cross(com, force)))


def wrench_at_origin(force, torque, vector_old_to_new):
    """Shift a wrench to a new origin without rotating its axes.

    All three inputs use the same axes. ``vector_old_to_new`` points from the
    old origin to the new origin in meters; tau_new = tau_old - r cross F.
    """
    force = _finite(force, (3,), "force")
    torque = _finite(torque, (3,), "torque")
    offset = _finite(vector_old_to_new, (3,), "vector_old_to_new")
    return np.concatenate((force, torque - np.cross(offset, force)))


def transform_wrench(wrench, rotation_new_from_old, old_origin_in_new):
    """Re-express a wrench in new axes and about the new origin.

    ``rotation_new_from_old`` rotates old-frame vectors into new-frame axes.
    ``old_origin_in_new`` is the old origin's position in the new frame (m), so
    tau_new = R tau_old + old_origin_in_new cross (R F_old).
    """
    wrench = _finite(wrench, (6,), "wrench")
    rotation = _rotation(rotation_new_from_old)
    offset = _finite(old_origin_in_new, (3,), "old_origin_in_new")
    force = rotation @ wrench[:3]
    return np.concatenate((force, rotation @ wrench[3:] + np.cross(offset, force)))
