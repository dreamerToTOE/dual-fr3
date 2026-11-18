"""Scalar normal-force admittance; positive position points into the contact."""
from __future__ import annotations

import math


def oriented_wall_gap(center_axis, rotation_row, half_size, target_center, inward_sign):
    """Signed gap to the target wall using the rotated Cube support radius."""
    return (inward_sign * (target_center - center_axis) + half_size
            - half_size * sum(abs(value) for value in rotation_row))


class NormalForceController:
    """Integrate an absolute normal position, bounded about its initial value.

    Force is positive for compression. Limits apply to commands, never to
    measured force. Clamping the integrator state prevents hidden windup.
    """

    def __init__(self, position_m, target_force_N=2.0, gain_m_per_Ns=1e-6,
                 max_offset_m=0.002, max_speed_m_s=0.00005, max_force_N=10.0):
        values = (position_m, target_force_N, gain_m_per_Ns,
                  max_offset_m, max_speed_m_s, max_force_N)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Force controller parameters must be finite")
        if (gain_m_per_Ns <= 0 or max_offset_m <= 0 or max_speed_m_s <= 0
                or max_force_N <= 0 or not 0 <= target_force_N <= max_force_N):
            raise ValueError("Invalid force controller gains, limits, or target")
        self.initial_position_m = float(position_m)
        self.position_m = float(position_m)
        self.target_force_N = float(target_force_N)
        self.gain_m_per_Ns = float(gain_m_per_Ns)
        self.max_offset_m = float(max_offset_m)
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_force_N = float(max_force_N)

    @property
    def offset_m(self):
        return self.position_m - self.initial_position_m

    def update(self, force_N, dt=1 / 120, target_force_N=None):
        target = self.target_force_N if target_force_N is None else target_force_N
        if (not all(math.isfinite(value) for value in (force_N, dt, target))
                or dt <= 0 or not 0 <= target <= self.max_force_N):
            raise ValueError("Force, target and positive sample interval must be finite and valid")
        if abs(force_N) > self.max_force_N:
            raise RuntimeError(f"Normal force overload: {force_N:.6f} N exceeds {self.max_force_N:.6f} N")
        self.target_force_N = float(target)
        velocity = self.gain_m_per_Ns * (self.target_force_N - force_N)
        velocity = max(-self.max_speed_m_s, min(self.max_speed_m_s, velocity))
        offset = self.offset_m + velocity * dt
        offset = max(-self.max_offset_m, min(self.max_offset_m, offset))
        self.position_m = self.initial_position_m + offset
        return self.position_m
