"""Causal two-boundary observer for the upper fourth box (side +X, rear +Y)."""

from collections import deque
import math

import numpy as np

from active_probe_replay import (AVERAGE_END_S, MAX_COMMAND_M, MAX_TCP_FOLLOW_FRACTION,
                                 MIN_COMMAND_M, MIN_FORCE_RISE_N, WINDOW_S, response)
from fusion_upper5 import FRONT_LOAD_N, axes_for, yaw_of
from seven_observer import SevenObserver


SIDE = "/World/Cube_8"
REAR = "/World/Environment/wall_back"


class U4ContactObserver:
    """Input comprises the marker, wrist wrenches, encoders and issued right-arm Y target."""

    def __init__(self, config):
        self.base = SevenObserver(config, "fusion_compensated")
        self.window = deque()
        self.direct_side = False
        self.wall = False

    def update(self, sensor, neighbors=None):
        result = self.base.update(sensor, neighbors)
        now = sensor["time_s"]
        left, right = (np.asarray(sensor[s + "_wrench_world"], dtype=float) for s in ("left", "right"))
        n_left, n_right = (np.asarray(sensor["cup_normals_world"][s]) for s in ("left", "right"))
        forward = n_right[1] > .95 and -right[:3] @ n_right >= FRONT_LOAD_N
        lateral = n_left[0] > .95 and -left[:3] @ n_left >= FRONT_LOAD_N
        y_tcp = sensor["tcp_positions_m"]["right"][1]
        command = sensor.get("right_command_y_m")
        if forward and command is not None and math.isfinite(command):
            if self.window and now - self.window[-1][0] > .013:
                self.window.clear()
            self.window.append((now, (command, y_tcp, -right[1])))
            while len(self.window) > 1 and self.window[1][0] <= now - WINDOW_S + 1.e-7:
                self.window.popleft()
        else:
            self.window.clear()

        side = rear = None
        side_gap = rear_gap = None
        visible = result["camera_age_s"] is not None and result["camera_age_s"] <= .4
        witnessed = None
        if visible:
            center = result["position_m"]
            yaw = self.base.optical[3]
            radius = self.base.half * np.abs(axes_for(yaw)).sum(axis=1)
            side_marker = (neighbors or {}).get(SIDE)
            if side_marker and side_marker["capture_time_s"] <= side_marker["received_time_s"] <= now + 1.e-7:
                neighbor = side_marker["position_m"]
                neighbor_radius = self.base.half * np.abs(axes_for(yaw_of(side_marker["rotation_world"]))).sum(axis=1)
                if abs(neighbor[2] - center[2]) < self.base.half:
                    side_gap = neighbor[0] - center[0] - radius[0] - neighbor_radius[0]
                    if (abs(neighbor[1] - center[1]) > radius[1] + neighbor_radius[1] + .01
                            or side_gap > .012):
                        side = False
                    elif side_gap >= -.012:
                        if lateral:
                            self.direct_side |= SIDE in result["boundaries"]
                            if self.direct_side:
                                side = True
                        elif forward:
                            # Positive X pressure or signed yaw torque on the front cup supports a side reaction.
                            if right[0] < -.15 or (right[5] < -.055 and right[0] < .35):
                                side = True
                            elif right[0] > .35 and right[5] > -.015:
                                side = False
            wall_plane = self.base.walls[REAR][2][1]
            front_center_y = y_tcp + self.base.half * n_right[1] if forward else center[1]
            rear_gap = wall_plane - front_center_y - radius[1]
            if rear_gap > (.002 if forward else .015):
                rear = False
            if forward and -.002 <= rear_gap <= .0008:
                if self.window and now - self.window[0][0] >= WINDOW_S - 1.e-6:
                    # The capture runs at 120 Hz; select averaging windows by sensor timestamps.
                    width = sum(row[0] < self.window[0][0] + AVERAGE_END_S - 1.e-7 for row in self.window)
                    delta = response([row[1] for row in self.window], max(1, width))
                    advance = delta["command_advance_m"]
                    if (MIN_COMMAND_M <= advance <= MAX_COMMAND_M and
                            abs(delta["encoder_advance_m"]) <= MAX_TCP_FOLLOW_FRACTION * advance and
                            delta["axial_force_rise_N"] >= MIN_FORCE_RISE_N):
                        self.wall = True
                        witnessed = delta
                if self.wall:
                    rear = True
        if not lateral or side is False:
            self.direct_side = False
        if not forward or rear is False or not visible:
            self.wall = False

        boundaries = [path for path, label in ((SIDE, side), (REAR, rear)) if label is True]
        if forward or lateral:
            if len(boundaries) == 2:
                state = "S6"
            elif len(boundaries) == 1 and side is not None and rear is not None:
                state = "S5"
            elif side is None or rear is None:
                state = "UNKNOWN"
            else:
                state = "S1"
        else:
            state = result["state"] if result["state"] not in ("S5", "S6") else "UNKNOWN"
        return {**result, "state": state, "boundaries": sorted(boundaries),
                "side_loaded": side, "rear_loaded": rear, "side_gap_m": side_gap,
                "rear_gap_m": rear_gap, "wall_response": witnessed}
