"""Causal two-boundary observer for the upper fourth box (side +X, rear +Y)."""

from collections import deque
import math

import numpy as np

from active_probe_replay import (AVERAGE_END_S, MAX_COMMAND_M, MAX_TCP_FOLLOW_FRACTION,
                                 MIN_COMMAND_M, MIN_FORCE_RISE_N, WINDOW_S, response)
from fusion_upper5 import BLOCKING_LOAD_N, CAMERA_MAX_AGE_S, FRONT_LOAD_N, axes_for, yaw_of
from seven_observer import SevenObserver


SIDE = "/World/Cube_8"
REAR = "/World/Environment/wall_back"


class U4ContactObserver:
    """Input comprises the marker, wrist wrenches, encoders and issued right-arm Y target."""

    def __init__(self, config, *, handover=False):
        self.base = SevenObserver(config, "fusion_compensated", revised=handover)
        self.handover = handover
        self.lateral_seen = False
        self.supported_view = None
        self.side_streak = deque(maxlen=3)
        self.confirmed_side = None
        self.window = deque()
        self.direct_side = False
        self.wall = False

    def supported_pose(self, packet, now):
        if packet and packet.get("supported_capture_time_s") is not None:
            exposure, arrival = packet["supported_capture_time_s"], packet["received_time_s"]
            if (packet["status"] == "detected" and 0 <= exposure <= arrival <= now + 1e-7
                    and (self.supported_view is None or exposure > self.supported_view[0])):
                yaw = yaw_of(packet["supported_rotation_world"])
                self.supported_view = exposure, packet["supported_position_m"], yaw
        if self.supported_view is not None and now - self.supported_view[0] <= CAMERA_MAX_AGE_S:
            _, center, yaw = self.supported_view
            return center, yaw
        return None

    def update(self, sensor, neighbors=None):
        now = sensor["time_s"]
        if now - self.base.last_now > .014:
            self.side_streak.clear()
        result = self.base.update(sensor, neighbors)
        left, right = (np.asarray(sensor[s + "_wrench_world"], dtype=float) for s in ("left", "right"))
        n_left, n_right = (np.asarray(sensor["cup_normals_world"][s]) for s in ("left", "right"))
        forward = n_right[1] > .95 and -right[:3] @ n_right >= FRONT_LOAD_N
        left_compression = -left[:3] @ n_left
        lateral = n_left[0] > .95 and left_compression >= FRONT_LOAD_N
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
        supported = self.supported_pose(sensor.get("camera"), now) if self.handover else None
        side_overlap = .004 if supported is not None else .012
        in_corridor = False
        if visible:
            center = result["position_m"]
            yaw = self.base.optical[3]
            radius = self.base.half * np.abs(axes_for(yaw)).sum(axis=1)
            side_marker = (neighbors or {}).get(SIDE)
            if side_marker and side_marker["capture_time_s"] <= side_marker["received_time_s"] <= now + 1.e-7:
                neighbor = side_marker["position_m"]
                neighbor_radius = self.base.half * np.abs(axes_for(yaw_of(side_marker["rotation_world"]))).sum(axis=1)
                side_center, side_yaw = supported or (center, yaw)
                side_radius = self.base.half * np.abs(axes_for(side_yaw)).sum(axis=1) if supported else radius
                if abs(neighbor[2] - side_center[2]) < self.base.half:
                    side_gap = neighbor[0] - side_center[0] - side_radius[0] - neighbor_radius[0]
                    if (abs(neighbor[1] - side_center[1]) > side_radius[1] + neighbor_radius[1] + .01
                            or side_gap > .012):
                        side = False
                    elif side_gap >= -side_overlap:
                        in_corridor = True
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
                if self.handover:
                    self.wall |= REAR in result["boundaries"]
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

        if self.handover:
            if n_right[1] <= .95 or result["tool_retreat_complete"]["right"]:
                self.lateral_seen = False
            if forward and lateral and left_compression >= BLOCKING_LOAD_N and in_corridor:
                # The two tool orientations establish the U4 handoff context, not the side-load label.
                self.lateral_seen = True
            exchange = self.lateral_seen and forward and abs(left_compression) < FRONT_LOAD_N and visible
            bounds = set(result["boundaries"])
            if exchange and result["state"] in ("S1", "S5", "S6", "UNKNOWN"):
                self.side_streak.append(side)
                if len(self.side_streak) < self.side_streak.maxlen or any(value is not side for value in self.side_streak):
                    side = None
                if rear is True:
                    bounds.add(REAR)
                if side is True and in_corridor:
                    bounds.add(SIDE)
                elif side is False:
                    bounds.discard(SIDE)
                if sorted(bounds) != result["boundaries"]:
                    result = {**result, "state": ("S1", "S5", "S6")[min(len(bounds), 2)],
                              "boundaries": sorted(bounds)}
            else:
                self.side_streak.clear()
                side = None
            side_loaded = side if side is not None else True if SIDE in bounds else None
            side_event = None
            if side_loaded is not None:
                if self.confirmed_side is not None and side_loaded != self.confirmed_side:
                    side_event = "side_load_recovered" if side_loaded else "side_load_lost"
                self.confirmed_side = side_loaded
            return {**result, "side_loaded": side_loaded,
                    "rear_loaded": True if REAR in bounds else rear, "side_gap_m": side_gap,
                    "rear_gap_m": rear_gap, "handover_active": bool(exchange),
                    "supported_view_used": supported is not None,
                    "side_event": side_event,
                    "wall_response": witnessed}

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
