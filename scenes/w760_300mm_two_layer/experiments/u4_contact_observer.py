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
SIDE_HOLD_S = .20
SILENT_LOSS_FORCE_N = .15
SILENT_LOSS_RETRACT_M = .0003
SILENT_LOSS_RIGHT_MOTION_M = .00008
SILENT_LOSS_WINDOW_S = .25
SILENT_LOSS_CONFIRM_STEPS = 12
U4_SILENT_LOSS_PROBE_ACTIVE = False


class U4ContactObserver:
    """Input comprises the marker, wrist wrenches, encoders and issued right-arm Y target."""

    def __init__(self, config, *, handover=False):
        self.base = SevenObserver(config, "fusion_compensated", revised=handover)
        self.handover = handover
        self.lateral_seen = False
        self.supported_view = None
        self.side_streak = deque(maxlen=3)
        self.confirmed_side = None
        self.last_side_true_s = None
        self.side_false_streak = 0
        self.sustained = None
        if handover:
            from u4_sustained_observer import SustainedSideObserver
            self.sustained = SustainedSideObserver()
        self.tcp_motion_window = deque()
        self.silent_loss_anchor_x = None
        self.silent_loss_streak = 0
        self.silent_loss_active = False
        self.silent_loss_retract_m = None
        self.silent_loss_right_motion_m = None
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
        right_tcp = np.asarray(sensor["tcp_positions_m"]["right"], dtype=float)
        if self.tcp_motion_window and now - self.tcp_motion_window[-1][0] > .014:
            self.tcp_motion_window.clear()
        self.tcp_motion_window.append((now, right_tcp.copy()))
        while (len(self.tcp_motion_window) > 1
               and self.tcp_motion_window[1][0] <= now - SILENT_LOSS_WINDOW_S + 1.e-7):
            self.tcp_motion_window.popleft()
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
            raw_side = side
            side_hold_used = False
            if raw_side is False:
                self.side_false_streak += 1
            else:
                self.side_false_streak = 0
            if raw_side is True:
                self.silent_loss_anchor_x = float(sensor["tcp_positions_m"]["left"][0])
                self.silent_loss_streak = 0
                self.silent_loss_active = False
            elif self.silent_loss_anchor_x is None and self.confirmed_side is True:
                self.silent_loss_anchor_x = float(sensor["tcp_positions_m"]["left"][0])
            if exchange and result["state"] in ("S1", "S5", "S6", "UNKNOWN"):
                self.side_streak.append(raw_side)
                confirmed_sample = (len(self.side_streak) == self.side_streak.maxlen
                                     and all(value is raw_side for value in self.side_streak))
                if raw_side is True and self.confirmed_side is True:
                    # A reload after a confirmed bearing does not need to wait
                    # for a second three-sample acquisition.
                    side = True
                elif not confirmed_sample:
                    side = None
                # The right wrist side reaction is intermittent while the rear
                # response is being established. Once the side boundary has
                # been confirmed, bridge only this short, rear-contact window.
                # Three consecutive explicit side=False samples or an expired
                # window clears the evidence, so a sustained loss is not hidden.
                if (raw_side in (None, False) and side is None and rear is True and in_corridor
                        and self.confirmed_side is True
                        and (raw_side is None or self.side_false_streak < 3)
                        and self.last_side_true_s is not None
                        and now - self.last_side_true_s <= SIDE_HOLD_S + 1.e-7):
                    side = True
                    side_hold_used = True
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
            # The U4 positive probe deliberately removes the side sensor
            # evidence after the rear wall has already been confirmed.  Keep
            # the physical rear and encoder signals intact so the silent-loss
            # branch is exercised without changing the production observer.
            probe_side_dropout = bool(sensor.get("u4_silent_loss_probe_active", False)
                                     or U4_SILENT_LOSS_PROBE_ACTIVE)
            if probe_side_dropout:
                raw_side = None
                side = None
            side_loaded = side if side is not None else True if SIDE in bounds else None
            side_event = None
            if side_loaded is not None:
                if self.confirmed_side is not None and side_loaded != self.confirmed_side:
                    side_event = "side_load_recovered" if side_loaded else "side_load_lost"
                self.confirmed_side = side_loaded
                if raw_side is True:
                    self.last_side_true_s = now
                elif raw_side is False and side_loaded is False:
                    self.last_side_true_s = None
            elif self.last_side_true_s is not None and now - self.last_side_true_s > SIDE_HOLD_S:
                self.last_side_true_s = None
            sustained_output = None
            sustained_override = False
            if self.sustained is not None:
                sustained_output = self.sustained.update(
                    sensor,
                    {"side_loaded": side_loaded, "rear_loaded": True if REAR in bounds else rear,
                     "boundaries": sorted(bounds), "state": result["state"]},
                )
                if (exchange and rear is True and side_loaded is None
                        and sustained_output.get("sustained_applied")
                        and sustained_output.get("side_loaded") is False):
                    sustained_side = sustained_output["side_loaded"]
                    previous_side = self.confirmed_side
                    side_loaded = sustained_side
                    if sustained_side:
                        bounds.add(SIDE)
                    else:
                        bounds.discard(SIDE)
                    result = {**result, "state": ("S1", "S5", "S6")[min(len(bounds), 2)],
                              "boundaries": sorted(bounds)}
                    if previous_side is not None and sustained_side != previous_side:
                        side_event = "side_load_recovered" if sustained_side else "side_load_lost"
                    self.confirmed_side = sustained_side
                    sustained_override = True
            silent_loss_event = None
            left_tcp_x = float(sensor["tcp_positions_m"]["left"][0])
            if (exchange and rear is True and raw_side is None
                    and self.confirmed_side is True and in_corridor
                    and left_compression < SILENT_LOSS_FORCE_N
                    and self.silent_loss_anchor_x is not None):
                self.silent_loss_retract_m = float(
                    -n_left[0] * (left_tcp_x - self.silent_loss_anchor_x))
                positions = np.asarray([item[1] for item in self.tcp_motion_window])
                self.silent_loss_right_motion_m = float(
                    np.max(np.linalg.norm(positions - positions[0], axis=1))) if len(positions) else None
                if (self.silent_loss_retract_m >= SILENT_LOSS_RETRACT_M
                        and self.silent_loss_right_motion_m is not None
                        and self.silent_loss_right_motion_m <= SILENT_LOSS_RIGHT_MOTION_M):
                    self.silent_loss_streak += 1
                else:
                    self.silent_loss_streak = 0
            else:
                self.silent_loss_streak = 0
                if raw_side is not None or not exchange:
                    self.silent_loss_retract_m = None
                    self.silent_loss_right_motion_m = None
            if (self.silent_loss_streak >= SILENT_LOSS_CONFIRM_STEPS
                    and not self.silent_loss_active):
                self.silent_loss_active = True
                silent_loss_event = "side_load_lost_silent"
            if (self.silent_loss_active and exchange and rear is True
                    and raw_side is None):
                bounds.discard(SIDE)
                side_loaded = False
                result = {**result, "state": "S5", "boundaries": sorted(bounds)}
                if self.confirmed_side is not False:
                    silent_loss_event = silent_loss_event or "side_load_lost_silent"
                self.confirmed_side = False
            if silent_loss_event is not None:
                side_event = silent_loss_event
            return {**result, "side_loaded": side_loaded,
                    "rear_loaded": True if REAR in bounds else rear, "side_gap_m": side_gap,
                    "rear_gap_m": rear_gap, "handover_active": bool(exchange),
                    "supported_view_used": supported is not None,
                    "side_event": side_event,
                    "side_raw": raw_side,
                    "side_false_streak": self.side_false_streak,
                    "side_hold_used": bool(locals().get("side_hold_used", False)),
                    "silent_loss_active": bool(self.silent_loss_active),
                    "silent_loss_streak": self.silent_loss_streak,
                    "sustained_side_loaded": (None if sustained_output is None
                                               else sustained_output.get("side_loaded")),
                    "sustained_applied": sustained_override,
                    "sustained_model_applied": bool(sustained_output and
                                                     sustained_output.get("sustained_applied")),
                    "sustained_score": (None if sustained_output is None
                                         else sustained_output.get("sustained_score")),
                    "sustained_reason": (None if sustained_output is None
                                          else sustained_output.get("sustained_reason")),
                    "silent_loss_event": silent_loss_event,
                    "silent_loss_retract_m": self.silent_loss_retract_m,
                    "silent_loss_right_motion_m": self.silent_loss_right_motion_m,
                    "u4_silent_loss_probe_active": probe_side_dropout,
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
