"""Causal contact estimates from RGB/ArUco, encoders and compensated wrist wrenches."""

from collections import deque
import math

import numpy as np

from fusion_upper5 import CAMERA_HOLD_S, FRONT_LOAD_N, axes_for, yaw_of


MODES = ("vision_fk", "wrench_fk", "fusion_arrival", "fusion_compensated")
SIDES = ("left", "right")


class SevenObserver:
    def __init__(self, config, mode, delay_ms=0):
        if mode not in MODES or not math.isfinite(delay_ms) or delay_ms < 0:
            raise ValueError("Invalid observation mode or delivery delay")
        self.mode = mode
        self.delay = delay_ms / 1000.
        self.half = config["cube_size_m"] / 2.
        self.weight = config["physics"]["cube_mass_kg"] * abs(config["physics"]["gravity_m_s2"][2])
        self.boxes = {name: (np.asarray(p), np.asarray(size)) for name, p, size in config["static_boxes"]}
        self.walls = {}
        for name, axis, sign in (("wall_left", 0, -1), ("wall_right", 0, 1),
                                 ("wall_back", 1, 1), ("wall_front", 1, -1)):
            center, size = self.boxes[name]
            self.walls["/World/Environment/" + name] = (axis, sign, center - sign * size / 2)
        self.frames = deque()
        self.last_frame = None
        self.optical = None
        self.camera_history = deque(maxlen=5)
        self.history = deque()
        self.known_grip = np.zeros(2, dtype=bool)
        self.last_now = -math.inf
        self.previous_boundaries = set()

    def _read_camera(self, packet, now):
        if packet and packet["frame_id"] != self.last_frame:
            self.last_frame = packet["frame_id"]
            captured, received = packet.get("capture_time_s"), packet.get("received_time_s")
            if (packet.get("status") == "detected" and packet.get("position_m") is not None
                    and captured is not None and received is not None
                    and 0 <= captured <= received <= now + 1e-7):
                position = np.asarray(packet["position_m"], dtype=float)
                rotation = packet.get("cube_rotation_world")
                if position.shape == (3,) and np.all(np.isfinite(position)) and rotation is not None:
                    self.frames.append((received + self.delay, captured, position, yaw_of(rotation)))
        while self.frames and self.frames[0][0] <= now + 1e-7:
            arrival = self.frames.popleft()
            if self.optical is None or arrival[1] > self.optical[1]:
                self.optical = arrival
                self.camera_history.append(arrival)
        return self.optical

    def _support_level(self, center):
        # The upper support height is a geometric possibility inside this two-layer tray.
        # Its elevation alone never identifies which lower box bears the load.
        heights = []
        for name in ("floor", "supply"):
            pos, size = self.boxes[name]
            if np.all(np.abs(center[:2] - pos[:2]) <= size[:2] / 2 + self.half):
                heights.append(pos[2] + size[2] / 2 + self.half)
                if name == "floor":
                    heights.append(heights[-1] + 2 * self.half)
        return min((abs(center[2] - h) for h in heights), default=math.inf)

    def _boundaries(self, center, yaw, push, neighbors, now, loaded_faces=None):
        basis = axes_for(yaw)
        radius = self.half * np.abs(basis).sum(axis=1)
        hits = {}
        for path, (axis, sign, plane) in self.walls.items():
            name = path.rsplit("/", 1)[-1]
            wall, size = self.boxes[name]
            other = 1 - axis
            overlaps = (abs(center[other] - wall[other]) < size[other] / 2 + radius[other]
                        and abs(center[2] - wall[2]) < size[2] / 2 + self.half)
            coordinate = center[axis]
            band = .001
            if loaded_faces is not None and axis in loaded_faces:
                coordinate = loaded_faces[axis]
                band = .00045
            gap = sign * (plane[axis] - coordinate) - radius[axis]
            if overlaps and -.002 <= gap <= band and (push is None or push[axis] * sign >= .6):
                hits[path] = (axis, sign)
        if neighbors and self.mode != "wrench_fk":
            for path, info in neighbors.items():
                if not (info["capture_time_s"] <= info["received_time_s"] and
                        info["received_time_s"] + self.delay <= now + 1e-7):
                    continue
                other = np.asarray(info["position_m"])
                if abs(other[2] - center[2]) > self.half:
                    continue
                delta = other[:2] - center[:2]
                other_axes = axes_for(yaw_of(info["rotation_world"]))
                directions = np.column_stack((basis, other_axes))
                dot = directions.T @ delta
                span = self.half * (np.abs(directions.T @ basis).sum(axis=1)
                                    + np.abs(directions.T @ other_axes).sum(axis=1))
                gaps = np.abs(dot) - span
                idx = int(np.argmax(gaps))
                direction = np.sign(dot[idx]) * directions[:, idx]
                if -.004 <= gaps[idx] <= .0018 and (push is None or push[:2] @ direction >= .45):
                    axis = int(np.argmax(np.abs(direction)))
                    hits[path] = (axis, int(np.sign(direction[axis])))
        return hits

    def update(self, sensor_row, neighbors=None):
        now = sensor_row["time_s"]
        if not math.isfinite(now) or now <= self.last_now:
            raise ValueError("Sensor time must increase")
        self.last_now = now
        use_force = self.mode != "vision_fk"
        use_vision = self.mode != "wrench_fk"
        tcp = np.asarray([sensor_row["tcp_positions_m"][s] for s in SIDES], dtype=float)
        inward = np.asarray([sensor_row["cup_normals_world"][s] for s in SIDES], dtype=float)
        forces = (np.asarray([sensor_row[s + "_wrench_world"][:3] for s in SIDES]) if use_force
                  else np.zeros((2, 3)))
        if not all(np.all(np.isfinite(x)) for x in (tcp, inward, forces)):
            raise ValueError("Invalid encoder or wrench sample")
        compression = -np.sum(inward * forces, axis=1)
        force_norm = np.linalg.norm(forces, axis=1)
        loaded = ((np.abs(compression) >= FRONT_LOAD_N) | (np.abs(forces[:, 2]) > .9))
        vertical = -sum(forces[:, 2]) / self.weight
        self.history.append((now, tcp, loaded, vertical, -forces.sum(axis=0)))
        while len(self.history) > 2 and self.history[1][0] < now - CAMERA_HOLD_S - .05:
            self.history.popleft()
        optical = self._read_camera(sensor_row.get("camera"), now) if use_vision else None
        age = now - optical[1] if optical else None
        optical_ok = optical is not None and 0 <= age <= .40
        xyz = optical[2].copy() if optical_ok else None
        yaw = optical[3] if optical_ok else 0.
        opposed = bool(inward[0] @ inward[1] < -.8)
        holding_load = bool(np.all(loaded) and opposed and vertical > .68
                            and np.all(-forces[:, 2] > .1 * self.weight))
        self.known_grip &= loaded
        self.known_grip |= (compression < -.5) | (holding_load & loaded)
        if xyz is not None and self.mode == "fusion_compensated":
            segment = [e for e in self.history if e[0] >= optical[1] - 1e-7]
            if segment and abs(segment[0][0] - optical[1]) <= .012:
                continuous = np.logical_and.reduce([e[2] for e in segment])
                if any(continuous):
                    d = tcp - segment[0][1]
                    if all(continuous) and opposed and holding_load:
                        xyz += np.mean(d, axis=0)
                    else:
                        arm = np.argmax(np.where(continuous, np.abs(compression), -1.))
                        xyz += (d[arm] @ inward[arm]) * inward[arm]
        # Compression resolves the push-normal coordinate; it does not fix tangential slip.
        if loaded.any():
            arm = int(np.argmax(np.where(loaded, force_norm, -1)))
            n = inward[arm]
            face_center = tcp[arm] + self.half * n
            if xyz is None:
                xyz = face_center.copy()
            else:
                axial_error = (face_center - xyz) @ n
                xyz += .85 * axial_error * n
        support_error = self._support_level(xyz) if xyz is not None else math.inf
        recent = next((e for e in reversed(self.history) if e[0] <= now - .10), self.history[0])
        vertical_change = abs(vertical - recent[3])
        palm_motion_z = np.max(np.abs(tcp[:, 2] - recent[1][:, 2]))
        hands = {s: "loaded" if loaded[i] else "unconfirmed" for i, s in enumerate(SIDES)}
        if xyz is not None:
            separation = np.linalg.norm(np.maximum(np.abs(tcp - xyz) - self.half, 0), axis=1)
            for i, s in enumerate(SIDES):
                if not loaded[i] and separation[i] > .025:
                    hands[s] = "separated"
        state, hits = "UNKNOWN", {}
        if not use_force and optical_ok and len(self.camera_history) >= 3:
            first = self.camera_history[0]
            earlier = min(self.history, key=lambda entry: abs(entry[0] - first[1]))
            at_exposure = min(self.history, key=lambda entry: abs(entry[0] - optical[1]))
            if (optical[1] - first[1] >= .1 and age <= .25
                    and abs(earlier[0] - first[1]) < .012 and abs(at_exposure[0] - optical[1]) < .012):
                delta_box = optical[2] - first[2]
                delta_tool = at_exposure[1] - earlier[1]
                offsets = xyz - tcp
                face_error = np.abs(np.sum(offsets * inward, axis=1) - self.half)
                touch = (face_error < .012) & (np.linalg.norm(offsets, axis=1) < 1.35 * self.half)
                follows = np.linalg.norm(delta_tool - delta_box, axis=1) < .01
                if all(touch & follows) and opposed and support_error > .04:
                    state = "S4"
                elif all(touch & follows) and opposed and abs(delta_box[2]) > .008 and support_error < .025:
                    state = "S3"
                elif touch.any() and support_error < .04:
                    if np.linalg.norm(delta_box[:2]) < .0015:
                        hits = self._boundaries(xyz, yaw, None, neighbors, now)
                    if hits:
                        state = "S6" if len(set(hits.values())) >= 2 else "S5"
                    elif np.any(touch & follows) and np.linalg.norm(delta_box[:2]) > .0015:
                        state = "S1"
        if holding_load:
            state = "S4"
        elif (.12 < vertical < .88 and loaded.all() and opposed and
              (vertical_change > .07 or palm_motion_z > .0004) and support_error < .04):
            state = "S3"
        elif loaded.any() and support_error <= .04 and abs(vertical) < .40:
            push = -forces.sum(axis=0)
            aligned = {}
            for i, n in enumerate(inward):
                axis = int(np.argmax(np.abs(n[:2])))
                if loaded[i] and abs(n[axis]) > .95:
                    aligned[axis] = (i, tcp[i, axis] + self.half * n[axis])
            hits = self._boundaries(xyz, yaw, push, neighbors, now,
                                    {axis: value[1] for axis, value in aligned.items()})
            earlier = next((e for e in reversed(self.history) if e[0] <= now - .24), None)
            for path, (axis, sign) in list(hits.items()):
                if path in self.previous_boundaries:
                    continue
                # A new constraint needs a causal force rise along its normal while motion slows.
                # A retained transverse constraint can be inferred from the other loaded wrist.
                if earlier is None or not loaded.any() or (push[axis] - earlier[4][axis]) * sign < .30:
                    del hits[path]
                    continue
                i = aligned[axis][0] if axis in aligned else arm
                if (not earlier[2][i] or abs(tcp[i, axis] - earlier[1][i, axis]) > .000020):
                    del hits[path]
            if hits:
                state = "S6" if len(set(hits.values())) >= 2 else "S5"
            elif self.known_grip.any():
                state = "S2"
            elif sum(loaded) == 1 and np.max(compression) >= FRONT_LOAD_N:
                state = "S1"
        elif (optical_ok and support_error <= .04 and all(v == "separated" for v in hands.values())
              and (not use_force or np.max(force_norm) < .3)):
            state = "S0"
        self.previous_boundaries = set(hits)
        for i, side in enumerate(SIDES):
            if self.known_grip[i]:
                hands[side] = "inferred_seal"
        return {"state": state, "boundaries": sorted(hits), "camera_age_s": age,
                "position_m": xyz.tolist() if xyz is not None else None,
                "tools": hands, "vertical_load_fraction": vertical if use_force else None}
