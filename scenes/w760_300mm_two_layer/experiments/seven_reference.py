"""Independent PhysX labels for the upright, two-layer packing experiment."""

from collections import deque
from math import isfinite
from statistics import median


MIN_IMPULSE_NS = 1e-5
WALL_FACE = {"wall_back": (1, 1), "wall_front": (1, -1),
             "wall_left": (0, -1), "wall_right": (0, 1)}


class SevenReference:
    """Use present-step physical contact, solver pose and the native seal flags only."""

    def __init__(self, config, hz=120, object_positions=None):
        self.size = config["cube_size_m"]
        self.floor = config["floor_top_m"]
        self.hz = hz
        physics = config["physics"]
        self.gravity = abs(physics["gravity_m_s2"][2])
        self.weight = physics["cube_mass_kg"] * self.gravity
        self.tops = {name: pos[2] + dims[2] / 2 for name, pos, dims in config["static_boxes"]
                     if name in ("floor", "supply")}
        self.floor_box = next((pos, dims) for name, pos, dims in config["static_boxes"] if name == "floor")
        self.centers = {str(key).rsplit("_", 1)[-1]: xyz for key, xyz in (object_positions or {}).items()}
        self.last_step = None
        self.cube_id = None
        self.seals = ()
        self.load = deque(maxlen=max(4, round(.12 * hz)))

    def _box_face(self, other, pair, pos):
        # Original runs retained point-normal impulse magnitudes only. Use the
        # measured neighboring box center to orient a witnessed contact pair.
        center = self.centers.get(other.rsplit("_", 1)[-1])
        if center is None:
            return None
        delta = [a - b for a, b in zip(center, pos)]
        impulse = pair.get("normal_axis_impulse_abs_Ns")
        if impulse is None:
            components = [abs(d) for d in delta]
        else:
            components = impulse
        axes = sorted(range(3), key=components.__getitem__, reverse=True)
        best, second = axes[:2]
        # A corner contact with no dominant normal has no unique face.
        if components[best] - components[second] < (.015 if impulse is None else .1 * components[best]):
            return None
        sign = 1 if delta[best] > 0 else -1
        return (best, sign)

    def update(self, frame):
        step = frame["physics_step"]
        cube_id = frame["active_cube_id"]
        if cube_id != self.cube_id or self.last_step is None or step != self.last_step + 1:
            self.load.clear()
        self.cube_id, self.last_step = cube_id, step
        pos = frame["cube_position_m"]
        sealed = sorted(k for k, held in frame["attachments"].items() if held)
        if tuple(sealed) != self.seals:
            self.load.clear()
        self.seals = tuple(sealed)
        supports, sides, cups, directions = set(), set(), set(), set()
        z_impulse, normal_load = 0., 0.
        has_axes = True
        uncertain = False
        tool_uncertain = False
        ambiguous_contacts = set()
        reason = "missing_current_contact_evidence"
        stable_support = False
        state, fraction, trend, inferred_tool_share = "UNKNOWN", None, None, None
        active = f"/World/Cube_{cube_id}"
        snapshot = frame["cube_contacts"]
        for pair in snapshot["pairs"] if snapshot["valid"] else ():
            if active not in (pair["collider0"], pair["collider1"]):
                continue
            other = pair["collider1"] if active == pair["collider0"] else pair["collider0"]
            impulse = pair.get("normal_impulse_Ns")
            separation = pair.get("separation_m")
            observed = pair["current_step_observed"]
            label = pair["label"]
            if not observed or impulse is None:
                # A dormant pair or missing contact manifold is not evidence of separation.
                if separation is None or separation <= 0. or (impulse or 0.) > MIN_IMPULSE_NS:
                    uncertain = True
                    ambiguous_contacts.add(other)
                    tool_uncertain |= label.endswith("_cup")
                continue
            if impulse <= MIN_IMPULSE_NS:
                if label.endswith("_cup") and separation is not None and separation <= 0.:
                    cups.add("unloaded_touch")
                continue
            axes = pair.get("normal_axis_impulse_abs_Ns")
            if label.endswith("_cup"):
                cups.add(label[:-4])
            elif label in self.tops and abs(pos[2] - self.size / 2 - self.tops[label]) < .02:
                supports.add(other)
                normal_load += impulse
                if axes is None:
                    has_axes = False
                else:
                    z_impulse += axes[2]
            elif label in WALL_FACE:
                direction = WALL_FACE[label]
                if axes and axes[direction[0]] < max(axes) * .9:
                    uncertain = True  # e.g. a contact on top of a high wall
                    ambiguous_contacts.add(other)
                else:
                    sides.add(other)
                    directions.add(direction)
            elif label == "cube_neighbor":
                face = self._box_face(other, pair, pos)
                if face == (2, -1):
                    supports.add(other)
                    normal_load += impulse
                    if axes is None:
                        has_axes = False
                    else:
                        z_impulse += axes[2]
                elif face is not None and face[0] < 2:
                    sides.add(other)
                    directions.add(face)
                else:
                    uncertain = True
                    ambiguous_contacts.add(other)
            else:
                uncertain = True
                ambiguous_contacts.add(other)
        if not uncertain and snapshot["valid"] and has_axes:
            fraction = z_impulse * self.hz / self.weight
            self.load.append((fraction, pos[2]))
            if len(self.load) == self.load.maxlen:
                samples = list(self.load)
                group = max(1, len(samples) // 4)
                trend = median(load for load, _ in samples[-group:]) - median(load for load, _ in samples[:group])
                # Difference of equal-duration pose slopes gives vertical acceleration
                # from physical positions, independently of the measured wrist wrench.
                loads, heights = zip(*samples)
                stable_support = (min(loads) >= .10 and max(loads) - min(loads) < .08
                                  and max(heights) - min(heights) < .0005)
                n = len(samples)
                v0 = (samples[group][1] - samples[0][1]) * self.hz / group
                v1 = (samples[-1][1] - samples[-1-group][1]) * self.hz / group
                acceleration = (v1 - v0) * self.hz / (n - 1 - group)
                if not sides:
                    inferred_tool_share = 1. + acceleration / self.gravity - fraction
        else:
            self.load.clear()

        # The box remains upright. A positive clearance above both possible support
        # levels under its footprint disambiguates flight; the supply pad is outside the tray.
        half_z = self.size * sum(abs(v) for v in frame["cube_rotation_world"][2]) / 2
        bottom = pos[2] - half_z
        gaps = [bottom - top for top in self.tops.values()]
        floor_center, floor_size = self.floor_box
        overlaps_tray = all(abs(pos[i] - floor_center[i]) <=
                            (floor_size[i] + self.size * sum(abs(r) for r in frame["cube_rotation_world"][i])) / 2
                            for i in (0, 1))
        if overlaps_tray:
            gaps.append(bottom - self.floor - self.size)
        clear = bottom > self.floor + .002 and all(abs(gap) > .002 for gap in gaps)
        # An unorientable lower-box edge cannot negate two independently witnessed
        # horizontal bearings, or reattach an already separated tool.
        if snapshot["valid"] and supports and not sealed and not cups and not tool_uncertain:
            state, reason = "S0", "tool_separated_bottom_bearing"
        elif snapshot["valid"] and supports and not sealed and cups - {"unloaded_touch"} and len(directions) > 1:
            state, reason = "S6", "multiple_bearing_faces"
        elif not uncertain and snapshot["valid"]:
            if sealed and not supports and clear:
                state, reason = "S4", "native_attachment_clear_of_support"
            elif (sealed and supports and fraction is not None and .04 <= fraction <= .94
                  and inferred_tool_share is not None and inferred_tool_share >= .05 and abs(trend) >= .08):
                state = "S3"
                reason = "loading_bottom" if trend > 0 else "unloading_bottom"
            elif supports and (sealed or cups - {"unloaded_touch"}):
                if len(directions) > 1:
                    state, reason = "S6", "multiple_bearing_faces"
                elif directions:
                    state, reason = "S5", "single_bearing_face"
                elif not sealed:
                    state, reason = "S1", "cup_impulse_bottom_bearing"
                elif stable_support:
                    state, reason = "S2", "native_attachment_stable_bottom_share"
                elif (fraction if fraction is not None else normal_load * self.hz / self.weight) >= .94:
                    state, reason = "S2", "native_attachment_bottom_bears_weight"
                else:
                    reason = "partial_support_without_confirmed_transfer"
        if fraction is not None and not isfinite(fraction):
            raise ValueError("Nonfinite support load")
        return {"state": state, "reason": reason, "support_load_fraction": fraction,
                "support_objects": sorted(supports), "horizontal_objects": sorted(sides),
                "sealed_arms": sealed, "inferred_tool_load_fraction": inferred_tool_share,
                "ambiguous_contacts": sorted(ambiguous_contacts)}
