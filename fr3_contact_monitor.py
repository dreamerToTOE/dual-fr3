"""Fail closed on unintended PhysX contacts during the FR3 experiment.

This is an execution guard; collision-free trajectory validation must happen
before execution.  Call ``raise_if_collision`` after every physics step.
"""
from __future__ import annotations

import math


CUBE_CONTACT_LABELS = ("left_cup", "right_cup", "floor", "supply", "wall_back",
                       "wall_left", "wall_right", "wall_front")


def _cube_contact_label(path0, path1):
    if "/World/Cube" not in (path0, path1):
        return None
    other = path1 if path0 == "/World/Cube" else path0
    if other.startswith("/World/Environment/") and other.rsplit("/", 1)[-1] in CUBE_CONTACT_LABELS:
        return other.rsplit("/", 1)[-1]
    side = _robot_side(other)
    if side is not None:
        tools = (f"/World/{side}_fr3/ABSideTool/", f"/World/{side}_fr3/fr3_link8/ABSideTool/")
        if other in {tool + f"cup_r{row}_c{column}" for tool in tools
                     for row in (1, 2) for column in (1, 2)}:
            return side + "_cup"
    return None


class CubeContactTracker:
    """PhysX pair lifecycle for evaluation only; never a recognition input.

    Sleeping pairs remain active until LOST. Last separation and impulse are
    explicitly dated; an old impulse is never reused as current load evidence.
    ContactEventType in Isaac 4.5 is FOUND=0, LOST=1, PERSIST=2.
    """

    def __init__(self):
        self.physics_step = None
        self._pairs = {}

    def advance(self, physics_step):
        if physics_step is None:
            return
        if not isinstance(physics_step, int) or physics_step < 0:
            raise ValueError("A contact snapshot needs a nonnegative integer physics step")
        if self.physics_step is not None and physics_step < self.physics_step:
            self._pairs.clear()
        self.physics_step = physics_step

    def update(self, path0, path1, *, event_type, physics_step, separation_m=None, normal_impulse_Ns=None):
        label = _cube_contact_label(path0, path1)
        if label is None or physics_step is None:
            return
        if self.physics_step is not None and physics_step < self.physics_step:
            return
        self.advance(physics_step)
        pair = tuple(sorted((path0, path1)))
        if int(event_type) == 1:
            self._pairs.pop(pair, None)
            return
        if int(event_type) not in (0, 2):
            raise ValueError("Unknown PhysX contact event")
        if ((separation_m is not None and not math.isfinite(separation_m))
                or normal_impulse_Ns is not None and (not math.isfinite(normal_impulse_Ns) or normal_impulse_Ns < 0)):
            raise ValueError("Contact separation and nonnegative impulse must be finite")
        previous = self._pairs.get(pair, {})
        self._pairs[pair] = {
            "collider0": pair[0], "collider1": pair[1], "label": label,
            "separation_m": separation_m if separation_m is not None else previous.get("separation_m"),
            "normal_impulse_Ns": normal_impulse_Ns,
            "last_observed_physics_step": physics_step,
            "last_separation_physics_step": (physics_step if separation_m is not None
                                             else previous.get("last_separation_physics_step")),
        }

    def snapshot(self, physics_step):
        valid = physics_step is not None and physics_step == self.physics_step
        result = {"physics_step_index": physics_step, "valid": valid,
                  "source": "PhysX contact lifecycle and contact-data normal impulse; evaluation only",
                  "contacts": dict.fromkeys(CUBE_CONTACT_LABELS, False if valid else None),
                  "loaded_contacts": dict.fromkeys(CUBE_CONTACT_LABELS, False if valid else None), "pairs": []}
        if not valid:
            return result
        for pair in sorted(self._pairs):
            value = self._pairs[pair]
            fresh = value["last_observed_physics_step"] == physics_step
            touching = value["separation_m"] is not None and value["separation_m"] <= 0
            loaded = touching or (fresh and value["normal_impulse_Ns"] is not None
                                  and value["normal_impulse_Ns"] > 1e-8)
            result["contacts"][value["label"]] |= touching
            result["loaded_contacts"][value["label"]] |= loaded
            result["pairs"].append({**value, "current_step_observed": fresh,
                                    "current_separation_observed": value["last_separation_physics_step"] == physics_step,
                                    "touching": touching, "loaded_contact": loaded})
        return result


def _contact_normal_impulse(points):
    total = 0.0
    for point in points:
        if not hasattr(point, "impulse") or not hasattr(point, "normal"):
            return None
        impulse, normal = tuple(point.impulse), tuple(point.normal)
        if (len(impulse) != 3 or len(normal) != 3
                or not all(math.isfinite(value) for value in (*impulse, *normal))):
            raise ValueError("PhysX contact impulse and normal must be finite vectors")
        norm = math.sqrt(sum(value * value for value in normal))
        if norm <= 0:
            raise ValueError("PhysX contact normal cannot be zero")
        total += abs(sum(a * b for a, b in zip(impulse, normal))) / norm
    return total if points else None


def _robot_side(path):
    for side in ("left", "right"):
        root = f"/World/{side}_fr3"
        if path == root or path.startswith(root + "/"):
            return side
    return None


def is_unexpected_robot_contact(path0, path1, allowed_cube_contacts=(), separation_m=0.0):
    """Allow authorized cups to touch Cube with at most 3 mm penetration."""
    if _robot_side(path0) is None and _robot_side(path1) is None:
        return False
    for cup, other in ((path0, path1), (path1, path0)):
        side = _robot_side(cup)
        if side not in allowed_cube_contacts or other != "/World/Cube":
            continue
        root = f"/World/{side}_fr3"
        # The standalone loading smoke still uses flange-owned tool geometry.
        tools = (f"{root}/ABSideTool/", f"{root}/fr3_link8/ABSideTool/")
        if cup in {tool + f"cup_r{r}_c{c}" for tool in tools for r in (1, 2) for c in (1, 2)}:
            return separation_m < -0.003
    return True


class PhysxContactMonitor:
    def __init__(self, stage):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, UsdPhysics

        self.allowed_cube_contacts = set()
        self.phase = "initialization"
        self.contact_count = 0
        self.unexpected_contact_count = 0
        self.first_collision = None
        self.minimum_separation_m = None
        self.minimum_separation_by_pair = {}
        self.minimum_separation_by_phase_pair = {}
        self.physics_step_index = None
        self.cube_contacts = CubeContactTracker()
        self.reporter_count = 0
        self.cube_reporter_count = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if (path == "/World/Cube" and prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                self.cube_reporter_count += 1
            if (_robot_side(path) is not None
                    and prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                self.reporter_count += 1
        if not self.reporter_count:
            raise ValueError("No FR3 rigid bodies found for PhysX contact monitoring")
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contacts)

    def _on_contacts(self, headers, data):
        from pxr import PhysicsSchemaTools

        for header in headers:
            path0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            path1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            start = header.contact_data_offset
            points = [data[i] for i in range(start, start + header.num_contact_data)]
            separation = min((float(point.separation) for point in points), default=None)
            if _cube_contact_label(path0, path1) is not None:
                self.cube_contacts.update(path0, path1, event_type=getattr(header, "type", 0),
                                          physics_step=self.physics_step_index, separation_m=separation,
                                          normal_impulse_Ns=_contact_normal_impulse(points))
            if not header.num_contact_data:
                continue
            if _robot_side(path0) is None and _robot_side(path1) is None:
                continue
            self.contact_count += 1
            if self.minimum_separation_m is None or separation < self.minimum_separation_m:
                self.minimum_separation_m = separation
            pair = " <-> ".join(sorted((path0, path1)))
            previous = self.minimum_separation_by_pair.get(pair)
            if previous is None or separation < previous:
                self.minimum_separation_by_pair[pair] = separation
            phase_pairs = self.minimum_separation_by_phase_pair.setdefault(self.phase, {})
            previous = phase_pairs.get(pair)
            if previous is None or separation < previous["separation_m"]:
                phase_pairs[pair] = {
                    "separation_m": separation,
                    "physics_step_index": self.physics_step_index,
                }
            if (separation <= 0.0 and is_unexpected_robot_contact(
                    path0, path1, self.allowed_cube_contacts, separation)):
                self.unexpected_contact_count += 1
                if self.first_collision is None:
                    self.first_collision = {
                        "phase": self.phase, "collider0": path0, "collider1": path1,
                        "separation_m": separation,
                    }

    def raise_if_collision(self):
        if self.first_collision is not None:
            hit = self.first_collision
            raise RuntimeError(
                f"Unintended PhysX contact during {hit['phase']}: "
                f"{hit['collider0']} <-> {hit['collider1']}, "
                f"separation={hit['separation_m']:.6f} m"
            )

    def cube_contact_snapshot(self):
        self.cube_contacts.advance(self.physics_step_index)
        return self.cube_contacts.snapshot(self.physics_step_index)

    def report(self):
        return {
            "reporter_count": self.reporter_count,
            "cube_reporter_count": self.cube_reporter_count,
            "contact_count": self.contact_count,
            "unexpected_contact_count": self.unexpected_contact_count,
            "minimum_separation_m": self.minimum_separation_m,
            "minimum_separation_by_pair_m": self.minimum_separation_by_pair,
            "minimum_separation_by_phase_pair": self.minimum_separation_by_phase_pair,
            "first_collision": self.first_collision,
        }

    def close(self):
        self._subscription = None
