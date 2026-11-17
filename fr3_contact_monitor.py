"""Fail closed on unintended PhysX contacts during the FR3 experiment.

This is an execution guard; collision-free trajectory validation must happen
before execution.  Call ``raise_if_collision`` after every physics step.
"""
from __future__ import annotations


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
        self.reporter_count = 0
        for prim in stage.Traverse():
            if (_robot_side(str(prim.GetPath())) is not None
                    and prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                self.reporter_count += 1
        if not self.reporter_count:
            raise ValueError("No FR3 rigid bodies found for PhysX contact monitoring")
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contacts)

    def _on_contacts(self, headers, data):
        from pxr import PhysicsSchemaTools

        for header in headers:
            if not header.num_contact_data:
                continue
            path0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            path1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
            if _robot_side(path0) is None and _robot_side(path1) is None:
                continue
            start = header.contact_data_offset
            separation = min(float(data[i].separation)
                             for i in range(start, start + header.num_contact_data))
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

    def report(self):
        return {
            "reporter_count": self.reporter_count,
            "contact_count": self.contact_count,
            "unexpected_contact_count": self.unexpected_contact_count,
            "minimum_separation_m": self.minimum_separation_m,
            "minimum_separation_by_pair_m": self.minimum_separation_by_pair,
            "minimum_separation_by_phase_pair": self.minimum_separation_by_phase_pair,
            "first_collision": self.first_collision,
        }

    def close(self):
        self._subscription = None
