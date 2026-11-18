"""Full-arm planning checks using official Lula spheres and custom-tool OBBs.

These conservative planning checks do not certify the USD collision meshes.
Cup cylinders use enclosing boxes. Adjacent arm links and the
tool's wrist attachment are excluded from self checks because their proxy
volumes overlap by construction; all remain checked against the environment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fr3_trajectory import point_aabb_signed_clearance


def obb_signed_clearance(center_a, half_a, rotation_a, center_b, half_b, rotation_b):
    """SAT signed separation: <= 0 means the oriented boxes intersect.

    Positive separation is a conservative lower bound on Euclidean distance.
    """
    ra, rb = np.asarray(rotation_a), np.asarray(rotation_b)
    aabb_gap = np.max(np.abs(np.asarray(center_a) - center_b)
                      - np.abs(ra) @ half_a - np.abs(rb) @ half_b)
    if aabb_gap > 0:
        return float(aabb_gap)
    axes = np.concatenate((ra.T, rb.T,
                           np.cross(ra.T[:, None, :], rb.T[None, :, :]).reshape(-1, 3)))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-10] / norms[norms > 1e-10, None]
    separation = (np.abs(axes @ (np.asarray(center_a) - center_b))
                  - np.abs(axes @ ra) @ half_a - np.abs(axes @ rb) @ half_b)
    return float(np.max(separation))


def sphere_obb_clearance(center, radius, box_center, box_half, box_rotation):
    local = np.asarray(box_rotation).T @ (np.asarray(center) - box_center)
    return point_aabb_signed_clearance(local, np.zeros(3), 2 * np.asarray(box_half)) - radius


def _sphere_box_clearances(centers, radii, center, half, rotation):
    delta = np.abs((centers - center) @ rotation) - half
    return (np.linalg.norm(np.maximum(delta, 0), axis=1)
            + np.minimum(np.max(delta, axis=1), 0) - radii)


class FR3CollisionChecker:
    """Shared configuration/path validator for the planner and runtime.

    ``solvers`` are the current world-frame Lula FK solvers. Updating a solver's
    base pose immediately updates checks, including during rail motion.
    """

    def __init__(self, config, solvers, sphere_description=None):
        if sphere_description is None:
            import yaml
            from isaacsim.robot_motion.motion_generation import interface_config_loader

            path = interface_config_loader.load_supported_lula_kinematics_solver_config("FR3")["robot_description_path"]
            sphere_description = yaml.safe_load(Path(path).read_text())["collision_spheres"]
        self.config, self.solvers = config, solvers
        self.spheres = [(link, np.asarray(sphere["center"], dtype=float), float(sphere["radius"]))
                        for entry in sphere_description for link, spheres in entry.items()
                        if link.startswith("fr3_link") for sphere in spheres]
        if not self.spheres:
            raise ValueError("FR3 collision sphere description is empty")
        if any(center.shape != (3,) or not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0
               for _, center, radius in self.spheres):
            raise ValueError("Collision spheres must have finite centers and positive radii")
        self.link_indices = np.array([int(link.removeprefix("fr3_link")) for link, _, _ in self.spheres])
        self.radii = np.array([radius for _, _, radius in self.spheres])
        self.static_boxes = [(name, np.asarray(center, dtype=float), np.asarray(size, dtype=float) / 2, np.eye(3))
                             for name, center, size in config["static_boxes"]]
        self.other_cubes = []
        if any(center.shape != (3,) or half.shape != (3,) or not np.all(np.isfinite(center))
               or not np.all(np.isfinite(half)) or np.any(half <= 0)
               for _, center, half, _ in self.static_boxes):
            raise ValueError("Static boxes must have finite centers and positive sizes")
        if not np.isfinite(config["cube_size_m"]) or config["cube_size_m"] <= 0:
            raise ValueError("Cube size must be positive and finite")
        n = len(self.spheres)
        a, b = np.triu_indices(n, 1)
        mask = np.abs(self.link_indices[a] - self.link_indices[b]) > 1
        self.self_pairs = (a[mask], b[mask])

    def set_other_cubes(self, poses, *, preserve_unobserved=False):
        """Refresh observed obstacles, retaining occluded bodies when requested."""
        boxes = []
        for name, center, rotation in poses:
            center, rotation = np.asarray(center, dtype=float), np.asarray(rotation, dtype=float)
            if (not isinstance(name, str) or not name or name == "Cube"
                    or center.shape != (3,) or rotation.shape != (3, 3)
                    or not np.all(np.isfinite(center)) or not np.all(np.isfinite(rotation))
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
                raise ValueError("Other Cube obstacles need a name, finite center and proper rotation")
            boxes.append((name, center.copy(), np.full(3, self.config["cube_size_m"] / 2), rotation.copy()))
        if len({name for name, *_ in boxes}) != len(boxes):
            raise ValueError("Other Cube obstacle names must be unique")
        if preserve_unobserved:
            refreshed_names = {name for name, *_ in boxes}
            boxes = [box for box in self.other_cubes if box[0] not in refreshed_names] + boxes
        self.other_cubes = boxes

    def geometry(self, side, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("FR3 configuration must be seven finite joint positions")
        frames = {link: self.solvers[side].compute_forward_kinematics(link, q)
                  for link in dict.fromkeys(link for link, _, _ in self.spheres)}
        frames = {link: (np.asarray(p).reshape(3), np.asarray(r).reshape(3, 3))
                  for link, (p, r) in frames.items()}
        centers = np.array([frames[link][0] + frames[link][1] @ local
                            for link, local, _ in self.spheres])
        flange, rotation = self.solvers[side].compute_forward_kinematics("fr3_link8", q)
        flange, rotation = np.asarray(flange).reshape(3), np.asarray(rotation).reshape(3, 3)
        sign = -1 if side == "left" else 1
        # The vertical envelope contains the sensor housing plus shortened stem.
        specs = [("vertical", [0, 0, .04], [.028, .028, .08]),
                 ("lateral", [0, sign * .065, .08], [.018, .13, .018]),
                 ("manifold", [0, sign * .138, .08], [.07, .014, .07])]
        specs.extend((f"cup_r{row}_c{column}", [x, sign * .15, .08 + z], [.02, .01, .02])
                     for row, z in enumerate((-.024, .024), 1)
                     for column, x in enumerate((-.024, .024), 1))
        tools = [(name, flange + rotation @ local, np.asarray(size) / 2, rotation)
                 for name, local, size in specs]
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(flange)) or not np.all(np.isfinite(rotation)):
            raise ValueError("FK returned non-finite collision geometry")
        return centers, tools

    def check(self, q_by_side, *, cube_center=None, cube_rotation=None, cup_contacts=(),
              margin_m=0.002, contact_tolerance_m=0.003, first_only=False):
        """Return collision issues; only named cup/Cube contacts get tolerance.

        ``cup_contacts`` contains arm names; it never exempts links or supports.
        A strictly positive margin keeps transit paths away from obstacle faces.
        """
        if not q_by_side or set(q_by_side) - (set(self.solvers) & {"left", "right"}):
            raise ValueError("Configurations must name available FR3 solvers")
        if not np.isfinite(margin_m) or not np.isfinite(contact_tolerance_m) or margin_m < 0 or contact_tolerance_m < 0:
            raise ValueError("Collision margins must be nonnegative")
        if set(cup_contacts) - set(q_by_side):
            raise ValueError("Contact arms must have configurations to check")
        obstacles = [*self.static_boxes, *self.other_cubes]
        if cube_center is not None:
            center = np.asarray(cube_center, dtype=float)
            rotation = np.eye(3) if cube_rotation is None else np.asarray(cube_rotation, dtype=float)
            if center.shape != (3,) or rotation.shape != (3, 3) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(rotation)):
                raise ValueError("Cube pose must be finite center and rotation matrix")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
                raise ValueError("Cube rotation must be a proper orthonormal matrix")
            obstacles.append(("Cube", center, np.full(3, self.config["cube_size_m"] / 2), rotation))
        geometry = {side: self.geometry(side, q) for side, q in q_by_side.items()}
        issues = []

        def add(a, b, clearance, kind, threshold=margin_m):
            if clearance < threshold - 1e-10:
                issues.append({"a": a, "b": b, "clearance_m": float(clearance), "kind": kind})
                return first_only
            return False

        if cube_center is not None:
            _, center, half, rotation = obstacles[-1]
            for name, other_center, other_half, other_rotation in (*self.static_boxes, *self.other_cubes):
                clearance = obb_signed_clearance(center, half, rotation, other_center, other_half, other_rotation)
                if add("Cube", name, clearance, "cube_obstacle", -contact_tolerance_m):
                    return issues

        for side, (centers, tools) in geometry.items():
            for name, center, half, rotation in obstacles:
                clearances = _sphere_box_clearances(centers, self.radii, center, half, rotation)
                for index in np.flatnonzero(clearances < margin_m - 1e-10):
                    if add(f"{side}/{self.spheres[index][0]}", name, clearances[index], "link_obstacle"):
                        return issues
                for tool_name, tool_center, tool_half, tool_rotation in tools:
                    clearance = obb_signed_clearance(tool_center, tool_half, tool_rotation, center, half, rotation)
                    threshold = (-contact_tolerance_m if name == "Cube" and side in cup_contacts and tool_name.startswith("cup_")
                                 else margin_m)
                    if add(f"{side}/{tool_name}", name, clearance, "tool_obstacle", threshold):
                        return issues
            a, b = self.self_pairs
            clearance = np.linalg.norm(centers[a] - centers[b], axis=1) - self.radii[a] - self.radii[b]
            for index in np.flatnonzero(clearance < margin_m - 1e-10):
                if add(f"{side}/{self.spheres[a[index]][0]}", f"{side}/{self.spheres[b[index]][0]}",
                       clearance[index], "self"):
                    return issues
            for name, center, half, rotation in tools:
                # Official wrist spheres overlap the tool mounting attachment.
                clearances = _sphere_box_clearances(centers, self.radii, center, half, rotation)
                for index in np.flatnonzero((self.link_indices <= 5) & (clearances < margin_m - 1e-10)):
                    if add(f"{side}/{self.spheres[index][0]}", f"{side}/{name}", clearances[index], "self_tool"):
                        return issues

        sides = list(geometry)
        for i, side in enumerate(sides):
            centers, tools = geometry[side]
            for other in sides[i + 1:]:
                other_centers, other_tools = geometry[other]
                clearances = np.linalg.norm(centers[:, None] - other_centers[None, :], axis=2) - self.radii[:, None] - self.radii[None, :]
                for a, b in np.argwhere(clearances < margin_m - 1e-10):
                    if add(f"{side}/{self.spheres[a][0]}", f"{other}/{self.spheres[b][0]}", clearances[a, b], "interarm"):
                        return issues
                for body_side, body_centers, tool_side, boxes in ((side, centers, other, other_tools),
                                                                (other, other_centers, side, tools)):
                    for name, center, half, rotation in boxes:
                        signed = _sphere_box_clearances(body_centers, self.radii, center, half, rotation)
                        for index in np.flatnonzero(signed < margin_m - 1e-10):
                            if add(f"{body_side}/{self.spheres[index][0]}", f"{tool_side}/{name}", signed[index], "interarm_tool"):
                                return issues
                for name, center, half, rotation in tools:
                    for other_name, other_center, other_half, other_rotation in other_tools:
                        clearance = obb_signed_clearance(center, half, rotation, other_center, other_half, other_rotation)
                        if add(f"{side}/{name}", f"{other}/{other_name}", clearance, "intertool"):
                            return issues
        return issues

    def is_clear(self, q_by_side, **kwargs):
        return not self.check(q_by_side, first_only=True, **kwargs)

    def validate_path(self, samples_by_side, *, cube_centers=None, cube_rotations=None,
                      max_joint_step_rad=0.025, max_cube_step_m=0.005,
                      progress_callback=None, **kwargs):
        """Check actual sampled motion, densifying long joint-space intervals.

        The caller passes the final time-parameterized samples, including held
        arms. This checks interpolation between those samples as well. Cube
        centers/rotations can be constant or one pose per input sample.
        """
        paths = {side: np.asarray(path, dtype=float) for side, path in samples_by_side.items()}
        counts = {len(path) for path in paths.values()}
        if (not paths or len(counts) != 1 or next(iter(counts)) == 0
                or not np.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0
                or not np.isfinite(max_cube_step_m) or max_cube_step_m <= 0):
            raise ValueError("Paths need the same positive sample count and joint step")
        count = next(iter(counts))
        for path in paths.values():
            if path.shape != (count, 7) or not np.all(np.isfinite(path)):
                raise ValueError("FR3 paths must have shape (samples, 7) and finite joints")
        centers = None if cube_centers is None else np.asarray(cube_centers, dtype=float)
        rotations = None if cube_rotations is None else np.asarray(cube_rotations, dtype=float)
        if centers is not None and centers.shape not in ((3,), (count, 3)):
            raise ValueError("cube_centers must be a center or one center per path sample")
        if rotations is not None and rotations.shape not in ((3, 3), (count, 3, 3)):
            raise ValueError("cube_rotations must be a matrix or one matrix per path sample")
        checked = 0
        for sample in range(count):
            previous = max(0, sample - 1)
            steps = max(1, int(np.ceil(max(np.max(np.abs(path[sample] - path[previous]))
                                             for path in paths.values()) / max_joint_step_rad)))
            if centers is not None and centers.ndim == 2:
                steps = max(steps, int(np.ceil(np.linalg.norm(centers[sample] - centers[previous]) / max_cube_step_m)))
            for substep in range(1, steps + 1):
                fraction = substep / steps
                q = {side: path[previous] + fraction * (path[sample] - path[previous]) for side, path in paths.items()}
                center = centers
                if centers is not None and centers.ndim == 2:
                    center = centers[previous] + fraction * (centers[sample] - centers[previous])
                rotation = rotations
                if rotations is not None and rotations.ndim == 3:
                    # Slerp preserves a rotation while densifying pose samples.
                    from scipy.spatial.transform import Rotation, Slerp
                    rotation = Slerp([0, 1], Rotation.from_matrix(rotations[[previous, sample]]))([fraction]).as_matrix()[0]
                issues = self.check(q, cube_center=center, cube_rotation=rotation, first_only=True, **kwargs)
                checked += 1
                if progress_callback is not None and checked % 100 == 0:
                    progress_callback(checked)
                if issues:
                    return {"status": "BLOCKED", "samples_checked": checked, "sample": sample,
                            "segment_fraction": fraction, "issues": issues}
        if progress_callback is not None:
            progress_callback(checked)
        return {"status": "CLEAR", "samples_checked": checked, "issues": []}

    def assert_clear_path(self, samples_by_side, **kwargs):
        report = self.validate_path(samples_by_side, **kwargs)
        if report["status"] != "CLEAR":
            issue = report["issues"][0]
            raise RuntimeError(f"Collision at path sample {report['sample']}: {issue['a']} / {issue['b']} "
                               f"clearance {issue['clearance_m']:.6f} m ({issue['kind']})")
        return report
