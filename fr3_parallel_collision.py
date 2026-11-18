"""Check paired trajectory intervals in independent native Lula processes."""
from concurrent.futures import ProcessPoolExecutor, wait
import multiprocessing

import numpy as np

from fr3_collision import FR3CollisionChecker, path_arrays


class _WorldFK:
    def __init__(self, kinematics):
        self.kinematics = kinematics

    def compute_forward_kinematics(self, link, q):
        pose = self.kinematics.pose(q[:, None], link)
        return self.rotation @ pose.translation + self.position, self.rotation @ pose.rotation.matrix()


def _init_worker(scene, robot_config):
    import lula
    import yaml
    from pathlib import Path

    global _checker
    description = robot_config["robot_description_path"]
    model = lula.load_robot(description, robot_config["urdf_path"])
    solvers = {side: _WorldFK(model.kinematics()) for side in scene["fr3"]["bases"]}
    spheres = yaml.safe_load(Path(description).read_text())["collision_spheres"]
    _checker = FR3CollisionChecker(scene, solvers, spheres)


def _check_interval(paths, bases, neighbors, centers, rotations, contacts, offset):
    for side, (position, rotation) in bases.items():
        solver = _checker.solvers[side]
        solver.position, solver.rotation = position, rotation
    _checker.other_cubes = neighbors
    return _checker.validate_path(
        paths, cube_centers=centers, cube_rotations=rotations, cup_contacts=contacts,
        start_sample=int(offset != 0),
    )


class ParallelPathValidator:
    def __init__(self, workers, scene, robot_config):
        self.workers = workers
        self.pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker, initargs=(scene, robot_config),
        )

    def validate(self, paths, *, bases, neighbors, centers, rotations, contacts, progress):
        paths, count, centers, rotations = path_arrays(paths, centers, rotations)
        boundaries = np.linspace(0, count, min(count, self.workers) + 1, dtype=int)
        jobs = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # Include the preceding pose so the worker checks the boundary's full interpolated edge.
            offset = max(0, int(start) - 1)
            rows = slice(offset, end)
            future = self.pool.submit(
                _check_interval, {side: path[rows] for side, path in paths.items()}, bases, neighbors,
                centers[rows] if np.ndim(centers) == 2 else centers,
                rotations[rows] if np.ndim(rotations) == 3 else rotations,
                contacts, int(start),
            )
            jobs.append((offset, future))
        pending = {future for _, future in jobs}
        while pending:
            _, pending = wait(pending, timeout=.05)
            progress(len(jobs) - len(pending), len(jobs))
        checked = 0
        for offset, future in jobs:
            result = future.result()
            checked += result["samples_checked"]
            if result["status"] != "CLEAR":
                return {**result, "samples_checked": checked, "sample": result["sample"] + offset}
        return {"status": "CLEAR", "samples_checked": checked, "issues": []}

    def close(self):
        self.pool.shutdown(cancel_futures=True)
