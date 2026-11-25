"""Small, dependency-light time parameterization for FR3 joint paths."""
from __future__ import annotations

import numpy as np


def plan_collision_free_path(start, goal, is_clear, lower, upper, *, max_iterations=2500):
    """Deterministic RRT-Connect, checking every joint-space edge at 0.005 rad."""
    start, goal, lower, upper = (np.asarray(q, dtype=float) for q in (start, goal, lower, upper))
    if (start.ndim != 1 or any(q.shape != start.shape for q in (goal, lower, upper))
            or not all(np.all(np.isfinite(q)) for q in (start, goal, lower, upper))
            or np.any(lower >= upper)):
        raise ValueError("Invalid joint vectors or limits")

    def clear(q):
        return bool(np.all(q >= lower) and np.all(q <= upper) and is_clear(q))

    def edge(a, b):
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / 0.005)))
        return all(clear(q) for q in np.linspace(a, b, count + 1))

    if not clear(start) or not clear(goal):
        raise RuntimeError("Motion endpoint is in collision or outside joint limits")
    if edge(start, goal):
        return np.array([start, goal])

    rng = np.random.default_rng(7)
    trees = [([start], [-1]), ([goal], [-1])]

    def extend(tree, target):
        nodes, parents = tree
        nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes) - target, axis=1)))
        delta = target - nodes[nearest]
        q = nodes[nearest] + delta * min(1.0, 0.22 / max(float(np.linalg.norm(delta)), 1e-12))
        if not edge(nodes[nearest], q):
            return None
        nodes.append(q)
        parents.append(nearest)
        return len(nodes) - 1

    def trace(tree, index):
        path = []
        while index >= 0:
            path.append(tree[0][index])
            index = tree[1][index]
        return path[::-1]

    for iteration in range(max_iterations):
        first = iteration % 2
        a, b = trees[first], trees[1 - first]
        target = b[0][0] if iteration % 5 == 0 else rng.uniform(lower, upper)
        ia = extend(a, target)
        if ia is None:
            continue
        while True:
            ib = extend(b, a[0][ia])
            if ib is None:
                break
            if np.allclose(b[0][ib], a[0][ia], atol=1e-10):
                path = trace(a, ia) + trace(b, ib)[-2::-1]
                if first:
                    path.reverse()
                # Long clear edges remove unnecessary stops before retiming.
                result = [path[0]]
                index = 0
                while index < len(path) - 1:
                    following = len(path) - 1
                    while following > index + 1 and not edge(path[index], path[following]):
                        following -= 1
                    result.append(path[following])
                    index = following
                return np.asarray(result)
    raise RuntimeError(f"No collision-free joint path found after {max_iterations} RRT iterations")


def plan_joint_trajectory(
    joint_path,
    *,
    max_joint_speed: float = 0.8,
    min_segment_time_s: float = 0.08,
    min_duration_s: float = 0.0,
    sample_hz: float = 120.0,
    stop_at_waypoints: bool = False,
):
    """Return uniformly sampled cubic-Hermite motion through a joint path.

    The path is time-parameterized by the largest joint displacement in each
    segment.  Centered knot velocities remove the stop/start behavior caused
    by issuing one target and waiting before issuing the next target.
    A minimum duration stretches the schedule without changing the waypoints.
    """
    q = np.asarray(joint_path, dtype=float)
    if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] == 0:
        raise ValueError("joint_path must contain at least two non-empty joint vectors")
    if not np.all(np.isfinite(q)):
        raise ValueError("joint_path contains non-finite values")
    if max_joint_speed <= 0 or min_segment_time_s <= 0 or sample_hz <= 0:
        raise ValueError("trajectory timing parameters must be positive")
    if not np.isfinite(min_duration_s) or min_duration_s < 0.0:
        raise ValueError("minimum trajectory duration must be finite and nonnegative")

    displacement = np.max(np.abs(np.diff(q, axis=0)), axis=1)
    # Cubic blending briefly exceeds the endpoint slope when it ramps a
    # segment in or out.  Leave a 1.5x timing margin so the sampled motion
    # remains below max_joint_speed instead of creating a hidden speed spike.
    durations = np.maximum(float(min_segment_time_s), 1.5 * displacement / float(max_joint_speed))
    durations *= max(1.0, min_duration_s / float(durations.sum()))
    knot_times = np.concatenate(([0.0], np.cumsum(durations)))

    # Zero endpoint velocities prevent an approach/retraction impulse.  The
    # centered estimates keep interior velocity continuous between segments.
    velocities = np.zeros_like(q)
    if len(q) > 2 and not stop_at_waypoints:
        velocities[1:-1] = (q[2:] - q[:-2]) / (knot_times[2:, None] - knot_times[:-2, None])
        velocities[1:-1] = np.clip(velocities[1:-1], -max_joint_speed, max_joint_speed)

    sample_count = max(2, int(np.ceil(knot_times[-1] * sample_hz)) + 1)
    times = np.linspace(0.0, knot_times[-1], sample_count)
    samples = np.empty((sample_count, q.shape[1]), dtype=float)
    for row, t in enumerate(times):
        segment = min(int(np.searchsorted(knot_times, t, side="right") - 1), len(q) - 2)
        t0, t1 = knot_times[segment], knot_times[segment + 1]
        h = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        h = float(np.clip(h, 0.0, 1.0))
        h00 = 2.0 * h**3 - 3.0 * h**2 + 1.0
        h10 = h**3 - 2.0 * h**2 + h
        h01 = -2.0 * h**3 + 3.0 * h**2
        h11 = h**3 - h**2
        dt = t1 - t0
        samples[row] = (
            h00 * q[segment]
            + h10 * dt * velocities[segment]
            + h01 * q[segment + 1]
            + h11 * dt * velocities[segment + 1]
        )
    # Keep the exact endpoints despite floating-point interpolation error.
    samples[0], samples[-1] = q[0], q[-1]
    return times, samples


def point_aabb_signed_clearance(point, center, size):
    """Signed clearance from a point to an axis-aligned box.

    Positive values are outside clearance; a negative value is penetration
    depth.  This is intentionally geometric and does not replace PhysX
    contact reporting.
    """
    point, center, size = (np.asarray(value, dtype=float) for value in (point, center, size))
    delta = np.abs(point - center) - 0.5 * size
    outside = np.maximum(delta, 0.0)
    if np.any(delta > 0.0):
        return float(np.linalg.norm(outside))
    return -float(np.min(-delta))
