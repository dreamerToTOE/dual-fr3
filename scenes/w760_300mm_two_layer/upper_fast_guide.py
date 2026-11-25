"""Use the existing native Lula worker pool for per-tick moving-rail collision checks."""
from concurrent.futures import wait

import numpy as np

from source_hooks import replace_once


def _rail_interval(paths, rails, origins, neighbors, centers, rotation, contacts, first, margin_m):
    from fr3_parallel_collision import _checker

    _checker.other_cubes = neighbors
    for side, (position, orientation) in origins.items():
        solver = _checker.solvers[side]
        solver.position = position.copy()
        solver.rotation = orientation
    for local, center in enumerate(centers):
        for side, solver in _checker.solvers.items():
            solver.position[1] = rails[side][local]
        issues = _checker.check({side: path[local] for side, path in paths.items()},
                                cube_center=center, cube_rotation=rotation, cup_contacts=contacts,
                                margin_m=margin_m, first_only=True)
        if issues:
            return first + local, issues[0]
    return None


def check_rail_path(validator, samples, rails, origins, neighbors, centers, rotation, contacts, progress,
                    *, include_start=False, margin_m=.001):
    frames = len(centers) - 1
    if frames < 1 or any(len(a) != frames + 1 for a in (*samples.values(), *rails.values())):
        raise ValueError("Rail, joint and payload guides must have identical sample counts")
    paths = {side: np.asarray(q) for side, q in samples.items()}
    jobs = []
    indices = np.arange(0 if include_start else 1, frames + 1)
    for interval in np.array_split(indices, min(len(indices), validator.workers * 2)):
        first = int(interval[0])
        rows = slice(first, int(interval[-1]) + 1)
        jobs.append(validator.pool.submit(
            _rail_interval, {side: path[rows] for side, path in paths.items()},
            {side: rail[rows] for side, rail in rails.items()}, origins, neighbors, centers[rows],
            rotation, contacts, first, margin_m))
    pending = set(jobs)
    while pending:
        _, pending = wait(pending, timeout=.05)
        progress()
    for job in jobs:
        collision = job.result()
        if collision is not None:
            return collision
    return None


def check_empty_rail(validator, side, states, bases, rail_states, obstacles, center, rotation, contacts, progress):
    # Both joints and payload stay fixed, only the named base translates on Y.
    ys = np.asarray(bases)[:, 1]
    paths = {arm: np.broadcast_to(q, (len(ys), len(q))) for arm, q in states.items()}
    rails = {arm: ys if arm == side else np.full_like(ys, base[1]) for arm, base in rail_states.items()}
    origins = {}
    for arm, base in rail_states.items():
        c, s = np.cos(base[3]), np.sin(base[3])
        origins[arm] = (base[:3], np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]))
    return check_rail_path(validator, paths, rails, origins, obstacles,
                           np.broadcast_to(center, (len(ys), 3)), rotation, tuple(contacts), progress,
                           include_start=True, margin_m=.002)


def cruise_progress(t, duration, ramp_s):
    """Integral of a quintic velocity ramp, including a constant-speed middle."""
    if not 0 < ramp_s < duration / 2:
        raise ValueError("The move needs two ramps and a positive cruising interval")
    seconds = t * duration
    edge = np.minimum(seconds, duration - seconds)
    x = np.clip(edge / ramp_s, 0., 1.)
    distance = np.where(edge < ramp_s, ramp_s * x**4 * (2.5 - 3*x + x**2),
                        edge - ramp_s / 2)
    u = distance / (duration - ramp_s)
    return np.where(t <= .5, u, 1 - u)


def inject(source):
    source = replace_once(source, '            guides_start = {side: tcp_fk(side, guide_q[side][0])[0] for side in robots}',
                          '''            guides_start = {side: tcp_fk(side, guide_q[side][0])[0] for side in robots}
            if not carrying:
                # The release pose differs slightly from the guide in orientation as well as position.
                release_rotation_offsets = {
                    side: Rotation.from_matrix(actual_tool[side][1] @ tcp_fk(side, guide_q[side][0])[1].T).as_rotvec()
                    for side in robots}
'''.rstrip())
    source = replace_once(source,
                          '                                target = guided_tcp + (1. - u[index]) * (measured_tcp - guides_start[side])',
                          '''                                target = guided_tcp + (1. - u[index]) * (measured_tcp - guides_start[side])
                                rotation = (Rotation.from_rotvec((1. - u[index]) * release_rotation_offsets[side])
                                            .as_matrix() @ rotation)''')
    source = replace_once(source, '            if not replay_rail:\n                try:\n                    for base in bases:',
                          '''            rail_check_started = time.perf_counter()
            use_pool = (not replay_rail and parallel_validator is not None and count >= 120
                        and np.array_equal(original[[0, 2, 3]], destination[[0, 2, 3]]))
            if use_pool:
                from upper_fast_guide import check_empty_rail
                blocked = check_empty_rail(parallel_validator, side, states, bases, rail_states,
                    checker.other_cubes, center, rotation, allowed_contacts,
                    lambda: planning_update(f"{side}: rail collision checking"))
                if blocked is not None:
                    raise RuntimeError(f"Rail path is in collision: {blocked[1]} (frame {blocked[0]})")
            elif not replay_rail:
                try:
                    for base in bases:''')
    source = replace_once(source, '            for base in bases:\n                set_robot_base_pose(side, base)',
                          '''            report.setdefault("empty_rail_checks", []).append({
                "cube_id": current_task["cube_id"], "side": side, "samples": count,
                "collision_workers": parallel_validator.workers if use_pool else 1,
                "cache_replay": replay_rail, "planning_wall_s": time.perf_counter() - rail_check_started})
            for base in bases:
                set_robot_base_pose(side, base)''')
    source = replace_once(source, '            for _ in range(3):\n                frames = max(120,',
                          '''            plan_started = time.perf_counter()
            ik_seconds = collision_seconds = 0.
            from upper_fast_guide import check_rail_path, cruise_progress
            cruise = carrying and phase["phase"] == "loaded_rail_transfer" and duration >= 40.
            if cruise:
                # Each axis cruises under the original limits; ease the sealed payload over 8 s at either end.
                rail_slope = max(np.max(np.abs(np.diff(y))) for y in rails.values()) * (count - 1)
                joint_slope = max(np.max(np.abs(np.diff(q, axis=0))) for q in guide_q.values()) * (count - 1)
                load_slope = np.max(np.linalg.norm(np.diff(centers, axis=0), axis=1)) * (count - 1)
                duration = 8. + max(rail_slope / .025, joint_slope / .30, load_slope / .06)
            # Retain the measured slow withdrawal used for upper slot 2.
            if not carrying and current_task.get("slot") == 2:
                duration = max(duration, 25.0)
            for _ in range(10):
                frames = max(120,''')
    source = replace_once(source, '                u = (10 - 15 * t + 6 * t * t) * t ** 3',
                          '                u = cruise_progress(t, duration, 8.) if cruise else (10 - 15 * t + 6 * t * t) * t ** 3')
    source = replace_once(source, '                samples = {side: [initial[side]] for side in robots}',
                          '''                samples = {side: [initial[side]] for side in robots}
                parallel = parallel_validator is not None and frames >= 240
                ik_started = time.perf_counter()''')
    start = source.index('                        issues = checker.check({side: q[-1]')
    end = source.index('                finally:', start)
    serial_check = source[start:end]
    source = replace_once(source, serial_check, '                        if not parallel:\n' +
        ''.join('    ' + line for line in serial_check.splitlines(keepends=True)))
    source = replace_once(source, '                speed = max(float(np.max(np.abs(np.diff(q, axis=0)))) * 120.',
                          '''                ik_seconds += time.perf_counter() - ik_started
                if parallel:
                    collision_started = time.perf_counter()
                    origins = {side: (base[:3], quats_to_rot_matrices(np.array([
                        math.cos(base[3] / 2), 0., 0., math.sin(base[3] / 2)])))
                        for side, base in original.items()}
                    blocked = check_rail_path(parallel_validator, samples, bases, origins,
                        checker.other_cubes, payload_path, payload_rotation, tuple(allowed_contacts),
                        lambda: planning_update(f"{name}: checking moving-rail collisions"))
                    collision_seconds += time.perf_counter() - collision_started
                    if blocked is not None:
                        raise RuntimeError(f"{name} guided collision at frame {blocked[0]}: {blocked[1]}")
                speed = max(float(np.max(np.abs(np.diff(q, axis=0)))) * 120.''')
    source = replace_once(source, '                "carrying": carrying,\n            })',
                          '''                "carrying": carrying, "planning_wall_s": time.perf_counter() - plan_started,
                "collision_workers": parallel_validator.workers if parallel else 1,
                "profile": "8s_quintic_velocity_ramps" if cruise else "quintic_position",
                "duration_sim_s": frames / 120.,
                "ik_loop_wall_s": ik_seconds, "parallel_collision_wall_s": collision_seconds,
            })
            execution_started = time.perf_counter()
            motion_timing = report["side_transfer_motion"][-1]''')
    return replace_once(source, '            record(name, "CLEAR_PHYSX_EXECUTION")',
                        '''            motion_timing["execution_wall_s"] = time.perf_counter() - execution_started
            record(name, "CLEAR_PHYSX_EXECUTION")''')
