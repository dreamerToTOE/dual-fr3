"""Execute independent empty-arm routes using current joint feedback."""

from itertools import product
import math

import numpy as np

from fr3_trajectory import plan_collision_free_path, plan_joint_trajectory


def arm_route(start, goal, limits, clear):
    """Plan in seven joint dimensions and preserve the checked spatial edges."""
    if np.max(np.abs(start - goal)) < 1e-6:
        if not clear(goal):
            raise RuntimeError("Parked arm is outside the planned clearance")
        return np.asarray([goal])
    knots = plan_collision_free_path(start, goal, clear, *limits)
    return plan_joint_trajectory(knots, max_joint_speed=.4, stop_at_waypoints=True)[1]


def pair_step_is_clear(measured, forecast, targets, clear):
    """Check the next swept commands at independently advancing arm fractions."""
    arms = tuple(measured)
    sweeps = []
    for arm in arms:
        q = measured[arm]
        span = np.max(np.abs(targets[arm] - q))
        count = max(1, math.ceil(span / .005))
        track = list(np.linspace(q, targets[arm], count + 1)) if span > 1e-9 else [q]
        if np.max(np.abs(forecast[arm] - q)) > 1e-7:
            track.append(forecast[arm])
        sweeps.append(track)
    return all(clear(dict(zip(arms, pair))) for pair in product(*sweeps))


def following_target(measured, velocity, target):
    error = target - measured
    # Allow ordinary servo lag only on joints measured moving toward their target.
    following_speed = np.clip(np.sign(error) * velocity, 0., .4)
    return np.max(np.abs(error) - .08 * following_speed) < .012


def run_independent_pair(routes, observe, clear, command, step, replan, announce):
    """Yield using the measured pair, or replan at a stationary conflict."""
    arms = tuple(routes)
    goals = {arm: routes[arm][-1].copy() for arm in arms}
    cursor = dict.fromkeys(arms, 0)
    held = {arm: routes[arm][0].copy() for arm in arms}
    preferred = arms[0]
    wait = 0
    last = None
    replans = 0
    tick = 0
    last_log = -120
    stats = dict(overlap_steps=0, yield_steps=0, tracking_wait_steps=0, replans=0)
    while True:
        actual, velocities = observe()
        forecast = {arm: actual[arm] + velocities[arm] / 120 for arm in arms}
        pending = tuple(arm for arm in arms if cursor[arm] + 1 < len(routes[arm]))
        if not pending and all(np.max(np.abs(actual[arm] - goals[arm])) < .005 for arm in arms):
            return stats
        ready = tuple(arm for arm in pending if following_target(actual[arm], velocities[arm], held[arm]))
        next_q = {arm: routes[arm][cursor[arm] + 1] for arm in ready}
        if len(ready) == 2:
            alternate = next(arm for arm in ready if arm != preferred)
            choices = (ready, (preferred,), (alternate,), ())
        else:
            choices = (ready, ())
        checked = {}

        def clear_once(states):
            key = tuple(states[arm].tobytes() for arm in arms)
            if key not in checked:
                checked[key] = clear(states)
            return checked[key]

        for advancing in choices:
            proposal = {**held, **{arm: next_q[arm] for arm in advancing}}
            if pair_step_is_clear(actual, forecast, proposal, clear_once):
                break
        else:
            raise RuntimeError("Dynamic pair has no collision-checked hold from measured joints")

        if pending and len(advancing) < len(pending):
            reason = "TRACKING_WAIT" if advancing == ready else "YIELD"
            if reason != last and tick - last_log >= 30:
                announce(reason, advancing=list(advancing), waiting=[arm for arm in pending if arm not in advancing])
                last_log = tick
                last = reason
            stats["tracking_wait_steps" if reason == "TRACKING_WAIT" else "yield_steps"] += 1
        elif last is not None and (tick - last_log >= 30 or not pending):
            announce("RESUMED", advancing=list(advancing))
            last = None
            last_log = tick
        if len(advancing) == 2:
            stats["overlap_steps"] += 1
        held = proposal
        for arm in advancing:
            cursor[arm] += 1
        for arm in arms:
            command(arm, held[arm])
        step()
        tick += 1
        if advancing:
            wait = 0
            if len(advancing) == 1:
                preferred = advancing[0]
            continue
        wait += 1
        if wait < 120:
            continue
        if not pending:
            raise RuntimeError("Dynamic pair joints did not settle at their targets")
        if replans == 2:
            raise RuntimeError("Dynamic pair cannot advance after replanning from measured joints")
        actual, _ = observe()
        announce("REPLANNING", goals=list(pending))
        routes = replan(actual, goals)
        cursor = dict.fromkeys(arms, 0)
        replans += 1
        stats["replans"] = replans
        wait = 0
