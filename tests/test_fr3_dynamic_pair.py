import numpy as np

from fr3_dynamic_pair import following_target, run_independent_pair


def simulate(routes, clear, replan, *, left_delay_steps=0):
    actual = {arm: path[0].copy() for arm, path in routes.items()}
    target = actual.copy()
    events = []
    steps = 0

    def observe():
        return actual.copy(), {arm: np.zeros_like(q) for arm, q in actual.items()}

    def advance():
        nonlocal steps
        steps += 1
        for arm, q in target.items():
            if arm != "left" or steps > left_delay_steps:
                actual[arm] = q
        assert clear(actual)

    def announce(status, **details):
        events.append((status, details))

    result = run_independent_pair(routes, observe, clear, target.__setitem__, advance, replan, announce)
    assert all(np.array_equal(actual[arm], path[-1]) for arm, path in routes.items())
    return result, events


def test_tracking_delay_does_not_force_the_other_arm_to_stop():
    route = np.linspace(0., .06, 15)[:, None]

    def no_replan(*args):
        raise AssertionError("Tracking caught up without a new route")

    result, events = simulate(dict.fromkeys(("left", "right"), route), lambda _: True, no_replan,
                              left_delay_steps=6)
    assert result["overlap_steps"] > 0 and result["tracking_wait_steps"] > 0
    assert any(status == "TRACKING_WAIT" and info["advancing"] == ["right"] for status, info in events)


def test_measured_velocity_distinguishes_following_and_stalled_joints():
    measured = np.zeros(7)
    target = np.full(7, .035)
    assert following_target(measured, np.full(7, .4), target)
    assert not following_target(measured, np.full(7, .048), target)
    assert not following_target(measured, np.full(7, -.4), target)


def test_conflicting_progress_yields_then_resumes():
    route = np.linspace(0., .04, 11)[:, None]

    def clear(states):
        return not all(.009 < q[0] < .031 for q in states.values())

    def no_replan(*args):
        raise AssertionError("The other arm can wait at the crossing")

    result, events = simulate(dict.fromkeys(("left", "right"), route), clear, no_replan)
    assert result["yield_steps"] > 0 and result["overlap_steps"] > 0
    assert any(status == "YIELD" and info["advancing"] == ["left"] for status, info in events)


def test_blocked_pair_replans_from_the_measured_hold():
    line = np.column_stack((np.linspace(0., .04, 11), np.zeros(11)))

    def clear(states):
        left, right = states.values()
        def blocked(q, other):
            return .009 < q[0] < .031 and q[1] < .01 and other[1] < .01
        return not blocked(left, right) and not blocked(right, left)

    def replan(actual, goals):
        assert all(np.allclose(q, [.008, 0.]) for q in actual.values())
        return {arm: np.vstack((q, [q[0], .02], [goals[arm][0], .02], goals[arm]))
                for arm, q in actual.items()}

    result, events = simulate(dict.fromkeys(("left", "right"), line), clear, replan)
    assert result["replans"] == 1
    assert any(status == "REPLANNING" for status, _ in events)
