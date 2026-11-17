import numpy as np

from fr3_trajectory import plan_joint_trajectory


def test_plan_is_continuous_and_hits_knots():
    path = np.array([[0.0, 0.0], [0.2, -0.1], [0.4, 0.0], [0.7, 0.2]])
    times, samples = plan_joint_trajectory(path, max_joint_speed=0.5)
    assert np.allclose(samples[0], path[0])
    assert np.allclose(samples[-1], path[-1])
    assert np.all(np.isfinite(samples))
    # A sampled trajectory must not introduce a jump larger than the source path.
    assert np.max(np.abs(np.diff(samples, axis=0))) <= 0.5 / 120.0 * 1.2
    assert np.all(np.diff(times) > 0)


def test_point_box_clearance_reports_penetration():
    from fr3_trajectory import point_aabb_signed_clearance
    assert point_aabb_signed_clearance([2, 0, 0], [0, 0, 0], [1, 1, 1]) == 1.5
    assert point_aabb_signed_clearance([0, 0, 0], [0, 0, 0], [1, 1, 1]) == -0.5


def test_planner_routes_around_obstacle_and_refuses_colliding_goal():
    import pytest
    from fr3_trajectory import plan_collision_free_path
    clear = lambda q: np.linalg.norm(q) > 0.35
    path = plan_collision_free_path([-0.8, 0], [0.8, 0], clear, [-1, -1], [1, 1])
    assert len(path) > 2
    for a, b in zip(path, path[1:]):
        assert all(clear(q) for q in np.linspace(a, b, 500))
    with pytest.raises(RuntimeError, match="endpoint"):
        plan_collision_free_path([-0.8, 0], [0, 0], clear, [-1, -1], [1, 1])


def test_safe_corner_retiming_stays_on_checked_straight_edges():
    _, samples = plan_joint_trajectory([[0, 0], [.4, 0], [.4, .4]], stop_at_waypoints=True)
    assert np.all((np.abs(samples[:, 1]) < 1e-12) | (np.abs(samples[:, 0] - .4) < 1e-12))


def test_duplicate_start_holds_without_moving_opposite_to_the_goal():
    # move() already supplies the measured start; the execution layer can
    # prepend that same point. A centered tangent must not bend into contact.
    times, samples = plan_joint_trajectory([[0.0], [0.0], [1.0]], max_joint_speed=0.4)
    assert np.all(samples[times <= 0.08, 0] == 0)
    assert np.min(samples[:, 0]) >= 0
    assert np.min(np.diff(samples[:, 0])) >= 0


def test_duplicate_end_and_interior_knots_preserve_the_stationary_segment():
    for path, level, start in (([[0.], [1.], [1.]], 1., 3.75),
                               ([[0.], [1.], [1.], [2.]], 1., 3.75)):
        times, samples = plan_joint_trajectory(path, max_joint_speed=0.4)
        held = (times >= start) & (times <= start + 0.08)
        np.testing.assert_allclose(samples[held, 0], level, atol=1e-12)
        assert np.min(np.diff(samples[:, 0])) >= -1e-12


def test_all_stationary_knots_keep_the_hold_duration_and_finite_samples():
    times, samples = plan_joint_trajectory([[0.2, -0.4]] * 3)
    assert np.isclose(times[-1], 0.16)
    assert np.all(np.isfinite(samples))
    np.testing.assert_allclose(samples, np.tile([0.2, -0.4], (len(samples), 1)), atol=1e-12)
