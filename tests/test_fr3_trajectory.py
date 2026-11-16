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
