import numpy as np
import pytest

from fr3_collision import FR3CollisionChecker, obb_signed_clearance


class FakeFK:
    def __init__(self, body=(10., 0., 0.), flange=(20., 0., 0.)):
        self.body, self.flange = np.array(body), np.array(flange)

    def compute_forward_kinematics(self, link, q):
        center = self.flange if link == "fr3_link8" else self.body + [q[0], 0, 0]
        return center, np.eye(3)


def checker(boxes=(), solvers=None, spheres=None):
    return FR3CollisionChecker(
        {"static_boxes": list(boxes), "cube_size_m": .1},
        solvers or {"right": FakeFK()},
        spheres or [{"fr3_link2": [{"center": [0, 0, 0], "radius": .04}]}],
    )


def test_full_arm_rejects_collision_far_from_tcp():
    collision = checker([("wall", [10., 0, 0], [.01, 1, 1])])
    issues = collision.check({"right": np.zeros(7)})
    assert any(issue["a"] == "right/fr3_link2" and issue["b"] == "wall" for issue in issues)


def test_oriented_box_sat_avoids_aabb_false_positive():
    angle = np.pi / 4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    assert obb_signed_clearance([0, 0, 0], [.8, .02, .02], rotation,
                                [0, .2, 0], [.8, .02, .02], rotation) > 0
    assert obb_signed_clearance([0, 0, 0], [.8, .02, .02], rotation,
                                [0, .02, 0], [.8, .02, .02], rotation) < 0


def test_cube_contact_exempts_only_cups_and_limits_penetration():
    collision = checker(solvers={"right": FakeFK(flange=(0., 0., 0.))})
    q = {"right": np.zeros(7)}
    assert not collision.is_clear(q, cube_center=[0, .204, .08])
    assert collision.is_clear(q, cube_center=[0, .204, .08], cup_contacts=("right",))
    assert not collision.is_clear(q, cube_center=[0, .201, .08], cup_contacts=("right",))
    issues = collision.check(q, cube_center=[0, .190, .08], cup_contacts=("right",))
    assert any(issue["a"] == "right/manifold" for issue in issues)


def test_interarm_and_tool_body_collisions():
    collision = checker(solvers={"left": FakeFK(), "right": FakeFK(flange=(30., 0, 0))})
    q = {side: np.zeros(7) for side in ("left", "right")}
    assert any(issue["kind"] == "interarm" for issue in collision.check(q))
    collision = checker(solvers={"left": FakeFK(body=(20., .138, .08), flange=(30., 0, 0)),
                                 "right": FakeFK(body=(10., 0, 0), flange=(20., 0, 0))})
    assert any(issue["kind"] == "interarm_tool" for issue in collision.check(q))
    geometry = {arm: collision.geometry(arm, angles) for arm, angles in q.items()}
    static = [issue for arm in q for issue in collision.check({arm: q[arm]}, geometry={arm: geometry[arm]})]
    assert static + collision.check_interarm(geometry) == collision.check(q)


def test_path_checks_between_clear_endpoints():
    collision = checker([("wall", [10., 0, 0], [.01, 1, 1])])
    path = np.zeros((2, 7))
    path[:, 0] = [-.2, .2]
    assert all(collision.is_clear({"right": q}) for q in path)
    report = collision.validate_path({"right": path})
    assert report["status"] == "BLOCKED" and report["sample"] == 1
    with pytest.raises(RuntimeError, match="right/fr3_link2 / wall"):
        collision.assert_clear_path({"right": path})


def test_parallel_checks_match_serial_at_an_interpolated_partition_boundary(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import fr3_parallel_collision as parallel

    collision = checker([("wall", [10., 0, 0], [.01, 1, 1])])
    monkeypatch.setattr(parallel, "_checker", collision, raising=False)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor",
                        lambda max_workers, **_: ThreadPoolExecutor(max_workers=max_workers))
    path = np.zeros((4, 7))
    path[:, 0] = [-.3, -.2, .2, .3]
    paths = {"right": path}
    expected = collision.validate_path(paths)
    validator = parallel.ParallelPathValidator(2, {}, {})
    try:
        actual = validator.validate(paths, bases={}, neighbors=[], centers=None, rotations=None,
                                    contacts=(), progress=lambda done, total: None)
        for malformed, centers in [({"right": path[:0]}, None),
                                   ({"right": path, "left": path[:3]}, None),
                                   (paths, np.zeros((5, 3)))]:
            with pytest.raises(ValueError):
                validator.validate(malformed, bases={}, neighbors=[], centers=centers, rotations=None,
                                   contacts=(), progress=lambda done, total: None)
    finally:
        validator.close()
    assert expected["status"] == "BLOCKED" and expected["sample"] == 2
    assert actual == expected


def test_path_progress_callback_reports_checked_samples():
    collision = checker()
    progress = []
    report = collision.validate_path({"right": np.zeros((2, 7))}, progress_callback=progress.append)
    assert report["status"] == "CLEAR"
    assert progress[-1] == report["samples_checked"]


def test_carried_cube_cannot_pass_through_wall():
    collision = checker([("wall", [0, 0, 0], [.01, 1, 1])])
    path = np.zeros((2, 7))
    report = collision.validate_path({"right": path}, cube_centers=[[-.2, 0, 0], [.2, 0, 0]])
    assert report["status"] == "BLOCKED"
    assert report["issues"][0]["kind"] == "cube_obstacle"
    # A cube may rest on the support; shallow solver contact remains allowed.
    assert collision.is_clear({"right": path[0]}, cube_center=[.054, 0, 0])


def test_nonadjacent_self_collision_is_rejected():
    spheres = [{f"fr3_link{link}": [{"center": [0, 0, 0], "radius": .04}]}
               for link in (1, 3)]
    collision = checker(spheres=spheres)
    assert any(issue["kind"] == "self" for issue in collision.check({"right": np.zeros(7)}))


def test_invalid_inputs_fail_closed():
    collision = checker()
    with pytest.raises(ValueError):
        collision.check({"right": np.full(7, np.nan)})
    with pytest.raises(ValueError):
        collision.validate_path({"right": np.zeros((2, 6))})
    with pytest.raises(ValueError):
        collision.check({"right": np.zeros(7)}, cup_contacts=("left",))
    with pytest.raises(ValueError):
        collision.check({"right": np.zeros(7)}, margin_m=float("nan"))
    with pytest.raises(ValueError):
        collision.check({"right": np.zeros(7)}, cube_center=[0, 0, 0], cube_rotation=np.zeros((3, 3)))


def test_placed_cubes_block_tools_and_carried_cube_paths():
    collision = checker(solvers={"right": FakeFK(flange=(0., 0., 0.))})
    collision.set_other_cubes([("Cube_1", [0., .204, .08], np.eye(3))])
    assert not collision.is_clear({"right": np.zeros(7)}, cup_contacts=("right",))
    collision = checker()
    collision.set_other_cubes([("Cube_1", [0., 0., 0.], np.eye(3))])
    result = collision.validate_path({"right": np.zeros((2, 7))},
                                     cube_centers=[[-.2, 0., 0.], [.2, 0., 0.]])
    assert result["status"] == "BLOCKED"
    assert result["issues"][0]["kind"] == "cube_obstacle"
    assert result["issues"][0]["b"] == "Cube_1"
    with pytest.raises(ValueError):
        collision.set_other_cubes([("Cube_1", [0., 0., 0.], np.zeros((3, 3)))])
    assert len(collision.other_cubes) == 1


def test_partial_visual_refresh_retains_occluded_collision_obstacles():
    collision = checker()
    collision.set_other_cubes([("Cube_1", [0., 0., 0.], np.eye(3)),
                               ("Cube_2", [1., 0., 0.], np.eye(3))])
    collision.set_other_cubes([("Cube_2", [2., 0., 0.], np.eye(3))], preserve_unobserved=True)
    q = {"right": np.zeros(7)}
    issues = collision.check(q, cube_center=[0., 0., 0.])
    assert any(issue["b"] == "Cube_1" for issue in issues)
    assert collision.is_clear(q, cube_center=[1., 0., 0.])
    assert not collision.is_clear(q, cube_center=[2., 0., 0.])


@pytest.mark.parametrize("phase, age, normal, fresh_x, wrong_cube, passes", [
    ("B2_push_seat", .10, 4.7, .098, False, True),
    ("B2_push_seat", .10, 4.7, .0968, False, False),
    ("B2_push_seat", .10, 4.7, .098, True, False),
    ("B2_push_seat", .21, 4.7, .098, False, False),
    ("B2_push_seat", .10, .1, .098, False, False),
    ("B2_push_seat", .10, 10.1, .098, False, False),
    ("B2_push_retreat", .10, 4.7, .098, False, False),
    ("B2_push_unload", .10, .1, .098, False, True),
])
def test_runtime_seating_reconciles_only_fresh_rgb_and_loaded_fk(phase, age, normal, fresh_x, wrong_cube, passes):
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    collision = checker()
    observed = [.0968, 0., 0.]
    neighbors = [("Cube_1", observed, np.eye(3))]
    if wrong_cube:
        neighbors.append(("Cube_2", [-.0968, 0., 0.], np.eye(3)))
    collision.set_other_cubes(neighbors)
    tree = ast.parse((Path(__file__).resolve().parents[1] / "fr3_ab_visual_runner.py").read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "check_current_collision")
    env = {"np": np, "checker": collision, "joint_state": lambda: {"right": np.zeros(7)},
           "sensed_cube": lambda: (np.zeros(3), np.eye(3)), "allowed_contacts": {"right"},
           "current_task": {"cube_id": 3, "contact_target": "Cube_1"}, "world": SimpleNamespace(current_time=5.),
           "push_sign": -1, "push_side": "right", "last_force_sample": {"right": {"tcp_gravity_corrected_world": [normal, 0., 0.]}},
           "live_neighbors": {"Cube_1": (np.array([fresh_x, 0., 0.]), np.eye(3), 5.-age)},
           "report": {"neighbor_observations": [{"neighbors": [{"name": "Cube_1", "position_m": observed, "capture_time_s": 0.}]}]}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<runtime geometry guard>", "exec"), env)
    if passes:
        env["check_current_collision"](phase)
        assert env["report"]["neighbor_geometry_rechecks"]
    else:
        with pytest.raises(RuntimeError, match="Measured collision"):
            env["check_current_collision"](phase)
