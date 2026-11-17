import numpy as np
import pytest

from fr3_wrench import gravity_wrench_local, transform_wrench, wrench_at_origin


def test_offset_point_force_has_known_moment_and_returns_to_zero_at_application():
    # A downward 10 N force 20 cm along +X gives +2 N m about Y at the origin.
    at_origin = wrench_at_origin([0, 0, -10], [0, 0, 0], [-0.2, 0, 0])
    np.testing.assert_allclose(at_origin, [0, 0, -10, 0, 2, 0])
    np.testing.assert_allclose(
        wrench_at_origin(at_origin[:3], at_origin[3:], [0.2, 0, 0]),
        [0, 0, -10, 0, 0, 0],
    )


@pytest.mark.parametrize("axis", range(6))
@pytest.mark.parametrize("sign", (-1, 1))
def test_reexpress_all_six_independent_load_axes(axis, sign):
    # A quarter turn about +Z maps +X to +Y and +Y to -X, including moments.
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    old = np.eye(6)[axis] * sign
    expected = np.zeros(6)
    target_axis, direction = ((1, 1), (0, -1), (2, 1))[axis % 3]
    expected[target_axis + (3 if axis >= 3 else 0)] = direction * sign
    np.testing.assert_allclose(transform_wrench(old, rotation, [0, 0, 0]), expected)


def test_rotation_and_moment_shift_use_new_frame_axes():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # Old +X force becomes new +Y. At new +X position it adds +Z moment.
    actual = transform_wrench([2, 0, 0, 0, 1, 0], rotation, [0.3, 0, 0])
    np.testing.assert_allclose(actual, [0, 2, 0, -1, 0, 0.6])


def test_gravity_uses_sensor_orientation_and_tool_center_of_mass():
    np.testing.assert_allclose(
        gravity_wrench_local(np.eye(3), 2, [0.1, 0, 0], [0, 0, -9.81]),
        [0, 0, -19.62, 0, 1.962, 0],
    )
    # Sensor +X points world +Z, so gravity is along sensor -X.
    rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(
        gravity_wrench_local(rotation, 2, [0, 0.1, 0], [0, 0, -9.81]),
        [-19.62, 0, 0, 0, 0, 1.962],
    )


def test_incoming_reaction_shift_and_explicit_sign_can_cancel_static_tool_gravity():
    com = np.array([0.03, 0.10, 0.08])
    # Hypothetical parent-on-tool reaction at the tool COM; actual sensor sign
    # must first be established by a physics calibration, not this math test.
    parent_on_tool_at_com = np.array([0, 0, 4.905, 0, 0, 0])
    parent_on_tool_at_sensor = wrench_at_origin(
        parent_on_tool_at_com[:3], parent_on_tool_at_com[3:], -com)
    tool_on_sensor = -parent_on_tool_at_sensor
    corrected = tool_on_sensor - gravity_wrench_local(np.eye(3), 0.5, com, [0, 0, -9.81])
    np.testing.assert_allclose(corrected, np.zeros(6), atol=1e-12)


def test_invalid_frames_and_nonfinite_loads_are_rejected():
    with pytest.raises(ValueError, match="orthonormal"):
        gravity_wrench_local(np.diag([1, 1, -1]), 1, [0, 0, 0], [0, 0, -9.81])
    with pytest.raises(ValueError, match="mass"):
        gravity_wrench_local(np.eye(3), -1, [0, 0, 0], [0, 0, -9.81])
    with pytest.raises(ValueError, match="finite"):
        wrench_at_origin([0, np.nan, 0], [0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match="shape"):
        transform_wrench([0, 0, 0], np.eye(3), [0, 0, 0])
