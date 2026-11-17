import numpy as np
import pytest

from fr3_contact_control import NormalForceController


def test_seating_uses_rotated_cube_support_and_micrometer_contact_tolerance():
    from fr3_contact_control import oriented_wall_gap

    # Saved pilot1 stopped 15.28 um short at its center but only 6.61 um at
    # its oriented support point; requiring center error <2 um overloaded it.
    quaternion = np.array([1., -4.13594098e-5, 1.23861795e-7, -1.64542689e-5])
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    row_y = [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)]
    gap = oriented_wall_gap(1.0759847164154053, row_y, .075, 1.076, 1)
    assert gap == pytest.approx(6.611e-6, abs=2e-9)
    assert gap < 20e-6
    assert oriented_wall_gap(1.074545741, [0, 1, 0], .075, 1.076, 1) > .001
    assert oriented_wall_gap(-.24498, [1, 0, 0], .075, -.245, -1) == pytest.approx(20e-6)


def test_seating_tolerance_covers_saved_physx_wall_residual():
    from scipy.spatial.transform import Rotation
    from fr3_ab_visual_runner import B2_CONTROLLER_PARAMETERS
    from fr3_contact_control import oriented_wall_gap

    # A-force pilot2 reached 10.158 N while the wall solver retained this gap.
    center = -.24497638642787933
    row_x = Rotation.from_quat([-4.212975923678641e-8, -1.1511872344271978e-7,
                               -1.859634011225353e-7, 1.0]).as_matrix()[0]
    gap = oriented_wall_gap(center, row_x, .075, -.245, -1)
    assert 20e-6 < gap <= B2_CONTROLLER_PARAMETERS["seating_gap_tolerance_m"] <= 30e-6


def test_compression_sign_and_absolute_position():
    controller = NormalForceController(1.001)
    before = controller.position_m
    assert controller.update(0) > before
    before = controller.position_m
    assert controller.update(4) < before
    # A negative compression reading remains unchanged and commands inward.
    before = controller.position_m
    assert controller.update(-0.2) > before


def test_submicrometer_updates_accumulate_without_losing_position_precision():
    controller = NormalForceController(1.001)
    for _ in range(120):
        controller.update(0)
    assert controller.offset_m == pytest.approx(2e-6, abs=1e-13)


def test_slew_and_offset_limits_do_not_wind_up():
    controller = NormalForceController(1.0, gain_m_per_Ns=1,
                                       max_offset_m=1e-4, max_speed_m_s=1e-3)
    first = controller.update(0, dt=.01)
    assert first - 1.0 == pytest.approx(1e-5)
    for _ in range(100):
        controller.update(0, dt=.01)
    assert controller.offset_m == pytest.approx(1e-4)
    # The saturated integrator responds immediately when the force reverses.
    previous = controller.position_m
    assert controller.update(4, dt=.01) < previous
    for _ in range(100):
        controller.update(4, dt=.01)
    assert controller.offset_m == pytest.approx(-1e-4)


@pytest.mark.parametrize("force", [-10.01, 10.01])
def test_overload_is_reported_without_clipping_or_changing_command(force):
    controller = NormalForceController(1.0)
    with pytest.raises(RuntimeError, match="overload"):
        controller.update(force)
    assert controller.position_m == 1.0


@pytest.mark.parametrize("force,dt,target", [(np.nan, 1/120, 2), (0, 0, 2),
                                            (0, np.inf, 2), (0, 1/120, -1),
                                            (0, 1/120, 11)])
def test_invalid_updates_are_rejected_without_mutating_state(force, dt, target):
    controller = NormalForceController(1.0)
    with pytest.raises(ValueError):
        controller.update(force, dt, target)
    assert controller.position_m == 1.0 and controller.target_force_N == 2.0


def test_default_gain_is_stable_for_stiff_contact_then_unloads():
    controller = NormalForceController(1.0)
    forces = []
    for _ in range(1200):
        force = max(0.0, 1e6 * (controller.position_m - 1.0))
        forces.append(force)
        controller.update(force)
    assert forces[-1] == pytest.approx(2.0, abs=1e-3)
    assert max(forces) <= 2.001
    before = controller.position_m
    for _ in range(1200):
        force = max(0.0, 1e6 * (controller.position_m - 1.0))
        controller.update(force, target_force_N=0.0)
    assert controller.position_m < before
    assert force < .001


@pytest.mark.parametrize("stiffness_N_m", [23000.0, 1e6])
def test_tuned_gain_settles_and_unloads_a_delayed_contact_without_force_peaks(stiffness_N_m):
    # Pilot1 identified ~23 kN/m static stiffness. Include the much stiffer
    # transient regime and three ticks of measurement delay, without filtering.
    dt = 1 / 120
    controller = NormalForceController(1.0, gain_m_per_Ns=2e-5)
    delayed_forces = [0.0] * 3
    forces, commands = [], [controller.position_m]
    unload_initial_force = None
    for index in range(25 * 120):
        force = max(0.0, stiffness_N_m * (controller.position_m - 1.0))
        forces.append(force)
        measured = delayed_forces.pop(0)
        delayed_forces.append(force)
        if index < 10 * 120:
            target = 2.0
        else:
            if unload_initial_force is None:
                unload_initial_force = force
            # Match the runner's one-second force-target ramp before zero hold.
            target = unload_initial_force * max(0.0, 1 - (index - 10 * 120) * dt)
        commands.append(controller.update(measured, dt, target_force_N=target))
    forces, commands = np.asarray(forces), np.asarray(commands)
    assert np.all((forces[9 * 120:10 * 120] >= 1.0) & (forces[9 * 120:10 * 120] <= 3.0))
    assert np.max(forces) < 10.0
    assert np.max(forces[-120:]) < 0.1  # At most 15 seconds of unloading.
    assert np.max(np.abs(np.diff(commands))) <= controller.max_speed_m_s * dt + 1e-14
    assert np.max(np.abs(commands - 1.0)) <= controller.max_offset_m + 1e-14
