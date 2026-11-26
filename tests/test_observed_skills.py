import sys
from pathlib import Path

import numpy as np

SCENES = Path(__file__).resolve().parents[1] / "scenes"
sys.path.insert(0, str(SCENES / "w760_300mm_two_layer"))

from markerless_rows import build, rear_depth_only
from dynamic_withdraw import install as dynamic_withdraw
from front_row_a import floor_guide
from markerless_tracking import install as markerless
from observed_b1 import install as observed_b1
from observed_rear import install as observed_rear
from observed_transport import install as observed_transport, register_transport


def test_registered_carry_connects_the_observed_endpoints():
    guide = floor_guide(2)
    pickup = np.array([.004, -.306, .787])
    landing = np.asarray(guide["landing_m"]) + [.004, .010, 0]
    live = register_transport(guide, pickup, landing)
    phases = [phase for phase in live["phases"] if phase["carrying"]]
    np.testing.assert_allclose(phases[0]["cube_centers_m"][0], pickup)
    np.testing.assert_allclose(phases[-1]["cube_centers_m"][-1], landing)
    for first, second in zip(phases, phases[1:]):
        np.testing.assert_allclose(first["cube_centers_m"][-1], second["cube_centers_m"][0])
        for arm, rail in first["rail_paths_m"].items():
            assert np.isclose(rail[-1], second["rail_paths_m"][arm][0])
    assert guide["landing_m"] != live["landing_m"]


def test_full_runner_assembles_with_observed_skills():
    generated = markerless(rear_depth_only(build()))
    generated = observed_rear(dynamic_withdraw(observed_b1(observed_transport(generated))))
    code = compile(generated, "<observed skills>", "exec")
    assert {"START_REAR", "STOP_AFTER_REAR"} <= set(code.co_names) | set(
        name for fn in code.co_consts if hasattr(fn, "co_names") for name in fn.co_names)
    assert "def move_to_observed_precontact(" in generated
    assert "def dynamic_withdraw(" in generated
