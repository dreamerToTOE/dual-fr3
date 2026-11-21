import numpy as np

from test_u4_contact_observer import front_sample
from u4_contact_observer import SIDE
from u4_side_probe_observer import PulseObserver


def run_probe(force, rear=False):
    observer = PulseObserver()
    passive = {"state": "UNKNOWN", "side_loaded": None, "rear_loaded": rear,
               "camera_age_s": .05, "side_gap_m": 0., "boundaries": []}
    trace = []
    tick = 0
    for segment, values in (("pre", [0.]*12), ("out_neg", np.linspace(0, -.00025, 36)),
                            ("hold_neg", [-.00025]*12), ("cross_pos", np.linspace(-.00025, .00025, 60)),
                            ("hold_pos", [.00025]*12), ("return", np.linspace(.00025, 0, 36)),
                            ("post", [0.]*24)):
        for dx in values:
            tick += 1
            sensor = front_sample(tick, lateral_force=-force(dx))
            sensor.update(physics_step=tick)
            sensor["right_wrench_world"][1] = -1.
            sensor["tcp_positions_m"]["right"][0] = dx
            trace.append(observer.update(sensor, passive, {"segment": segment, "physics_step": tick,
                                                           "command_dx_m": dx}))
    return trace, sensor, passive, observer


def test_preload_recovered_before_positive_motion():
    trace, sensor, passive, observer = run_probe(lambda x: 1000*x if x < 0 else 2000*x)
    assert all(r["probe_response"] is None and not r["probe_applied"] for r in trace[:179])
    assert trace[179]["probe_before_loaded"] is True
    assert trace[-1]["state"] == "S5" and trace[-1]["boundaries"] == [SIDE]
    sensor["physics_step"] += 1
    sensor["time_s"] += 1 / 120
    sensor["tcp_positions_m"]["right"][1] += .0002
    assert not observer.update(sensor, passive)["probe_applied"]


def test_new_contact_and_symmetric_friction():
    trace, *_ = run_probe(lambda x: max(0., x - .00008)*2500)
    assert trace[179]["probe_before_loaded"] is False
    assert trace[-1]["state"] == "S1"
    trace, *_ = run_probe(lambda x: 1600*x, rear=True)
    assert trace[179]["probe_before_loaded"] is None
    assert not trace[-1]["probe_applied"]
