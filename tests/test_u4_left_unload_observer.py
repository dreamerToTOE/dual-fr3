import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_left_unload_observer import INTERCEPT, WEIGHTS, LeftUnloadObserver


def stroke(right_shear, gap=-.003):
    passive = dict(state="UNKNOWN", boundaries=[], side_loaded=None, rear_loaded=False,
                   camera_age_s=.08, side_gap_m=gap)
    observer = LeftUnloadObserver()
    trace = []
    for i in range(80):
        force = max(0., 5.-max(0, i-8)*.10)
        # 5 µm of tool recoil per newton, then actual motion away once unloaded.
        x = (force-5)*.000005 - max(0, i-58)*.000012
        sensor = dict(time_s=i/120, tcp_positions_m={"left":[x,0.,0.],"right":[0.,0.,0.]},
                      cup_normals_world={"left":[1.,0.,0.],"right":[0.,1.,0.]},
                      left_wrench_world=[-force,0.,0.,0.,0.,0.],
                      right_wrench_world=[right_shear,-.7,0.,0.,0.,0.])
        trace.append(observer.update(sensor,passive))
    return trace,observer,sensor,passive


def test_withdrawal_recoil_requires_visible_contact_and_retreat():
    for shear, expected in ((-.01,True),(-.25,False)):
        trace,_,_,_ = stroke(shear)
        decisions = [r for r in trace if r["unload_event"]]
        assert len(decisions)==1 and decisions[0]["unload_evidence"]["bearing"] is expected
        assert trace[-1]["side_loaded"] is expected
        assert not any(r["unload_applied"] for r in trace[:58])
    assert not any(r["unload_event"] for r in stroke(-.01,.02)[0])


def test_contradiction_timeout_or_missing_samples_end_the_residual_witness():
    for mode in ("conflict","timeout","gap"):
        _,observer,sensor,passive = stroke(-.01)
        for i in range(1,130 if mode=="timeout" else 3):
            sensor["time_s"] += 1/120 if mode!="gap" else .04
            current = dict(passive,side_loaded=False) if mode=="conflict" and i==1 else passive
            result=observer.update(sensor,current)
        assert result["side_loaded"] is None and not result["unload_applied"]


def test_committed_parameters_match_paired_archive_fit():
    result=Path(__file__).resolve().parents[1]/"scenes/w760_300mm_two_layer/experiments/results/u4_left_unload_calibration.json"
    report=json.loads(result.read_text())
    assert np.allclose([report["intercept"],*report["weights"]],[INTERCEPT,*WEIGHTS],rtol=1e-10)
