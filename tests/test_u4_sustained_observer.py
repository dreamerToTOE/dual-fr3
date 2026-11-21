from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_sustained_observer import CONFIRM, FEATURES, WINDOW, ForceMotion, SustainedSideObserver


def sample(tick, right_x=0., right_fy=-1.):
    return {"time_s": tick / 120,
            "cup_normals_world": {"left": [1, 0, 0], "right": [0, 1, 0]},
            "left_wrench_world": [0, 0, 0, 0, 0, 0],
            "right_wrench_world": [0, right_fy, 0, 0, 0, 0],
            "tcp_positions_m": {"left": [0, 0, 0], "right": [right_x, 0, 0]}}


def test_same_wrench_with_distinct_encoder_motion_changes_the_evidence():
    static, advancing = ForceMotion(), ForceMotion()
    for tick in range(WINDOW):
        x = static.update(sample(tick))
        y = advancing.update(sample(tick, right_x=1e-6*tick))
    assert x[1] == 0 and y[1] > 0
    assert list(y[4:]) == [0]*7
    assert advancing.update(sample(WINDOW + 5, right_x=.001)) is None


def test_motion_updates_confirm_then_fall_back_immediately():
    weights = [0.]*len(FEATURES)
    weights[1] = 1e6
    observer = SustainedSideObserver({"center": [0.]*len(FEATURES), "scale": [1.]*len(FEATURES),
                                     "weights": weights, "intercept": 0., "clear_below": -1., "load_above": 1.})
    passive = {"side_loaded": None, "rear_loaded": False, "boundaries": [], "state": "UNKNOWN"}
    for tick in range(WINDOW+CONFIRM-2):
        result = observer.update(sample(tick, right_x=-tick*1e-6), passive)
        assert result["side_loaded"] is None
    tick = WINDOW+CONFIRM-2
    assert observer.update(sample(tick, right_x=-tick*1e-6), passive)["side_loaded"] is False
    assert observer.update(sample(tick+1, right_fy=0), passive)["side_loaded"] is None
    assert observer.update(sample(tick+2, right_x=-(tick+2)*1e-6), passive)["side_loaded"] is None
