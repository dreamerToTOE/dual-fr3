import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
from u4_side_continuous import ContinuousSideObserver, FEATURES, WINDOW
from u4_contact_observer import SIDE, REAR


def sample(step, force):
    return {"time_s": step/120, "tcp_positions_m": {"right": [.15, 1., .93]},
            "cup_normals_world": {"left": [1., 0., 0.], "right": [0., 1., 0.]},
            "left_wrench_world": [0., 0., 0., 0., 0., 0.],
            "right_wrench_world": [force, -1., 0., 0., 0., 0.]}


def test_current_window_can_change_both_directions_without_holding():
    parameters = {"center": [0.]*len(FEATURES), "scale": [1.]*len(FEATURES), "intercept": 0.,
                  "weights": [0., -1., *([0.]*(len(FEATURES)-2))], "clear_below": -.1, "load_above": .1}
    observer = ContinuousSideObserver(parameters)
    base = {"side_gap_m": .001, "camera_age_s": .065, "side_loaded": None,
            "rear_loaded": True, "state": "UNKNOWN", "boundaries": [REAR]}
    for step in range(1, WINDOW+1):
        bearing = observer.update(sample(step, -.2), base)
    assert bearing["side_loaded"] is True and bearing["state"] == "S6"
    assert set(bearing["boundaries"]) == {SIDE, REAR}

    neutral = observer.update(sample(WINDOW+1, 0.), base)
    assert neutral["side_loaded"] is True  # One ambiguous sample cannot move a three-sample median.
    neutral = observer.update(sample(WINDOW+2, 0.), base)
    assert neutral["side_loaded"] is None and not neutral["continuous_side_applied"]

    for step in range(WINDOW+3, WINDOW+6):
        cleared = observer.update(sample(step, .2), base)
    assert cleared["side_loaded"] is False and cleared["state"] == "S5"
    for step in range(WINDOW+6, WINDOW+9):
        reload = observer.update(sample(step, -.2), base)
    assert reload["side_loaded"] is True

    passive_already_loaded = dict(base, side_loaded=True, state="S6", boundaries=[SIDE, REAR])
    for step in range(WINDOW+9, WINDOW+12):
        preserved = observer.update(sample(step, .2), passive_already_loaded)
    assert preserved["side_loaded"] is True and preserved["continuous_side_score"] < -.1
    assert not preserved["continuous_side_applied"]

    stale = observer.update(sample(WINDOW+12, -.2), dict(base, camera_age_s=.2))
    assert stale["side_loaded"] is None and not stale["continuous_side_applied"]
    gap = observer.update(sample(WINDOW+20, -.2), base)
    assert gap["side_loaded"] is None and not gap["continuous_side_applied"]
