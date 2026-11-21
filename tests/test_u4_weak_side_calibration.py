import numpy as np

from test_u4_contact_observer import front_sample
from u4_weak_side_calibration import FEATURES, Features, evaluate, train


def test_history_needs_visible_neighbor_and_loaded_front_cup():
    sensor = front_sample(1)
    observe = Features()
    quality = {"side_gap_m": .002, "camera_age_s": .05}
    for _ in range(23):
        assert observe.update(sensor, quality) is None
    vector = observe.update(sensor, quality)
    assert vector is not None and len(vector) == len(FEATURES)
    assert observe.update(sensor, dict(quality, camera_age_s=1.)) is None
    assert observe.update(sensor, dict(quality, side_gap_m=.015)) is None


def test_supplement_keeps_explicit_passive_decisions():
    rows = []
    for flag in (0, 1):
        for n in range(20):
            features = [flag + n/100, *([0.]*(len(FEATURES)-1))]
            rows.append([1, flag, .5 if flag else 0., np.nan, *features])
    data = np.array(rows)
    model = train([data, data.copy(), data.copy()])
    held = data.copy()
    held[-1, 3] = 0.
    s = evaluate(held, model)["exchange"]
    assert s["weak_N"] == 20 and s["weak_added_tp"] == 19
    assert s["added_fp"] <= 1
