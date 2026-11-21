"""U4 side-bearing estimate from each recent, causal vision and wrist window."""
from collections import deque
import json
from pathlib import Path

import numpy as np

from fusion_upper5 import FRONT_LOAD_N
from u4_contact_observer import SIDE, REAR


WINDOW = 6
PARAMETERS = Path(__file__).with_name("results") / "u4_side_continuous_calibration.json"
FEATURES = ("side_gap_m", "right_fx_N", "right_mz_Nm", "right_compression_N",
            "left_compression_N", "delta_right_fx_N", "delta_right_mz_Nm",
            "delta_right_compression_N", "delta_left_compression_N",
            "delta_right_tcp_x_m", "delta_right_tcp_y_m")


class SideFeatures:
    """Summarize the latest six samples. No prior classification enters the feature vector."""

    def __init__(self):
        self.samples = deque(maxlen=WINDOW)
        self.last_time = None

    def update(self, sensor, passive):
        now = sensor["time_s"]
        if self.last_time is not None and not 0 < now - self.last_time <= .014:
            self.samples.clear()
        self.last_time = now
        nl, nr = (np.asarray(sensor["cup_normals_world"][side]) for side in ("left", "right"))
        left, right = (np.asarray(sensor[side + "_wrench_world"]) for side in ("left", "right"))
        x, y = sensor["tcp_positions_m"]["right"][:2]
        l = -float(nl @ left[:3])
        r = -float(nr @ right[:3])
        if nr[1] <= .95:
            self.samples.clear()
            return None
        self.samples.append((right[0], right[5], r, l, x, y))
        if len(self.samples) != WINDOW:
            return None
        gap, age = passive["side_gap_m"], passive["camera_age_s"]
        if (gap is None or not abs(gap) <= .012 or age is None or not 0 <= age <= .16
                or l >= 1.0 or r < .18):
            return None
        recent = np.median(np.asarray(self.samples)[3:], axis=0)
        if recent[2] < FRONT_LOAD_N:
            return None
        early = np.median(np.asarray(self.samples)[:3], axis=0)
        return np.r_[gap, recent[:4], recent - early]


class ContinuousSideObserver:
    """Fill only the existing passive estimator's unknown side-bearing observations."""

    def __init__(self, parameters=None):
        model = parameters if parameters is not None else json.loads(PARAMETERS.read_text())["model"]
        self.center = np.array(model["center"])
        self.scale = np.array(model["scale"])
        self.weights = np.array(model["weights"])
        self.intercept = model["intercept"]
        self.clear_below = model["clear_below"]
        self.load_above = model["load_above"]
        self.features = SideFeatures()

    def update(self, sensor, passive):
        x = self.features.update(sensor, passive)
        out = dict(passive, continuous_side_applied=False, continuous_side_score=None,
                   continuous_side_reason="no_current_evidence")
        if x is None:
            return out
        score = float(self.intercept + (x - self.center) / self.scale @ self.weights)
        out["continuous_side_score"] = score
        if passive["side_loaded"] is not None:
            out["continuous_side_reason"] = "passive_already_decided"
            return out
        side = True if score > self.load_above else False if score < self.clear_below else None
        if side is None:
            out["continuous_side_reason"] = "overlapping_response"
            return out
        rear = passive["rear_loaded"]
        boundaries = sorted(path for path, loaded in ((SIDE, side), (REAR, rear)) if loaded)
        state = "UNKNOWN" if rear is None else ("S1", "S5", "S6")[len(boundaries)]
        return dict(out, side_loaded=side, state=state, boundaries=boundaries,
                    continuous_side_applied=True, continuous_side_reason="current_window")
