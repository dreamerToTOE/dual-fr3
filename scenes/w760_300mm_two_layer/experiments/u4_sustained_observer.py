"""Estimate sustained U4 neighbor bearing from the measured force-motion response."""
from collections import deque
import json
from pathlib import Path

import numpy as np

from u4_contact_observer import REAR, SIDE


WINDOW = 30
END = 6
CONFIRM = 3
PARAMETERS = Path(__file__).with_name("results") / "u4_sustained_calibration.json"
FEATURES = (
    "left_dx_m", "right_dx_m", "right_dy_m", "right_dz_m",
    "left_normal_change_N", "right_normal_change_N",
    "right_fx_change_N", "right_fz_change_N", "right_mz_change_Nm",
    "right_lateral_work_proxy_J", "right_normal_work_proxy_J",
)


class ForceMotion:
    def __init__(self):
        self.samples = deque(maxlen=WINDOW)
        self.last_time = None

    def update(self, sensor):
        now = sensor["time_s"]
        if self.last_time is not None and not 0 < now - self.last_time <= .014:
            self.samples.clear()
        self.last_time = now
        nl, nr = (np.asarray(sensor["cup_normals_world"][arm]) for arm in ("left", "right"))
        left, right = (np.asarray(sensor[arm+"_wrench_world"]) for arm in ("left", "right"))
        l = -float(nl @ left[:3])
        r = -float(nr @ right[:3])
        lp, rp = (sensor["tcp_positions_m"][arm] for arm in ("left", "right"))
        self.samples.append((lp[0], *rp, l, r, right[0], right[2], right[5]))
        if len(self.samples) < WINDOW or nr[1] <= .95 or r <= .23 or l >= 1.:
            return None
        samples = np.asarray(self.samples)
        recent = np.median(samples[-END:], axis=0)
        change = recent - np.median(samples[:END], axis=0)
        return np.r_[change, recent[[6, 5]] * change[[1, 2]]]


class SustainedSideObserver:
    def __init__(self, parameters=None):
        cfg = parameters if parameters is not None else json.loads(PARAMETERS.read_text())["model"]
        self.center = np.asarray(cfg["center"])
        self.scale = np.asarray(cfg["scale"])
        self.weights = np.asarray(cfg["weights"])
        self.intercept = cfg["intercept"]
        self.clear_below = cfg["clear_below"]
        self.load_above = cfg["load_above"]
        self.motion = ForceMotion()
        self.previous = None
        self.confirming = 0

    def decision(self, score):
        label = (False if score is not None and score < self.clear_below else
                 True if score is not None and score > self.load_above else None)
        self.confirming = self.confirming + 1 if self.previous is label and label is not None else int(label is not None)
        self.previous = label
        return label if self.confirming >= CONFIRM else None

    def update(self, sensor, passive):
        x = self.motion.update(sensor)
        score = float(self.intercept + (x-self.center)/self.scale @ self.weights) if x is not None else None
        available = passive["side_loaded"] is None and score is not None
        bearing = self.decision(score if available else None)
        reason = ("preserve_passive" if passive["side_loaded"] is not None else
                  "no_supported_force_motion_window" if score is None else
                  "consistent_force_motion" if bearing is not None else
                  "insufficient_consistent_evidence")
        result = dict(passive, sustained_score=score, sustained_applied=bearing is not None,
                      sustained_reason=reason)
        if bearing is not None:
            rear = passive["rear_loaded"]
            bounds = sorted(name for name, present in ((SIDE, bearing), (REAR, rear)) if present)
            state = "UNKNOWN" if rear is None else ("S1", "S5", "S6")[len(bounds)]
            result.update(side_loaded=bearing, boundaries=bounds, state=state)
        return result
