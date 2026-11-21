"""Estimate residual bearing from the U4 left cup's normal unloading curve."""
from collections import deque
from statistics import median

import numpy as np

from fusion_upper5 import FRONT_LOAD_N
from u4_contact_observer import REAR, SIDE


# Recomputed from the archived, pre-intervention withdrawals by u4_left_unload_calibrate.py.
INTERCEPT = 0.8522049462037623
WEIGHTS = (0.07240978494889197, 9.955218366084464)


def curve(samples):
    """Read the final 4 -> 2 -> 1 -> 0.2 N descent and verify actual outward travel."""
    if len(samples) < 16:
        return None
    a = np.asarray(samples)
    force = a[:, 1]
    if max(force[-4:]) > .22:
        return None
    # A three-tick median suppresses individual spikes; every value used predates the decision.
    filtered = np.median(np.array([force[:-2], force[1:-1], force[2:]]), axis=0)
    a = a[2:]
    crossings = [np.flatnonzero(filtered >= level) for level in (4., 2., 1., .22)]
    if any(not len(indices) for indices in crossings):
        return None
    high, middle, low, free = (indices[-1] for indices in crossings)
    if not high < middle-3 < low-4 < free-4 or a[free,0] - a[high,0] > .75:
        return None
    outward = max(a[high:free+1, 2]) - a[-1, 2]
    if outward < .000060:
        return None
    recoil = 1e6 * (median(a[middle-2:middle+1, 2]) - median(a[free-2:free+1, 2]))
    shear = float(median(a[middle:low+1, 3]))
    return {"left_normal_recoil_um": float(recoil), "right_shear_N": shear,
            "left_withdrawal_mm": float(1000*outward),
            "unload_duration_s": float(a[free,0]-a[high,0])}


class LeftUnloadObserver:
    def __init__(self):
        self.history = deque(maxlen=144)
        self.front = deque(maxlen=6)
        self.latched = True
        self.witness = None
        self.last_time = None

    def update(self, sensor, passive):
        now = sensor["time_s"]
        if self.last_time is not None and not 0 < now-self.last_time <= .015:
            self.history.clear()
            self.front.clear()
            self.witness = None
            self.latched = True
        self.last_time = now
        n_left, n_right = (sensor["cup_normals_world"][s] for s in ("left","right"))
        left, right = (sensor[s+"_wrench_world"] for s in ("left","right"))
        left_x = sensor["tcp_positions_m"]["left"][0]
        right_tcp = np.asarray(sensor["tcp_positions_m"]["right"])
        normal = -float(np.dot(n_left, left[:3]))
        self.front.append(-float(np.dot(n_right, right[:3])))
        age, gap = passive["camera_age_s"], passive["side_gap_m"]
        visible = age is not None and 0 <= age <= .4 and gap is not None and abs(gap) <= .012
        front_contact = n_right[1] > .95 and len(self.front) == 6 and median(self.front) >= FRONT_LOAD_N
        event = False
        evidence = None
        if n_left[0] > .95 and n_right[1] > .95:
            self.history.append((now, normal, left_x, right[0]))
            if normal >= 4.:
                self.latched = False
            if not self.latched and visible and front_contact and normal < .22:
                evidence = curve(self.history)
                if evidence:
                    self.latched = True
                    score = (INTERCEPT + WEIGHTS[0]*evidence["left_normal_recoil_um"]
                             + WEIGHTS[1]*evidence["right_shear_N"])
                    bearing = True if score >= .65 else False if score <= .35 else None
                    evidence.update(score=float(score), bearing=bearing, gap_m=gap, camera_age_s=age)
                    event = True
                    self.witness = ((bearing, now, right_tcp.copy(), gap) if bearing is not None else None)
        else:
            self.history.clear()
        out = dict(passive, unload_event=bool(event), unload_evidence=evidence, unload_applied=False,
                   unload_stage="unloading" if normal >= .22 and n_left[0] > .95 else "idle")
        if self.witness is None:
            return out
        bearing, stamp, anchor, old_gap = self.witness
        contradiction = passive["side_loaded"] is not None and passive["side_loaded"] != bearing
        if (contradiction or now-stamp > 1. or not visible or abs(gap-old_gap) > .003 or
                np.linalg.norm(right_tcp-anchor) > .00035 or normal >= 1.):
            self.witness = None
            return out
        out["unload_stage"] = "residual_window"
        if not front_contact or passive["side_loaded"] is not None:
            return out
        rear = passive["rear_loaded"]
        bounds = [k for k, yes in ((SIDE,bearing),(REAR,rear)) if yes]
        state = "UNKNOWN" if rear is None else ("S1","S5","S6")[len(bounds)]
        return dict(out, state=state, side_loaded=bearing, boundaries=sorted(bounds), unload_applied=True)
