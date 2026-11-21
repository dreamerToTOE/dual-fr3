"""Interpret one reversible lateral displacement of the loaded U4 front cup."""
from collections import defaultdict, deque

import numpy as np

from fusion_upper5 import FRONT_LOAD_N
from u4_contact_observer import SIDE, REAR


class PulseObserver:
    """Consume the issued pulse and current measurements; leave the passive observer untouched."""

    def __init__(self):
        self.segments = defaultdict(list)
        self.front = deque(maxlen=6)
        self.last_step = None
        self.witness = None

    def _interpret(self):
        groups = self.segments
        # Use only samples on the approach to neutral from negative X. Positive
        # travel can close an initial gap and cannot witness pre-pulse load.
        cross = [r for r in groups["cross_pos"] if -.000075 <= r[0] <= 0]
        phases = [groups["pre"], groups["hold_neg"], cross, groups["hold_pos"], groups["post"]]
        if any(len(p) < 4 for p in phases):
            return None, {"reason": "incomplete_pulse"}
        # dx commanded, encoder X, force opposing +X, front compression, geometry validity
        pre, neg, zero, pos, post = (np.median(np.asarray(p[-8:]), axis=0) for p in phases)
        noise = np.median(np.abs(np.asarray(groups["pre"])[:, 2] - pre[2]))
        tol = max(.080, 1.5 * noise)
        unload, reload = pre[2] - neg[2], zero[2] - neg[2]
        compression = pos[2] - pre[2]
        stats = {"force_tolerance_N": float(tol), "unload_N": float(unload),
                 "reload_before_positive_command_N": float(reload),
                 "compression_N": float(compression), "return_residual_N": float(post[2] - pre[2]),
                 "negative_travel_m": float(pre[1] - neg[1]), "positive_travel_m": float(pos[1] - pre[1])}
        if any(row[3] < FRONT_LOAD_N or row[4] < 1 for row in (pre, neg, zero, pos, post)):
            return None, dict(stats, reason="cup_or_marker_unconfirmed")
        if min(pre[1] - neg[1], pos[1] - pre[1]) < .000040 or abs(post[1] - pre[1]) > .000080:
            return None, dict(stats, reason="encoder_stroke_or_return_unconfirmed")
        # An approximately symmetric stiffness can be caused by support friction or cup shear.
        if unload >= tol and reload >= .5 * tol and compression - unload >= .5 * tol:
            return True, dict(stats, reason="unload_recovered_before_neutral_and_directional_resistance")
        if abs(unload) <= .5 * tol and abs(reload) <= .5 * tol and compression >= 2 * tol:
            return False, dict(stats, reason="free_negative_stroke_resistance_only_at_positive_displacement")
        return None, dict(stats, reason="ambiguous_directional_response")

    def update(self, sensor, passive_result, pulse=None):
        step = sensor["physics_step"]
        if self.last_step is not None and step != self.last_step + 1:
            self.segments.clear()
            self.front.clear()
            self.witness = None
        self.last_step = step
        now = sensor["time_s"]
        wrench = sensor["right_wrench_world"]
        n = np.asarray(sensor["cup_normals_world"]["right"])
        tcp = np.asarray(sensor["tcp_positions_m"]["right"])
        front = -np.dot(n, wrench[:3])
        self.front.append(front)
        age = passive_result["camera_age_s"]
        gap = passive_result["side_gap_m"]
        near = age is not None and age <= .40 and gap is not None and abs(gap) <= .012
        before, evidence = None, None
        if pulse is not None:
            if pulse["physics_step"] != step:
                raise ValueError("Side pulse and observation disagree on physics step")
            segment = pulse["segment"]
            self.segments[segment].append((pulse["command_dx_m"], tcp[0], -wrench[0], front,
                                           near and n[1] > .95))
            if segment == "post" and len(self.segments["post"]) == 12:
                before, evidence = self._interpret()
                if before is not None and abs(evidence["return_residual_N"]) <= 1.5 * evidence["force_tolerance_N"]:
                    self.witness = (before, now, tcp.copy(), gap, float(np.median(self.front)))
        else:
            self.segments.clear()

        out = dict(passive_result, probe_before_loaded=before, probe_response=evidence, probe_applied=False)
        if self.witness is None:
            return out
        bearing, stamp, anchor, old_gap, old_force = self.witness
        current_force = np.median(self.front)
        if (not near or n[1] <= .95 or now - stamp > 1. or np.linalg.norm(tcp - anchor) > .00010
                or abs(gap - old_gap) > .003 or current_force < FRONT_LOAD_N
                or abs(current_force - old_force) > .4
                or (passive_result["side_loaded"] is not None and passive_result["side_loaded"] != bearing)):
            self.witness = None
            return out
        rear = passive_result["rear_loaded"]
        boundaries = [target for target, active in ((SIDE, bearing), (REAR, rear)) if active]
        state = "UNKNOWN" if rear is None else ("S1", "S5", "S6")[len(boundaries)]
        return dict(out, state=state, side_loaded=bearing, boundaries=sorted(boundaries), probe_applied=True)
