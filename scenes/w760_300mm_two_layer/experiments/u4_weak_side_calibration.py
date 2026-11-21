"""Leave one archived U4 run out when calibrating a passive weak lateral-load detector."""
from collections import deque
import gzip
import json
from pathlib import Path

import numpy as np

from fusion_upper5 import FRONT_LOAD_N
from seven_reference import SevenReference
from u4_contact_observer import SIDE, U4ContactObserver
from u4_handover_score import anchors, load_at


FEATURES = ("side_gap_mm", "right_shear_N", "right_yaw_Nm", "front_load_N",
            "shear_std_N", "yaw_std_Nm", "right_encoder_dx_um")
CASES = ("older_u4", "nominal", "forward04", "back04")


class Features:
    def __init__(self):
        self.samples = deque(maxlen=24)

    def update(self, sensor, fusion):
        x = sensor["tcp_positions_m"]["right"][0]
        force = np.asarray(sensor["right_wrench_world"])
        normal = sensor["cup_normals_world"]["right"]
        front = -np.dot(normal, force[:3])
        self.samples.append((x, -force[0], -force[5], front))
        gap = fusion["side_gap_m"]
        age = fusion["camera_age_s"]
        if (len(self.samples) < 24 or normal[1] < .95 or gap is None or abs(gap) > .012
                or age is None or age > .4 or front < FRONT_LOAD_N):
            return None
        left_normal = sensor["cup_normals_world"]["left"]
        if left_normal[0] > .95 and -np.dot(left_normal, sensor["left_wrench_world"][:3]) >= FRONT_LOAD_N:
            return None
        data = np.asarray(self.samples)
        shear, yaw, contact = np.median(data[:, 1:], axis=0)
        if contact < FRONT_LOAD_N:
            return None
        return np.array([gap*1000, shear, yaw, contact, np.std(data[:, 1]), np.std(data[:, 2]),
                         (np.mean(data[-4:, 0]) - np.mean(data[:4, 0]))*1e6])


def archive(source):
    rows = []
    with gzip.open(source, "rt") as f:
        info = json.loads(next(f))
        if info["case"] != "u4":
            raise ValueError("Expected a U4 trajectory")
        bounds = anchors(info)
        observer = U4ContactObserver(info["config"])
        reference = SevenReference(info["config"], info["hz"], info["reference_positions"])
        features = Features()
        for line in f:
            obs, phys = json.loads(line)
            estimate = observer.update(obs, info["optical_neighbors"])
            x = features.update(obs, estimate)
            label = reference.update(phys)
            tick = obs["physics_step"]
            if tick != phys["physics_step"]:
                raise ValueError("Unpaired reference")
            # Boundaries select reporting intervals. The estimator has consumed the full prefix.
            if not bounds[1] < tick <= bounds[3]:
                continue
            loaded, normal = load_at(phys["cube_contacts"], label, SIDE, 0, info["hz"])
            # Legacy capture has only scalar impulse. Keep it for binary supervision and
            # preserve the missing directional force for weak-load recall.
            rows.append([tick <= bounds[2], loaded, normal, estimate["side_loaded"],
                         *(x if x is not None else [np.nan]*len(FEATURES))])
    return np.asarray(rows, dtype=float)


def train(training):
    # Each run and each known binary class receive equal total weight.
    groups = []
    for data in training:
        valid = np.isnan(data[:, 3]) & np.isfinite(data[:, 4])
        # Train on weak bearings, or witnessed legacy bearings with missing force.
        valid &= ((data[:, 1] == 0) | ((data[:, 1] == 1)
                   & (np.isnan(data[:, 2]) | ((data[:, 2] >= .1) & (data[:, 2] < 1)))))
        for target in (0, 1):
            x = data[valid & (data[:, 1] == target), 4:]
            if not len(x):
                raise ValueError("Both load classes must exist in each training trajectory")
            groups.append((x, target))
    x = np.concatenate([v for v, _ in groups])
    y = np.concatenate([np.full(len(v), target) for v, target in groups])
    w = np.concatenate([np.full(len(v), 1 / (len(v)*len(groups))) for v, _ in groups])
    center = w @ x
    scale = np.maximum(np.sqrt(w @ (x-center)**2), 1e-8)
    z = np.column_stack((np.ones(len(x)), (x-center)/scale))
    beta = np.zeros(z.shape[1])
    regularization = np.diag([0, *([.1]*len(FEATURES))])
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(z @ beta, -40, 40)))
        grad = z.T @ (w*(p-y)) + regularization @ beta
        hessian = (z.T*(w*p*(1-p))) @ z + regularization
        update = np.linalg.solve(hessian, grad)
        beta -= update
        if np.max(np.abs(update)) < 1e-8:
            break
    cutoff = max(float(np.quantile(beta[0] + ((v-center)/scale) @ beta[1:], .99))
                 for v, target in groups if target == 0)
    return center, scale, beta, cutoff


def evaluate(data, model):
    center, scale, beta, cutoff = model
    extra = np.isnan(data[:, 3]) & np.isfinite(data[:, 4]) & ((data[:, 4:]-center)/scale @ beta[1:] + beta[0] > cutoff)
    passive = data[:, 3] == 1
    out = {}
    for name, included in (("exchange_and_rear", np.ones(len(data), dtype=bool)),
                           ("exchange", data[:, 0] == 1), ("rear", data[:, 0] == 0)):
        weak = included & (data[:, 1] == 1) & (data[:, 2] >= .1) & (data[:, 2] < 1)
        pos, neg = included & (data[:, 1] == 1), included & (data[:, 1] == 0)
        out[name] = {"weak_N": int(sum(weak)), "weak_passive_tp": int(sum(weak & passive)),
                     "weak_added_tp": int(sum(weak & extra)), "positive_N": int(sum(pos)),
                     "added_tp": int(sum(pos & extra)), "negative_N": int(sum(neg)),
                     "passive_fp": int(sum(neg & passive)), "added_fp": int(sum(neg & extra))}
    return out


def main(output):
    results = Path(__file__).resolve().parent/"results"
    paths = {case: results/("seven/u4.jsonl.gz" if case=="older_u4"
                            else f"u4_handover/{case}/paired.jsonl.gz") for case in CASES}
    data = {case: archive(path) for case, path in paths.items()}
    folds = {}
    for hold, held_data in data.items():
        model = train([v for k,v in data.items() if k != hold])
        folds[hold] = {"cutoff": model[-1], "standardized_weights": dict(zip(FEATURES,model[2][1:])),
                       "windows": evaluate(held_data,model)}
    output.write_text(json.dumps(folds, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    main(p.parse_args().output)
