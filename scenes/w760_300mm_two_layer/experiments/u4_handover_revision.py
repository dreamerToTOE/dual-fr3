"""Compare U4 handoff estimates on paired sensor and independent PhysX streams."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from seven_observer import SevenObserver
from seven_reference import SevenReference
from u4_contact_observer import REAR, SIDE, U4ContactObserver
from u4_handover_score import WINDOWS, anchors, load_at, object_scores, predictions, state_scores
from u4_sustained_score import trial_events


def evaluate(path):
    with gzip.open(path, "rt") as stream:
        header = json.loads(next(stream))
        case, hz = header["case"], header["hz"]
        if case not in ("u4", "u5") or not header.get("tcp_commands"):
            raise ValueError("Expected a command-enriched U4 trial or U5 control")
        reference = SevenReference(header["config"], hz, header["reference_positions"])
        baseline = SevenObserver(header["config"], "fusion_compensated", revised=True)
        handover = U4ContactObserver(header["config"], handover=True)
        points = anchors(header) if case == "u4" else ()
        rows = []
        for line in stream:
            sensor, physical = json.loads(line)
            tick = sensor["physics_step"]
            if (tick != len(rows) + 1 or tick != physical["physics_step"]
                    or sensor["time_s"] != physical["time_s"]):
                raise ValueError(f"Unpaired sample at step {tick}")
            old = baseline.update(sensor, header["optical_neighbors"])
            new = handover.update(sensor, header["optical_neighbors"])
            truth = reference.update(physical)
            window = next((w for w, a, b in zip(WINDOWS, points, points[1:]) if a < tick <= b), "other")
            side, side_n = (load_at(physical["cube_contacts"], truth, SIDE, 0, hz)
                            if case == "u4" else (None, None))
            rear, _ = (load_at(physical["cube_contacts"], truth, REAR, 1, hz)
                       if case == "u4" else (None, None))
            baseline_side, baseline_rear = predictions(old)
            rows.append({"step": tick, "window": window, "truth_state": truth["state"],
                         "truth_objects": "|".join(truth["horizontal_objects"]),
                         "truth_side": side, "truth_side_N": side_n,
                         "truth_rear": rear,
                         "baseline_state": old["state"], "fusion_state": new["state"],
                         "baseline_objects": "|".join(old["boundaries"]), "fusion_objects": "|".join(new["boundaries"]),
                         "baseline_side": baseline_side,
                         "fusion_side": new["side_loaded"],
                         "baseline_rear": baseline_rear, "fusion_rear": new["rear_loaded"],
                         "handover_active": new["handover_active"], "supported_view": new["supported_view_used"]})
    if not rows or rows[-1]["step"] != header["stop_step"]:
        raise ValueError(f"Incomplete sensor stream for {path}")
    exchange = [r for r in rows if r["window"] in ("handoff", "rear_stroke")]
    weak = [r for r in exchange if r["truth_side"] is True and r["truth_side_N"] is not None
            and .1 <= r["truth_side_N"] < 1.]
    return {"source": header["source"], "case": case, "ticks": len(rows),
            "paired_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_sha256": header["source_sha256"], "stop": header.get("incomplete_run"),
            "changed_state_ticks": sum(r["baseline_state"] != r["fusion_state"] for r in rows),
            "changed_boundary_ticks": sum(r["baseline_objects"] != r["fusion_objects"] for r in rows),
            "handover_active_ticks": sum(r["handover_active"] for r in rows),
            "supported_handover_ticks": sum(r["handover_active"] and r["supported_view"] for r in rows),
            "states": {mode: state_scores(rows, mode) for mode in ("baseline", "fusion")},
            "exchange_side": ({mode: object_scores(exchange, mode, "side", hz)
                               for mode in ("baseline", "fusion")} if exchange else None),
            "exchange_rear": ({mode: object_scores(exchange, mode, "rear", hz)
                               for mode in ("baseline", "fusion")} if exchange else None),
            "weak_side": {"truth_ticks": len(weak), **{mode: sum(r[mode+"_side"] is True for r in weak)
                          for mode in ("baseline", "fusion")}},
            "events": trial_events(rows, hz) if case == "u4" else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("clips", type=Path, nargs="+")
    args = parser.parse_args()
    data = [evaluate(clip) for clip in args.clips]
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
