"""Compare historical force-motion features while holding out one physical trial."""
import argparse
import json
from pathlib import Path

import numpy as np

from u4_sustained_calibrate import ARCHIVES, fit, history, validate
from u4_sustained_observer import FEATURES, PARAMETERS


ABLATIONS = {
    "full": range(len(FEATURES)),
    "force_changes_only": range(4, 9),
    "tcp_motion_only": range(4),
    "without_left_normal_change": [i for i in range(len(FEATURES)) if i != 4],
}
FOCUS = "u4_side_continuous/forward04"


def compare(output):
    root = PARAMETERS.parent
    runs = {}
    for archive in ARCHIVES:
        for path in sorted((root/archive).glob("*/paired.jsonl.gz")):
            run = history(path)
            if run is not None:
                runs[str(path.parent.relative_to(root))] = run
    results = {}
    for name, columns in ABLATIONS.items():
        omit = [i not in columns for i in range(len(FEATURES))]
        masked = {key: (events, np.where(np.isfinite(x) & omit, 0., x))
                  for key, (events, x) in runs.items()}
        folds = {}
        for key, run in masked.items():
            model = fit(v for held, v in masked.items() if held != key)
            folds[key] = validate(run, model)
            if name == "full" and key == FOCUS:
                declaration = folds[key]["results"]["fusion"]["first_clear_declaration"]
                clock, vectors = run
                vector = next(x for row, x in zip(clock, vectors, strict=True) if row[0] == declaration)
                terms = ((vector-model["center"])/model["scale"])*model["weights"]
                folds[key]["full_score_at_first_clear_declaration"] = {
                    "step_after_unload": declaration,
                    "feature_values": dict(zip(FEATURES, vector.tolist(), strict=True)),
                    "score_contributions": dict(zip(FEATURES, terms.tolist(), strict=True)),
                    "intercept": model["intercept"],
                    "score": float(model["intercept"] + terms.sum()),
                    "clear_cutoff": model["clear_below"],
                }
        summary = {}
        for model in ("baseline", "fusion"):
            metric = {
                "first_loss_hits": sum(f["results"][model]["first_clear_declaration"] is not None for f in folds.values()),
                "first_reload_hits": sum(f["results"][model]["first_reload_declaration"] is not None for f in folds.values()),
            }
            for section in ("handoff", "all"):
                metric[section] = {k: sum(f["results"][model][section][k] for f in folds.values())
                                   for k in ("clear", "false_clear", "loaded", "false_loaded")}
            summary[model] = metric
        results[name] = {"features": [FEATURES[i] for i in columns], "holdout_totals": summary,
                         "old_forward04": folds[FOCUS]}
    output.write_text(json.dumps({
        "scope": "Only pre-existing historical trials. Each fitted model excludes the whole evaluated trial. The same training settings and pressure/orientation eligibility apply to all ablations.",
        "trials": sorted(runs),
        "results": results,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    compare(parser.parse_args().output)
