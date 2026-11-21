"""Replay the frozen U4 observers and measure pre-push sustained side contact."""
import argparse
from collections import Counter
import csv
import gzip
from itertools import groupby
import json
from pathlib import Path
from statistics import median

from u4_left_unload_score import first_streak
from u4_pulse_score import digest
from u4_side_continuous_score import declarations
from u4_sustained_score import MODELS, duration_class, evaluate_bouts, run


SUSTAINED = 12
FROZEN = {
    "u4_contact_observer.py": "8f90aaf4b10eefb803390a9b92459d9133f2a7650173111a673053a2a349c7ab",
    "u4_sustained_observer.py": "528c0aab3228d7e2980e8d9660d729b755c6aaa3a495bb768ef69671627e62eb",
    "u4_sustained_calibration.json": "fbef1f84c6c1cb8e23aed4fb3f55a5f5c259615308fbc2d301885916fa522c36",
}
STATES = {True: "bearing", False: "clear", None: "unknown"}


def intervals(rows, key):
    return [(value, len(list(streak))) for value, streak in groupby(rows, key=lambda r: r[key])]


def contact_span(rows, hz):
    lengths = intervals(rows, "truth_side")
    return {
        "ticks": len(rows),
        "seconds": len(rows)/hz,
        "counts": {name: sum(t for key,t in lengths if key is state) for state,name in STATES.items()},
        "longest": {name: max((t for key,t in lengths if key is state), default=0)
                    for state,name in STATES.items()},
        "from_start": {STATES[lengths[0][0]]: lengths[0][1]} if lengths else {},
        "bout_counts": {name: dict(Counter(duration_class(t) for key,t in lengths if key is state))
                        for state,name in STATES.items()},
    }


def errors(rows, hz, prior):
    if not rows:
        return None
    from_step = rows[0]["step"]
    result = {}
    for model in MODELS:
        alerts = Counter()
        for _, confirm, call, history in declarations([*prior, *rows], model, 3):
            if confirm >= from_step:
                if (not call) in history:
                    alerts["false_bearing_alerts" if call else "false_clear_alerts"] += 1
                elif None in history:
                    alerts["unverifiable_alerts"] += 1
        result[model] = {
            "truth_bearing_steps": sum(r["truth_side"] is True for r in rows),
            "truth_clear_steps": sum(r["truth_side"] is False for r in rows),
            "false_clear_steps": sum(r["truth_side"] is True and r[model+"_side"] is False for r in rows),
            "false_bearing_steps": sum(r["truth_side"] is False and r[model+"_side"] is True for r in rows),
            **{key: alerts[key] for key in ("false_clear_alerts", "false_bearing_alerts", "unverifiable_alerts")},
        }
    return result

def annotated_bout(bout, zero, hz, limit, interrupted=False, duration=SUSTAINED):
    if bout is None:
        return None
    return {**bout, "sustained_confirm_step": bout["start"] + duration - 1,
            "onset_s": (bout["start"]-zero)/hz, "duration_s": bout["ticks"]/hz,
            "continues_to_window_boundary": bout["end"]==limit,
            "right_censored_at_stop": interrupted and bout["end"]==limit}


def preload_outcome(rows, step, handoff, rear, hz, incomplete):
    limit = handoff[0]["step"]-1 if handoff else rows[-1]["step"]
    before_unload = [{**r, "window": "preload"} for r in rows if step < r["step"] <= limit]
    prior = [r for r in rows if step-hz < r["step"] <= step]
    report = {
        "start_step": step,
        "reference": "unload_command" if handoff else "preload_command",
        "contact_during_preload": contact_span(before_unload, hz),
        "errors": errors(before_unload, hz, prior),
        "first_sustained_preload_loss": None,
        "first_sustained_reload_including_rear": None,
    }
    if not before_unload:
        return report
    pre_push = before_unload + handoff
    physics, _ = evaluate_bouts(pre_push, hz, prior)
    pos = next((i for i, b in enumerate(physics) if
                b["window"] == "preload" and b["state"] is False and b["ticks"] >= SUSTAINED), None)
    if pos is not None:
        zero = limit if handoff else step
        loss = annotated_bout(physics[pos], zero, hz, pre_push[-1]["step"], bool(incomplete and not rear))
        loss["onset_observed"] = physics[pos-1]["state"] is True if pos else bool(
            prior and prior[-1]["step"]==step and prior[-1]["truth_side"] is True)
        loss["preload_ticks"] = min(loss["end"], limit)-loss["start"]+1
        loss["handoff_ticks"] = loss["ticks"] - loss["preload_ticks"]
        report["first_sustained_preload_loss"] = loss
        exchange = pre_push + rear
        states = evaluate_bouts(exchange, hz, prior)[0] if rear else physics
        reload = following_reload(states, loss["end"])
        report["first_sustained_reload_including_rear"] = annotated_bout(
            reload, zero, hz, exchange[-1]["step"], bool(incomplete))
        report["reload_right_censored_at_stop"] = bool(incomplete and reload is None)
    return report


def following_reload(bouts, after, duration=SUSTAINED):
    return next((r for prev,r in zip(bouts,bouts[1:]) if r["state"] is True and r["ticks"]>=duration
                 and r["start"] > after and prev["state"] is False), None)


def analyze(rows, hz, incomplete, preload_start=None, *, require_observed_loss=False):
    handoff = [r for r in rows if r["window"]=="handoff"]
    rear = [r for r in rows if r["window"]=="rear_stroke"]
    preload = preload_outcome(rows, preload_start, handoff, rear, hz, incomplete) if preload_start is not None else None
    if not handoff:
        return {"handoff_observed": False, "right_push_observed": bool(rear), "preload": preload,
                "handoff_loss_eligible": None, "end_reason": "stopped_before_handoff" if incomplete else "missing_handoff"}

    first, last = handoff[0]["step"], handoff[-1]["step"]
    zero = first-1
    prior = [r for r in rows if zero-hz < r["step"] <= zero]
    exchange = handoff + rear
    end = exchange[-1]["step"]
    physics, _ = evaluate_bouts(handoff, hz, prior)
    combined, _ = evaluate_bouts(exchange, hz, prior)

    def loss_onset_observed(idx):
        if idx:
            return physics[idx-1]["state"] is True
        return bool(prior and prior[-1]["step"]==zero and prior[-1]["truth_side"] is True)

    def contact_events(duration):
        idx = next((i for i, r in enumerate(physics) if r["state"] is False and r["ticks"] >= duration
                    and (not require_observed_loss or loss_onset_observed(i))), None)
        loss = physics[idx] if idx is not None else None
        local_reload = following_reload(physics, loss["end"], duration) if loss else None
        reload = following_reload(combined, loss["end"], duration) if loss else None
        result = {
            "first_sustained_handoff_loss": annotated_bout(loss,zero,hz,last,bool(incomplete and not rear),duration),
            "first_sustained_reload_within_handoff": annotated_bout(
                local_reload,zero,hz,last,bool(incomplete and not rear),duration),
            "first_sustained_reload_including_rear": annotated_bout(reload,zero,hz,end,bool(incomplete),duration),
            "reload_right_censored_at_stop": bool(incomplete and loss is not None and reload is None),
        }
        if loss is not None:
            result["first_sustained_handoff_loss"]["onset_observed"] = loss_onset_observed(idx)
        return result

    primary = contact_events(SUSTAINED)
    cup_confirm = first_streak(handoff,"truth_left_cup",False,3)
    postrelease = [] if cup_confirm is None else [r for r in handoff if r["step"]>=cup_confirm]
    short = postrelease[:SUSTAINED]
    first12 = (STATES[short[0]["truth_side"]] if len(short)==SUSTAINED and
               all(r["truth_side"] is short[0]["truth_side"] for r in short)
               else "mixed" if len(short)==SUSTAINED else "insufficient")
    initial = postrelease[:round(1.5*hz)]
    before = [r["left_compression_N"] for r in prior[-SUSTAINED:] if r["left_compression_N"] is not None]
    return {
        "handoff_observed": True,
        "right_push_observed": bool(rear),
        "preload": preload,
        "handoff_start_step": zero,
        "last_handoff_step": last,
        "end_reason": "physical_stop_or_capture_end" if incomplete else "completed_capture",
        "handoff_bouts": contact_span(handoff,hz),
        "rear_bouts": contact_span(rear,hz),
        "last_12_pre_unload_median_left_compression_N": median(before) if before else None,
        "left_cup_clear_confirm_step": cup_confirm,
        "first_12_from_left_cup_clear": first12,
        "from_left_cup_clear": contact_span(postrelease,hz),
        "first_1p5s_from_left_cup_clear": contact_span(initial,hz),
        "handoff_loss_eligible": primary["first_sustained_handoff_loss"] is not None,
        **primary,
        "sensitivity_24": contact_events(24),
        "errors": {
            "handoff": errors(handoff,hz,prior),
            "rear_stroke": errors(rear,hz,(prior+handoff)[-hz:]),
        },
    }


def score_trial(trial):
    here = Path(__file__).parent
    hashes = {name: digest((here if name.endswith(".py") else here/"results")/name) for name in FROZEN}
    if hashes != FROZEN:
        raise ValueError("U4 passive or force-motion observer differs from the frozen version")
    run(trial)
    counts = json.loads((trial/"counts.json").read_text())
    with gzip.open(trial/"paired.jsonl.gz","rt") as f:
        header = json.loads(next(f))
    with gzip.open(trial/"timeline.csv.gz","rt") as f:
        rows = []
        for row in csv.DictReader(f):
            parsed = {name: {"1": True, "0": False, "": None}[row[name]] for name in
                      ("truth_side","truth_left_cup","baseline_side","fusion_side")}
            rows.append({**parsed, "step": int(row["step"]), "window": row["window"],
                         "left_compression_N": float(row["left_compression_N"]) if row["left_compression_N"] else None})
    preload = next((round(p["sim_time_s"]*header["hz"]) for p in header["phase_times_for_scoring_only"]
                    if p["phase"]=="upper_B_u4_left_preload" and p["status"]=="RUNNING"), None)
    result = {"source": counts["source"], "hz": header["hz"],
              "model_sha256": counts["model_sha256"], "paired_sha256": counts["paired_sha256"],
              "last_step": counts["last_step"], "incomplete_run": counts["incomplete_run"],
              **analyze(rows,header["hz"],counts["incomplete_run"],preload)}
    (trial/"handoff_counts.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def summarize(root):
    trials = [json.loads(p.read_text()) for p in sorted(root.glob("*/handoff_counts.json"))]
    if not trials or any(t["model_sha256"]!=FROZEN for t in trials):
        raise ValueError("Missing trials or unexpected observer provenance")
    result = {"trial_count": len(trials), "with_handoff": sum(t["handoff_observed"] for t in trials),
              "natural_handoff_sustained_loss_trials": sum(t["source"].startswith("natural_") and
                    bool(t["handoff_loss_eligible"]) for t in trials),
              "handoff_sustained_loss_trials": sum(bool(t["handoff_loss_eligible"]) for t in trials),
              "trials": trials}
    result["matches"] = {}
    for key in ("first_sustained_handoff_loss","first_sustained_reload_within_handoff",
                "first_sustained_reload_including_rear"):
        eligible = [t[key] for t in trials if t.get(key) is not None and t[key].get("onset_observed",True)]
        result["matches"][key] = {"truth_trials":len(eligible),
            **{m:sum(r[m]["confirm_step"] is not None for r in eligible) for m in MODELS}}
    result["errors"] = {}
    for w in ("preload","handoff","rear_stroke"):
        result["errors"][w] = {}
        for m in MODELS:
            pooled = Counter()
            for t in trials:
                e = (t.get("preload") or {}).get("errors") if w=="preload" else t.get("errors",{}).get(w)
                if e is not None:
                    pooled.update(e[m])
            result["errors"][w][m] = {
                **pooled,
                "false_clear_rate": pooled["false_clear_steps"]/pooled["truth_bearing_steps"]
                    if pooled["truth_bearing_steps"] else None,
                "false_bearing_rate": pooled["false_bearing_steps"]/pooled["truth_clear_steps"]
                    if pooled["truth_clear_steps"] else None,
            }
    (root/"handoff_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()
    (summarize if args.summary else score_trial)(args.path)
