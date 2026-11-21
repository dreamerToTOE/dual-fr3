"""Score the frozen U4 pulse against PhysX before and after the first lateral command."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from u4_handover_score import ACTIVE, events
from u4_left_unload_score import first_streak, left_cup_reference
from u4_preload_repeat_score import FROZEN, SUSTAINED, analyze, errors
from u4_pulse_score import digest, replay, write_timeline


PULSE_SHA = "71edc302e59bb6769aea353aa09a2b5f39376d03924e37ae1869d18f096d973b"
PHASE = "upper_B_u4_side_probe"


def boundary(rows, header):
    start = next((r["step"] for r in rows if r["segment"] == "out_neg"), None)
    marks = [round(p["sim_time_s"] * header["hz"]) + 1 for p in header["phase_times_for_scoring_only"]
             if (p["phase"], p["status"]) == (PHASE, "MOTION_COMMAND")]
    if start is None and marks:
        baseline = [r["step"] for r in rows if r["segment"] == "pre"]
        if not baseline or marks != [baseline[-1] + 1]:
            raise ValueError("A stopped pulse must end at the recorded motion boundary")
    elif marks != ([] if start is None else [start]):
        raise ValueError("The first lateral-motion sample must match the runner's command marker")
    return start


def reference_fields(physical, hz):
    fields = left_cup_reference(physical, hz)
    attached = physical["attachments"].get("left")
    contacts = [p.get("touching") if p["current_step_observed"] else None
                for p in physical["cube_contacts"]["pairs"]
                if p["label"] == "left_cup" and ACTIVE in (p["collider0"], p["collider1"])]
    detached = None
    if attached is True or fields["truth_left_cup"] is True or True in contacts:
        detached = False
    elif attached is False and fields["truth_left_cup"] is False and all(c is False for c in contacts):
        detached = True
    return {**fields, "truth_left_detached": detached}


def fixed_prestate(rows, start):
    """Use synchronized reference rows preceding motion, without reading any later truth."""
    prior = [r for r in rows if start - SUSTAINED <= r["step"] < start]
    full = [r["step"] for r in prior] == list(range(start - SUSTAINED, start))
    detached_steps = sum(r["truth_left_detached"] is True for r in prior)
    detached = full and detached_steps == SUSTAINED
    side = Counter(r["truth_side"] for r in prior)
    truth = next((state for state in (True, False) if detached and side[state] == SUSTAINED), None)
    forces = [r["truth_side_N"] for r in prior if r["truth_side_N"] is not None]
    label = {True: "bearing", False: "clear", None: "mixed_or_unobserved"}[truth]
    if not detached:
        label = "left_cup_clear_not_verified" if full else "incomplete_reference"
    return {"truth": truth, "label": label, "start_step": start - SUSTAINED, "end_step": start - 1,
            "left_cup_clear_steps": detached_steps,
            "bearing_steps": side[True], "clear_steps": side[False], "unknown_steps": side[None],
            "median_side_force_N": median(forces) if forces else None}


def prediction(call, truth, step):
    if truth is None:
        outcome = "unscored_reference"
    elif call is None:
        outcome = "abstain"
    elif call == truth:
        outcome = "correct"
    else:
        outcome = "false_bearing" if call else "false_clear"
    return {"call": call, "step": step, "outcome": outcome}


def transitions(rows, stable, state):
    if stable is not (not state):
        return None
    streak = first_streak(rows, "truth_side", state, SUSTAINED)
    forces = [r["truth_side_N"] for r in rows if r["truth_side"] is state and r["truth_side_N"] is not None]
    return {**events(rows, "truth_side", state),
            "first_12step_onset": streak - SUSTAINED + 1 if streak is not None else None,
            "first_12step_confirm": streak, "median_side_force_N": median(forces) if forces else None,
            "max_side_force_N": max(forces, default=None)}


def compare_trial(rows, start, hz, incomplete):
    # The pulse estimate about past state never enters historical event scoring.
    comparable = [{**r, "baseline_side": r["passive_side"], "fusion_side": r["active_side"],
                   "left_compression_N": None} for r in rows]
    pre = comparable if start is None else [r for r in comparable if r["step"] < start]
    observed = analyze(comparable, hz, incomplete, require_observed_loss=True)
    if start is None:
        return {"status": "NO_LATERAL_MOTION_SAMPLE", "before_motion": observed}
    moving = [r for r in comparable if r["step"] >= start and r["segment"]]
    last = moving[-1]["step"]
    new = [r for r in comparable if r["step"] > last]
    before = analyze(pre, hz, True, require_observed_loss=True)
    prestate = fixed_prestate(comparable, start)
    truth = prestate["truth"]
    decision = next((r for r in moving if r["probe_response"] is not None), None)
    passive = pre[-1] if pre and pre[-1]["step"] == start - 1 else None
    positive_step = next((r["step"] for r in moving
                          if r["segment"] == "cross_pos" and r["command_dx_m"] > 0), None)
    nonpositive = [r for r in moving if positive_step is None or r["step"] < positive_step]

    def phase_at(step):
        if step < start:
            return "before_motion"
        return "during_intervention" if step <= last else "after_intervention"

    for key in ("first_sustained_handoff_loss", "first_sustained_reload_within_handoff",
                "first_sustained_reload_including_rear"):
        event = observed.get(key)
        if event:
            event["onset_period"] = phase_at(event["start"])
            event["confirmation_period"] = phase_at(event["sustained_confirm_step"])
    return {
        "status": "LATERAL_MOTION_OBSERVED",
        "first_motion_step": start,
        "first_motion_after_unload_s": ((start - before["handoff_start_step"]) / hz
                                       if before["handoff_observed"] else None),
        "left_cup_detached_confirm_step": first_streak(
            [r for r in pre if r["window"] == "handoff"], "truth_left_detached", True, 3),
        "last_pulse_sample_step": last,
        "preexisting_side_load": prestate,
        "preexisting_state_comparison": {
            "passive_at_last_premotion_sample": prediction(
                passive["passive_side"], truth, passive["step"]) if passive else prediction(None, truth, None),
            "frozen_pulse_retrospective": prediction(decision["probe_before_loaded"], truth, decision["step"])
                if decision else prediction(None, truth, None),
            "passive_current_at_pulse_decision": decision["passive_side"] if decision else None,
            "frozen_pulse_response": decision["probe_response"] if decision else None,
        },
        "new_bearing_during_intervention_if_preclear": transitions(moving, truth, True),
        "unloading_before_positive_command_if_prebearing": transitions(nonpositive, truth, False),
        "new_bearing_before_positive_command_if_preclear": transitions(nonpositive, truth, True),
        "first_positive_command_step": positive_step,
        "premotion_events_only": before,
        "events_over_observed_motion": observed,
        "false_alarms_by_window": {
            "premotion_handoff": before.get("errors", {}).get("handoff"),
            "lateral_motion_and_settling": errors(moving, hz, pre[-hz:]),
            "post_pulse_handoff": errors([r for r in new if r["window"] == "handoff"], hz, (pre + moving)[-hz:]),
            "rear_stroke": observed.get("errors", {}).get("rear_stroke"),
        },
    }


def run(trial):
    paired = trial / "paired.jsonl.gz"
    commands = trial / "right_commands.json.gz"
    telemetry = trial / "u4_side_probe.jsonl"
    here = Path(__file__).parent
    models = {name: digest(here / name) for name in ("u4_side_probe_observer.py", "u4_contact_observer.py")}
    if models != {"u4_side_probe_observer.py": PULSE_SHA,
                  "u4_contact_observer.py": FROZEN["u4_contact_observer.py"]}:
        raise ValueError("Expected the archived, unchanged pulse and passive observers")
    # A stopped run can reach neither the pulse marker nor the telemetry file.
    header, _, rows = replay(paired, commands, telemetry, reference_fields=reference_fields)
    start = boundary(rows, header)
    hold = next((round(p["sim_time_s"] * header["hz"]) + 1 for p in header["phase_times_for_scoring_only"]
                 if (p["phase"], p["status"]) == (PHASE, "STARTED")), None)
    result = {"source": header["source"], "hz": header["hz"], "last_step": header["stop_step"],
              "incomplete_run": header.get("incomplete_run"),
              "observer_sha256": models, "paired_sha256": digest(paired), "commands_sha256": digest(commands),
              "probe_telemetry_sha256": digest(telemetry) if telemetry.is_file() else None,
              "side_load_before_baseline_hold": fixed_prestate(rows, hold) if hold is not None else None,
              **compare_trial(rows, start, header["hz"], header.get("incomplete_run"))}
    (trial / "probe_counts.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_timeline(trial / "probe_timeline.csv.gz", rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    run(parser.parse_args().trial)
