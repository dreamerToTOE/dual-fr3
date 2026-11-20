"""Compare offline skill selections at the supported-push boundary."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from skill_selection import Skill, entry_event, select_skill


UNKNOWN = "UNKNOWN"
FORCE_THRESHOLD_N = 7.0
WRIST_LIMIT_N = 9.0


def force_event(row):
    if row["resultant_force_N"] > WRIST_LIMIT_N:
        return "OVERLOAD_WHILE_ADVANCING"
    if row["axial_force_N"] >= FORCE_THRESHOLD_N:
        return "TARGET_CANDIDATE"
    return "ADVANCING"


def select_from_trace(rows):
    cases = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)

    report = {}
    for case_id, samples in cases.items():
        previous = {}
        decisions = defaultdict(Counter)
        matches = Counter()
        premature = Counter()
        premature_events = Counter()
        first_confirmation = {}
        first_target = next((row["time_s"] for row in samples
                             if row["reference"] == "TARGET_CONTACT"), None)
        target_first_match = {}
        scored = 0
        code = Skill.SUPPORTED_PUSH
        for row in samples:
            observed = row["reference"]
            if observed == UNKNOWN:
                continue
            expected = select_skill(code, observed)
            scored += 1
            events = {**row["predictions"], "force_threshold": force_event(row)}
            for mode, event in events.items():
                # All methods in the physical scene have the same measured wrist limit.
                if row["resultant_force_N"] > WRIST_LIMIT_N:
                    event = "OVERLOAD_WHILE_ADVANCING"
                choice = select_skill(code, event)
                decisions[mode][choice.name] += 1
                matches[mode] += choice == expected
                wrong_confirm = choice == Skill.CONFIRM_TARGET and expected != choice
                premature[mode] += wrong_confirm
                last_step, prior_wrong_confirm = previous.get(mode, (None, False))
                if wrong_confirm and (not prior_wrong_confirm or last_step != row["physics_step"] - 1):
                    premature_events[mode] += 1
                previous[mode] = (row["physics_step"], wrong_confirm)
                if choice == Skill.CONFIRM_TARGET:
                    first_confirmation.setdefault(mode, row["time_s"])
                    if expected == choice:
                        target_first_match.setdefault(mode, row["time_s"])
        report[case_id] = {
            "task": samples[0]["task"],
            "scored_steps": scored,
            "first_physx_target_s": first_target,
            "methods": {
                mode: {
                    "agreement_with_loaded_contact_reference_steps": matches[mode],
                    "selected_skills": dict(counts),
                    "candidate_without_native_target_load_steps": premature[mode],
                    "candidate_without_native_target_load_intervals": premature_events[mode],
                    "first_skill_4_candidate_s": first_confirmation.get(mode),
                    "first_candidate_overlapping_native_target_s": target_first_match.get(mode),
                }
                for mode, counts in decisions.items()
            },
        }
    return report


def replay_entry_gates(run):
    report = json.loads((run / "runner_report.json").read_text())
    size = report["scene_parameters"]["cube_size_m"]
    entries = []
    for gate in report["phases"]:
        if gate["phase"] != "upper_B_straight_entry_gate":
            continue
        event = entry_event(gate["left_gap_m"], gate["right_gap_m"],
                            gate["projected_cube_width_m"], size, gate["minimum_gap_per_side_m"])
        current = Skill.REALIGN if gate.get("after_realign") else Skill.APPROACH
        entries.append({"time_s": gate["sim_time_s"], "observed_yaw_deg": gate["observed_yaw_deg"],
                        "min_gap_mm": 1000 * min(gate["left_gap_m"], gate["right_gap_m"]),
                        "event": event, "current_skill": current.name,
                        "selected_skill": select_skill(current, event).name})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--entry-run", type=Path, action="append", default=[],
                        help="Also replay entry decisions from this upper-5 run directory")
    args = parser.parse_args()
    with args.trace.open() as stream:
        cases = select_from_trace(json.loads(line) for line in stream if line.strip())
    args.output.write_text(json.dumps({
        "scope": "Passive replay at skill 3. The same recorded physics drives every method.",
        "skill_4_semantics": "Candidates enter a confirmation action. Passive cues do not authorize completing it.",
        "known_states_only": True,
        "axial_force_threshold_N": FORCE_THRESHOLD_N,
        "shared_wrist_limit_N": WRIST_LIMIT_N,
        "cases": cases,
        "entry_decisions": {run.name: replay_entry_gates(run) for run in args.entry_run},
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
