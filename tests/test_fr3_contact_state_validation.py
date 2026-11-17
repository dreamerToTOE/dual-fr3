import ast
import json
from pathlib import Path

import numpy as np
import pytest

from fr3_contact_state_validation import recognition_metrics, reference_state


CONFIG = json.loads((Path(__file__).parents[1] / "fr3_four_wall_scene_config.json").read_text())


def _truth():
    return {"cube_contacts": {"valid": True, "loaded_contacts": dict.fromkeys([
        "left_cup", "right_cup", "floor", "supply", "wall_back", "wall_left", "wall_right"], False)},
        "cube_position_m": [0, .8, .787], "cube_rotation_world": np.eye(3).tolist(),
        "attachments": {"left": False, "right": False}}


def test_reference_uses_physics_and_attachment_not_phase_or_wrench():
    t = _truth()
    t.update(phase="B2_push", force_N=1000)
    assert reference_state(t, CONFIG)["state"] == "NO_TOOL_CONTACT"
    t["attachments"] = {"left": True, "right": True}
    t["cube_position_m"][2] = 1.1
    assert reference_state(t, CONFIG)["state"] == "BILATERAL_SUSPENDED"
    t["cube_contacts"]["loaded_contacts"]["floor"] = True
    assert reference_state(t, CONFIG)["state"] == "BILATERAL_SUPPORTED"
    t["cube_position_m"] = [-.244976, 1.076, .787]
    assert reference_state(t, CONFIG)["state"] == "DUAL_CORNER_CONTACT"
    t["attachments"] = {"left": False, "right": False}
    assert reference_state(t, CONFIG)["state"] == "RELEASED_CORNER"


def test_unknown_is_not_removed_to_inflate_accuracy():
    p = [{"state": state} for state in ["LEFT_FLOOR_CONTACT", "UNKNOWN", "NO_TOOL_CONTACT"]]
    t = [{"state": "LEFT_FLOOR_CONTACT"}] * 3
    result = recognition_metrics(p, t)
    assert result["accuracy_including_abstention_as_error"] == pytest.approx(1 / 3)
    assert result["decision_coverage"] == pytest.approx(2 / 3)
    assert result["accuracy_on_decided_samples"] == .5
    with pytest.raises(ValueError):
        recognition_metrics(p, t[:-1])


def test_invalid_reference_is_unknown_and_does_not_become_false_positive():
    t = _truth()
    t["cube_contacts"]["valid"] = False
    assert reference_state(t, CONFIG)["state"] == "UNKNOWN"
    metrics = recognition_metrics([{"state": "NO_TOOL_CONTACT"}] * 2,
                                   [{"state": "UNKNOWN"}, {"state": "NO_TOOL_CONTACT"}])
    assert metrics["by_state"]["NO_TOOL_CONTACT"]["precision"] == 1.
    assert metrics["predictions_on_unknown_reference"] == {"NO_TOOL_CONTACT": 1}


def test_skill_library_control_references_resolve_to_existing_runner_callables():
    root = Path(__file__).parents[1]
    tree = ast.parse((root / "fr3_ab_visual_runner.py").read_text())
    available = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    available |= {alias.asname or alias.name for node in ast.walk(tree)
                  if isinstance(node, ast.ImportFrom) for alias in node.names}
    library = json.loads((root / "fr3_contact_skills.json").read_text())
    assert all(skill["runtime_existing_functions"] for skill in library["skills"])
    assert {name for skill in library["skills"] for name in skill["runtime_existing_functions"]} <= available


def test_contact_event_delay_and_missed_state_are_reported_without_trimming_samples():
    truth = [{"state": "LEFT_FLOOR_CONTACT"}] * 60 + [{"state": "LEFT_REAR_CONTACT"}] * 60
    prediction = ([{"state": "UNKNOWN"}] * 12 + [{"state": "LEFT_FLOOR_CONTACT"}] * 48
                  + [{"state": "UNKNOWN"}] * 60)
    metrics = recognition_metrics(prediction, truth)
    assert metrics["samples"] == 120
    assert metrics["accuracy_including_abstention_as_error"] == .4
    assert metrics["recognition_delay_p95_s"] == .1
    assert metrics["stable_event_misses"] == 1
