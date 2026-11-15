"""Check the new FR3 A/B scene contract without starting Isaac Sim.

This is a gate, not an A/B runtime.  It deliberately reports missing runtime
pieces instead of treating the static preview as a successful manipulation run.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "fr3_four_wall_scene_config.json"
PREVIEW_PATH = ROOT / "preview" / "fr3_four_wall_preview.usda"
REPORT_PATH = ROOT / "validation" / "fr3_ab_preflight.json"


def _inside_cube(center, cfg):
    size = float(cfg["cube_size_m"])
    floor = float(cfg["floor_top_m"])
    walls = {name: (list(center), list(size_)) for name, center, size_ in cfg["static_boxes"]}
    inner_x = (walls["wall_left"][0][0] + walls["wall_left"][1][0] / 2,
               walls["wall_right"][0][0] - walls["wall_right"][1][0] / 2)
    inner_y = (walls["wall_front"][0][1] + walls["wall_front"][1][1] / 2,
               walls["wall_back"][0][1] - walls["wall_back"][1][1] / 2)
    half = size / 2
    gaps = {
        "left_m": center[0] - half - inner_x[0],
        "right_m": inner_x[1] - center[0] - half,
        "front_m": center[1] - half - inner_y[0],
        "back_m": inner_y[1] - center[1] - half,
        "floor_m": center[2] - half - floor,
    }
    return gaps, all(value >= -1.0e-9 for value in gaps.values())


def main() -> int:
    cfg = json.loads(CONFIG_PATH.read_text())
    task = cfg["task"]
    phases = {
        "A_landing": task["phase_A"]["landing_center_m"],
        "B1_back_wall": task["phase_B1"]["target_center_m"],
        "B2_left_wall": task["phase_B2"]["target_center_m"],
    }
    phase_checks = {}
    for name, center in phases.items():
        gaps, inside = _inside_cube(center, cfg)
        phase_checks[name] = {"center_m": center, "boundary_gaps_m": gaps, "inside": inside}

    preview = PREVIEW_PATH.read_text() if PREVIEW_PATH.exists() else ""
    tool_section = re.search(r"/World/(?:left|right)_fr3/fr3_link8/PreviewSideSuction.*", preview, re.DOTALL)
    tool_collision = bool(tool_section and "CollisionAPI" in tool_section.group(0))
    runtime_contract = {
        "physics_scene": "/physicsScene" in preview,
        "dynamic_cube": "RigidBodyAPI" in preview and "/World/Cube" in preview,
        "tool_collision": tool_collision,
        "ik_waypoint_preflight": (ROOT / "fr3_ab_ik_preflight.py").exists(),
        "ab_state_machine": False,
        "joint_trajectory": False,
        "suction_attach_release": False,
    }
    geometry_ok = all(item["inside"] for item in phase_checks.values())
    runtime_ok = all(runtime_contract.values())
    result = {
        "scene_id": cfg["scene_id"],
        "scope": cfg["task"]["scope"],
        "status": "PASS_GEOMETRY_BLOCKED_RUNTIME",
        "geometry_status": "PASS" if geometry_ok else "BLOCKED",
        "runtime_contract_status": "PASS" if runtime_ok else "BLOCKED",
        "phase_checks": phase_checks,
        "runtime_contract": runtime_contract,
        "ab_status": "BLOCKED",
        "blockers": [
            "preview USD is geometry-only; no dynamic Cube rigid body or PhysicsScene",
            "no FR3 joint trajectory controller is present",
            "no physical suction attach/release or B1/B2 contact state machine is present",
        ],
        "claim": "This report does not claim A/B dynamics passed.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if geometry_ok else 1


if __name__ == "__main__":
    sys.exit(main())
