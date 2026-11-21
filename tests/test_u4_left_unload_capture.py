import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer/experiments"))
import u4_handover_capture as capture


@pytest.mark.parametrize("outcomes,stop_early", [
    ([True, True, True], True), ([False, True, False, True, True], True), ([False]*6, False),
])
def test_only_reach_of_left_unload_selects_repeats(monkeypatch, tmp_path, outcomes, stop_early):
    names = []
    offsets = []

    def run(command, **kwargs):
        assert "--u4-side-probe" not in command and "--close-on-finish" in command
        directory = command[2]
        directory.mkdir()
        names.append(directory.name)
        offsets.append(json.loads(command[4].read_text())["u4_initial_condition"]["active_box_world_Y_offset_m"])
        reached = outcomes[len(names)-1]
        phases = ([{"phase": "upper_B_side_unload", "status": "RUNNING", "cube_id": 10}] if reached else [])
        (directory / "runner_report.json").write_text(json.dumps({"status": "STOPPED", "phases": phases}))
        return capture.subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(capture.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["capture", "--left-unload", "isaac_python", str(tmp_path/"runs")])
    if stop_early:
        capture.main()
    else:
        with pytest.raises(SystemExit, match="six attempts"):
            capture.main()
    expected = [name+s for s in ("", "_repeat") for name,_ in capture.TRIALS]
    assert names == expected[:len(outcomes)]
    assert offsets == ([dy for _,dy in capture.TRIALS]*2)[:len(outcomes)]
