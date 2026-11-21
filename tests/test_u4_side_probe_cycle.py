import sys
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer"))
from experiments.u4_side_probe import cycle, detached_ready, inject, near_rear
from two_row_physx import build


def test_transverse_cycle_returns_to_entry_after_both_directions():
    samples = list(cycle())
    assert Counter(key for key, _ in samples) == {
        "pre": 12, "out_neg": 36, "hold_neg": 12, "cross_pos": 60,
        "hold_pos": 12, "return": 36, "post": 24}
    offsets = [value for _, value in samples]
    assert offsets[0] == offsets[-1] == 0.
    assert min(offsets) == -.00025 and max(offsets) == .00025
    assert offsets.index(min(offsets)) < offsets.index(max(offsets))
    assert max(abs(a - b) for a, b in zip(offsets, offsets[1:])) < .000016


def test_near_wall_trigger_needs_loaded_stalled_response_and_geometric_proximity():
    def apply(measured_y, command_step):
        window = deque(maxlen=61)
        return [near_rear(window, 1.076, .075, 1.+command_step*i, measured_y, 2.+.009*i)
                for i in range(61)]
    assert apply(1.001, .000001) == [False] * 60 + [True]
    assert not any(apply(.995, .000001))
    assert not any(apply(1.001, 0.))


def test_detached_probe_gate_needs_elapsed_retreat_and_low_wrist_force():
    assert detached_ready(96, .00015, .12)
    assert not detached_ready(95, .00015, .12)
    assert not detached_ready(96, .00014, .12)
    assert not detached_ready(96, .00015, -.121)


def test_detached_probe_injects_mid_unload_callback_without_changing_early_mode():
    source = build()
    detached = inject(source, detached=True)
    compile(detached, "<detached pulse runner>", "exec")
    assert "after_step=after_unload_step" in detached
    assert '"MOTION_COMMAND", command_step=tick + 1' in detached
    assert "after_step=after_unload_step" not in inject(source)
