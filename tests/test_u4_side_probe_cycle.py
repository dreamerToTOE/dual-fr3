import sys
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenes/w760_300mm_two_layer"))
from experiments.u4_side_probe import cycle, near_rear


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
