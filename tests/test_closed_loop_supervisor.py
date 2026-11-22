import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scenes/w760_300mm_two_layer"))
sys.path.insert(0, str(ROOT / "scenes/w760_300mm_two_layer/experiments"))
from closed_loop_supervisor import ClosedLoopSupervisor, target_states


def test_phase_targets_are_state_sets():
    assert "BILATERAL_SUSPENDED" in target_states("A_lift")
    assert "LEFT_REAR_CONTACT" in target_states("upper_B_side")
    assert "DUAL_REAR_CONTACT" in target_states("upper_B_rear")
    assert "DUAL_CORNER_CONTACT" in target_states("upper_B_u3_side_pitch")


def test_unknown_requires_three_samples_then_holds():
    supervisor = ClosedLoopSupervisor("fusion")
    assert supervisor.update("UNKNOWN", "upper_B_rear")["action"] == "continue"
    assert supervisor.update("UNKNOWN", "upper_B_rear")["action"] == "continue"
    assert supervisor.update("UNKNOWN", "upper_B_rear")["action"] == "hold"
    assert supervisor.update("DUAL_REAR_CONTACT", "upper_B_rear")["action"] == "continue"


def test_confident_wrong_state_is_also_gated_after_confirmation_window():
    supervisor = ClosedLoopSupervisor("fusion")
    assert supervisor.update("BILATERAL_SUPPORTED", "upper_B_rear")["action"] == "continue"
    assert supervisor.update("BILATERAL_SUPPORTED", "upper_B_rear")["action"] == "continue"
    assert supervisor.update("BILATERAL_SUPPORTED", "upper_B_rear")["action"] == "hold"


def test_vision_and_tactile_select_distinct_estimators():
    assert ClosedLoopSupervisor("vision").estimator_mode == "vision_only"
    assert ClosedLoopSupervisor("tactile").estimator_mode == "force_only"
    assert ClosedLoopSupervisor("fusion").estimator_mode == "fusion"


def test_hold_budget_expires_instead_of_freezing_forever():
    supervisor = ClosedLoopSupervisor("fusion", hold_after_unknown=1, max_hold_steps=2)
    assert supervisor.update("UNKNOWN", "upper_B_rear")["action"] == "hold"
    assert supervisor.update("UNKNOWN", "upper_B_rear")["action"] == "hold"
    decision = supervisor.update("UNKNOWN", "upper_B_rear")
    assert decision["action"] == "continue"
    assert decision["reason"] == "hold_budget_exhausted"


def test_generated_runner_closed_loop_hook_is_optional_and_compiles():
    from experiments.closed_loop_inject import inject
    from two_row_physx import build

    source = build()
    compile(source, "<observer-only runner>", "exec")
    assert "closed_loop_supervisor" not in source
    for mode in ("vision", "tactile", "fusion"):
        injected = inject(source, mode)
        compile(injected, f"<{mode} runner>", "exec")
        assert f'"mode": CLOSED_LOOP_MODE' in injected
        assert '"supervisor_action"' in injected
        assert '"target_state"' in injected
        assert '"physx_state"' in injected
