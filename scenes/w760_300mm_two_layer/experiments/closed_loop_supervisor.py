"""Small closed-loop supervisor used by the optional Isaac replay mode."""

from __future__ import annotations

from dataclasses import dataclass


MODE_TO_ESTIMATOR = {
    "vision": "vision_only",
    "tactile": "force_only",
    "fusion": "fusion",
}


def target_states(phase):
    if phase.startswith("A_"):
        if any(word in phase for word in ("lift", "transport", "carry")):
            return {"BILATERAL_SUSPENDED"}
        return {"BILATERAL_SUPPORTED", "NO_TOOL_CONTACT"}
    if phase.startswith("upper_B"):
        return {"LEFT_REAR_CONTACT", "DUAL_REAR_CONTACT", "DUAL_CORNER_CONTACT"}
    return set()


@dataclass
class ClosedLoopSupervisor:
    mode: str
    hold_after_unknown: int = 3
    max_hold_steps: int = 8

    def __post_init__(self):
        if self.mode not in MODE_TO_ESTIMATOR:
            raise ValueError(f"Unknown closed-loop mode {self.mode}")
        self.unknown_steps = 0
        self.hold_steps = 0
        self.observed_steps = 0

    @property
    def estimator_mode(self):
        return MODE_TO_ESTIMATOR[self.mode]

    def update(self, predicted_state, phase):
        targets = target_states(phase)
        self.observed_steps += 1
        if not targets or predicted_state in targets:
            self.unknown_steps = 0
            self.hold_steps = 0
            return {"action": "continue", "hold_sides": [], "predicted_state": predicted_state,
                    "target_states": sorted(targets), "reason": "target_observed"}
        # Any state outside the phase target is unsafe to advance through. The
        # short confirmation window avoids freezing on one noisy sample while
        # still treating a confidently wrong state as a mismatch.
        self.unknown_steps += 1
        if self.unknown_steps >= self.hold_after_unknown and self.hold_steps < self.max_hold_steps:
            self.hold_steps = min(self.hold_steps + 1, self.max_hold_steps)
            return {"action": "hold", "hold_sides": ["left", "right"],
                    "predicted_state": predicted_state, "target_states": sorted(targets),
                    "reason": "target_not_observed"}
        if self.hold_steps >= self.max_hold_steps:
            self.unknown_steps = 0
            self.hold_steps = 0
            return {"action": "continue", "hold_sides": [], "predicted_state": predicted_state,
                    "target_states": sorted(targets), "reason": "hold_budget_exhausted"}
        return {"action": "continue", "hold_sides": [], "predicted_state": predicted_state,
                "target_states": sorted(targets), "reason": "waiting_for_confirmation"}

