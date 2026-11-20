"""Select the next packing skill from a sensor-derived event and task intent."""

from enum import IntEnum


class Skill(IntEnum):
    APPROACH = 1
    ESTABLISH_CONTACT = 2
    SUPPORTED_PUSH = 3
    CONFIRM_TARGET = 4
    HANDOFF = 5
    CROSS_SEAM = 6
    UNLOAD_WITHDRAW = 7
    REALIGN = 8


def entry_event(left_gap_m, right_gap_m, projected_width_m, cube_size_m, required_gap_m):
    gap = min(left_gap_m, right_gap_m)
    if gap >= required_gap_m - 1e-12:
        return "ENTRY_CLEAR"
    aligned_gap = gap + (projected_width_m - cube_size_m) / 2
    if aligned_gap >= required_gap_m - 1e-12:
        return "ENTRY_BLOCKED_YAW"
    return "ENTRY_BLOCKED_GEOMETRY"


def select_skill(current, event, *, handoff=False, another_push=False):
    transitions = {
        (Skill.APPROACH, "APPROACH_READY"): Skill.ESTABLISH_CONTACT,
        (Skill.APPROACH, "ENTRY_BLOCKED_YAW"): Skill.REALIGN,
        (Skill.ESTABLISH_CONTACT, "TOOL_CONTACT_STABLE"): Skill.SUPPORTED_PUSH,
        (Skill.SUPPORTED_PUSH, "SEAM_TRANSFER_REQUIRED"): Skill.CROSS_SEAM,
        (Skill.SUPPORTED_PUSH, "TARGET_CANDIDATE"): Skill.CONFIRM_TARGET,
        (Skill.SUPPORTED_PUSH, "TARGET_CONTACT"): Skill.CONFIRM_TARGET,
        (Skill.SUPPORTED_PUSH, "ALIGNED_POSITION"): Skill.UNLOAD_WITHDRAW,
        (Skill.SUPPORTED_PUSH, "OVERLOAD_WHILE_ADVANCING"): Skill.UNLOAD_WITHDRAW,
        (Skill.SUPPORTED_PUSH, "BLOCKED_EARLY"): Skill.UNLOAD_WITHDRAW,
        (Skill.CONFIRM_TARGET, "TARGET_STABLE"): Skill.HANDOFF if handoff else Skill.UNLOAD_WITHDRAW,
        (Skill.HANDOFF, "HANDOFF_CONTACT_STABLE"): Skill.SUPPORTED_PUSH,
        (Skill.CROSS_SEAM, "SUPPORT_REESTABLISHED"): Skill.SUPPORTED_PUSH,
        (Skill.UNLOAD_WITHDRAW, "WITHDRAWN"): Skill.APPROACH if another_push else None,
        (Skill.REALIGN, "ENTRY_CLEAR"): Skill.APPROACH,
    }
    return transitions.get((current, event), current)
