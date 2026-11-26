"""Task contact-state classes and projection from the original seven labels."""

SEVEN_STATES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
TASK_STATES = ("S0", "S1", "S2", "S3", "S4", "S5")
LEGACY_TO_TASK = {"S5": "S1", "S6": "S5"}


def task_state(label):
    """Project a legacy label into the current six-state numbering."""
    return LEGACY_TO_TASK.get(label, label)
