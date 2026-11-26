"""Task contact-state classes and projection from the original seven labels."""

SEVEN_STATES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
TASK_STATES = ("S0", "S1", "S2", "S3", "S4", "S6")


def task_state(label):
    """Supported pushes with zero or one bearing boundary share S1."""
    return "S1" if label == "S5" else label
