import ast
from pathlib import Path


def test_control_targets_cannot_read_body_truth_or_create_fixed_attachments():
    source = (Path(__file__).resolve().parents[1] / "fr3_ab_visual_runner.py").read_text()
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    approved = {
        ("main", "(initial_center, _)", False),
        ("step", "(reference_center, reference_orientation)", False),
        ("record", "(position, orientation)", False),
        ("main", "(position, orientation)", True),
    }
    reads = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        function = call.func
        if (isinstance(function, ast.Attribute) and function.attr == "Define"
                and isinstance(function.value, ast.Attribute)
                and function.value.attr == "FixedJoint"):
            raise AssertionError(f"Task attachment creates a rigid FixedJoint at line {call.lineno}")
        if not isinstance(function, ast.Attribute):
            continue
        if (isinstance(function.value, ast.Name) and function.value.id == "cube"
                and function.attr in {"get_linear_velocity", "get_angular_velocity", "get_local_pose"}):
            raise AssertionError(f"Control reads unapproved Cube state at line {call.lineno}")
        if function.attr != "get_world_pose":
            continue
        assert isinstance(function.value, ast.Name) and function.value.id == "cube", (
            f"Use encoder FK for body poses at line {call.lineno}"
        )
        scope, assignment, diagnostic = None, None, False
        ancestor = call
        while ancestor in parents:
            ancestor = parents[ancestor]
            if isinstance(ancestor, ast.Assign) and assignment is None:
                assignment = ", ".join(ast.unparse(target) for target in ancestor.targets)
            if isinstance(ancestor, ast.ExceptHandler):
                diagnostic = True
            if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = ancestor.name
                break
        site = (scope, assignment, diagnostic)
        assert site in approved, f"Control uses body truth at line {call.lineno} in {scope}"
        reads.append(site)
    assert len(reads) == len(approved) and set(reads) == approved
