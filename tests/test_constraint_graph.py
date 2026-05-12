import ast
from pathlib import Path

import pytest

from core.constraints import ConstraintGraph, SUPPORTED_CONSTRAINT_TYPES, build_constraint_graph, supports_constraint_type
from core.types import Constraint


def test_constraint_graph_supports_add_get_remove_and_list() -> None:
    graph = ConstraintGraph()
    tangent = Constraint(
        constraint_id="constraint_tangent_1",
        type="tangent",
        targets=("segment_a", "segment_b"),
        source="system",
        confidence=0.92,
    )

    updated_graph = graph.add_constraint(tangent)

    assert graph.list_constraints() == ()
    assert updated_graph.has_constraint("constraint_tangent_1") is True
    assert updated_graph.get_constraint("constraint_tangent_1") == tangent
    assert updated_graph.list_constraints() == (tangent,)

    reduced_graph = updated_graph.remove_constraint("constraint_tangent_1")

    assert reduced_graph.list_constraints() == ()
    assert updated_graph.list_constraints() == (tangent,)


def test_constraint_graph_indexes_constraints_by_target_id() -> None:
    tangent = Constraint(
        constraint_id="constraint_tangent_1",
        type="tangent",
        targets=("segment_a", "segment_b"),
    )
    shared_tangent = Constraint(
        constraint_id="constraint_shared_tangent_1",
        type="shared_tangent",
        targets=("anchor_a", "segment_a", "segment_b"),
    )
    g1_continuity = Constraint(
        constraint_id="constraint_g1_1",
        type="g1_continuity",
        targets=("anchor_a", "segment_b", "segment_c"),
    )
    graph = build_constraint_graph((tangent, shared_tangent, g1_continuity))

    assert graph.constraints_for_target("segment_a") == (tangent, shared_tangent)
    assert graph.constraints_for_target("anchor_a") == (shared_tangent, g1_continuity)
    assert graph.constraints_for_target("segment_c") == (g1_continuity,)
    assert graph.constraints_for_target("missing_target") == ()


def test_constraint_graph_accepts_required_constraint_types() -> None:
    constraints = tuple(
        Constraint(
            constraint_id=f"constraint_{constraint_type}",
            type=constraint_type,
            targets=(f"{constraint_type}_target_a", f"{constraint_type}_target_b"),
        )
        for constraint_type in SUPPORTED_CONSTRAINT_TYPES
    )

    graph = build_constraint_graph(constraints)

    assert graph.list_constraints() == constraints
    assert all(supports_constraint_type(constraint_type) for constraint_type in SUPPORTED_CONSTRAINT_TYPES)


def test_constraint_graph_rejects_unsupported_constraint_type() -> None:
    with pytest.raises(ValueError, match="unsupported constraint type"):
        ConstraintGraph().add_constraint(
            Constraint(
                constraint_id="constraint_bad",
                type="parallel",
                targets=("segment_a", "segment_b"),
            )
        )


def test_constraint_graph_rejects_duplicate_constraint_ids() -> None:
    tangent = Constraint(
        constraint_id="constraint_tangent_1",
        type="tangent",
        targets=("segment_a", "segment_b"),
    )
    graph = ConstraintGraph((tangent,))

    with pytest.raises(ValueError, match="duplicate constraint_id"):
        graph.add_constraint(tangent)


def test_constraint_graph_rejects_duplicate_target_ids_within_one_constraint() -> None:
    duplicate_targets = Constraint(
        constraint_id="dup_target",
        type="coincident",
        targets=("same", "same"),
    )

    with pytest.raises(ValueError, match="duplicate target ids"):
        ConstraintGraph((duplicate_targets,))

    with pytest.raises(ValueError, match="duplicate target ids"):
        ConstraintGraph().add_constraint(duplicate_targets)


def test_constraint_graph_remove_missing_constraint_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown constraint_id"):
        ConstraintGraph().remove_constraint("constraint_missing")


def test_constraint_graph_has_no_forbidden_dependencies() -> None:
    source_path = Path("core/constraints.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic", "ui"}

    assert imports.isdisjoint(forbidden_imports)
    assert "open(" not in source
