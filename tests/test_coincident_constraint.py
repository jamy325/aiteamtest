import ast
from pathlib import Path

import pytest

from core.document import add_anchor, add_object, add_path, create_document
from core.types import Anchor, CoordinateSystem, Object, Path as VectorPath
from services.snapping import (
    CoincidentConstraintConfig,
    CoincidentConstraintEngine,
    GlobalSnappingEngine,
    SnappingConfig,
)


def _base_document():
    document = create_document(
        document_id="doc_coincident",
        width=160.0,
        height=120.0,
        coordinate_system=CoordinateSystem(),
    )
    object_a = Object(object_id="object_a", type="shape")
    object_b = Object(object_id="object_b", type="shape")
    paths = (
        VectorPath(path_id="path_a", object_id="object_a"),
        VectorPath(path_id="path_b", object_id="object_a"),
        VectorPath(path_id="path_c", object_id="object_b"),
    )

    document = add_object(document, object_a)
    document = add_object(document, object_b)
    for path in paths:
        document = add_path(document, path)
    return document


def test_coincident_constraint_engine_generates_hard_constraint_for_high_confidence_unlocked_candidate() -> None:
    document = _base_document()
    for anchor in (
        Anchor(anchor_id="a1", path_id="path_a", position=(12.0, 24.0)),
        Anchor(anchor_id="b1", path_id="path_b", position=(12.0, 24.0)),
    ):
        document = add_anchor(document, anchor)

    engine = CoincidentConstraintEngine(
        snapping_engine=GlobalSnappingEngine(SnappingConfig(epsilon=0.2)),
    )

    constraints = engine.generate_constraints(document)

    assert len(constraints) == 1
    constraint = constraints[0]
    assert constraint.type == "coincident"
    assert constraint.targets == ("a1", "b1")
    assert constraint.strength == 1.0
    assert constraint.source == "global_snapping"
    assert constraint.confidence == 1.0
    assert constraint.metadata["mode"] == "hard"
    assert constraint.metadata["relation"] == "cross_path_same_object"
    assert constraint.metadata["distance"] == 0.0


def test_coincident_constraint_engine_generates_soft_constraint_for_locked_candidate_and_applies_it() -> None:
    document = _base_document()
    for anchor in (
        Anchor(anchor_id="a1", path_id="path_a", position=(20.0, 10.0)),
        Anchor(anchor_id="c1", path_id="path_c", position=(20.01, 10.0), locked=True),
    ):
        document = add_anchor(document, anchor)

    engine = CoincidentConstraintEngine(
        snapping_engine=GlobalSnappingEngine(SnappingConfig(epsilon=0.2)),
        config=CoincidentConstraintConfig(soft_confidence_threshold=0.5, hard_confidence_threshold=0.9),
    )

    constraints = engine.generate_constraints(document)
    updated_document = engine.apply_constraints(document)

    assert len(constraints) == 1
    constraint = constraints[0]
    assert constraint.type == "coincident"
    assert constraint.strength == 0.5
    assert constraint.confidence == pytest.approx(0.95)
    assert constraint.metadata["mode"] == "soft"
    assert constraint.metadata["locked_involved"] is True
    assert constraint.metadata["movable_anchor_ids"] == ["a1"]
    assert document.anchors[0].position == (20.0, 10.0)
    assert document.anchors[1].position == (20.01, 10.0)
    assert updated_document.anchors == document.anchors
    assert len(updated_document.constraints) == 1
    constraint_id = updated_document.constraints[0].constraint_id
    by_object_id = {obj.object_id: obj for obj in updated_document.objects}
    assert constraint_id in by_object_id["object_a"].constraints
    assert constraint_id in by_object_id["object_b"].constraints


def test_coincident_constraint_engine_filters_low_confidence_candidates() -> None:
    document = _base_document()
    for anchor in (
        Anchor(anchor_id="a1", path_id="path_a", position=(0.0, 0.0)),
        Anchor(anchor_id="b1", path_id="path_b", position=(0.15, 0.0)),
    ):
        document = add_anchor(document, anchor)

    engine = CoincidentConstraintEngine(
        snapping_engine=GlobalSnappingEngine(SnappingConfig(epsilon=0.2)),
    )

    assert engine.generate_constraints(document) == ()
    assert engine.apply_constraints(document).constraints == ()


def test_coincident_constraint_engine_skips_duplicate_existing_constraint_pair() -> None:
    document = _base_document()
    for anchor in (
        Anchor(anchor_id="a1", path_id="path_a", position=(5.0, 5.0)),
        Anchor(anchor_id="b1", path_id="path_b", position=(5.0, 5.0)),
    ):
        document = add_anchor(document, anchor)

    engine = CoincidentConstraintEngine(
        snapping_engine=GlobalSnappingEngine(SnappingConfig(epsilon=0.2)),
    )

    first_pass = engine.apply_constraints(document)
    second_pass = engine.apply_constraints(first_pass)

    assert len(first_pass.constraints) == 1
    assert second_pass == first_pass


def test_coincident_constraint_service_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/snapping.py")
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
