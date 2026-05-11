import ast
from pathlib import Path

import pytest

from core.document import add_anchor, add_constraint, add_object, add_path, add_segment, create_document, from_json, to_json
from core.types import Anchor, Constraint, CoordinateSystem, Object, Path as VectorPath, Segment, Style


def test_document_add_operations_and_json_round_trip() -> None:
    coordinate_system = CoordinateSystem(
        unit="px",
        y_axis="down",
        precision=4,
        view_box=(0.0, 0.0, 800.0, 600.0),
        scale={"px_to_mm": 0.2},
        metadata={"space": "vector"},
    )
    document = create_document(
        document_id="doc_1",
        width=800,
        height=600,
        coordinate_system=coordinate_system,
        metadata={"author": "codex"},
    )
    obj = Object(object_id="object_1", type="shape", semantic_label="outer_shape")
    path = VectorPath(
        path_id="path_1",
        object_id="object_1",
        closed=True,
        style=Style(fill_color=(12, 34, 56), opacity=0.8),
    )
    segment = Segment(
        segment_id="segment_1",
        path_id="path_1",
        type="line",
        params={"start": [0.0, 0.0], "end": [10.0, 10.0]},
        anchors=("anchor_1", "anchor_2"),
    )
    anchor = Anchor(
        anchor_id="anchor_1",
        path_id="path_1",
        position=(0.0, 0.0),
        out_handle=(1.0, 1.0),
    )
    constraint = Constraint(
        constraint_id="constraint_1",
        type="lock",
        targets=("object_1", "anchor_1"),
        strength=1.0,
        source="user",
        confidence=0.99,
    )

    original_document = document
    document = add_object(document, obj)
    document = add_path(document, path)
    document = add_segment(document, segment)
    document = add_anchor(document, anchor)
    document = add_constraint(document, constraint)

    assert original_document.objects == ()
    assert document.objects[0].paths == ("path_1",)
    assert document.paths[0].segments == ("segment_1",)
    assert document.objects[0].constraints == ("constraint_1",)
    assert document.anchors[0].anchor_id == "anchor_1"
    assert document.coordinate_system.scale["px_to_mm"] == 0.2
    assert document.metadata["author"] == "codex"

    payload = to_json(document)
    restored = from_json(payload)

    assert restored == document
    assert restored.paths[0].object_id == "object_1"
    assert restored.segments[0].anchors == ("anchor_1", "anchor_2")
    assert restored.constraints[0].targets == ("object_1", "anchor_1")
    assert restored.coordinate_system.view_box == (0.0, 0.0, 800.0, 600.0)


def test_document_add_requires_existing_parent_relationships() -> None:
    document = create_document(
        document_id="doc_2",
        width=100,
        height=100,
        coordinate_system=CoordinateSystem(),
    )

    with pytest.raises(ValueError):
        add_path(document, VectorPath(path_id="missing-object-path", object_id="missing_object"))

    with pytest.raises(ValueError):
        add_segment(document, Segment(segment_id="segment_1", path_id="missing_path", type="line"))

    with pytest.raises(ValueError):
        add_anchor(document, Anchor(anchor_id="anchor_1", path_id="missing_path", position=(0.0, 0.0)))


def test_core_document_has_no_forbidden_dependencies() -> None:
    source_path = Path("core/document.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic", "ui", "services"}

    assert imports.isdisjoint(forbidden_imports)
    assert "open(" not in source
