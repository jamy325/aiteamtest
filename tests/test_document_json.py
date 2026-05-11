import ast
import json
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
    obj = Object(
        object_id="object_1",
        type="shape",
        semantic_label="outer_shape",
        confidence=0.97,
        locked=True,
        metadata={"layer": "foreground"},
    )
    path = VectorPath(
        path_id="path_1",
        object_id="object_1",
        closed=True,
        style=Style(fill_color=(12, 34, 56), stroke_color=(90, 91, 92), opacity=0.8, stroke_width=1.25),
        topology_status="closed",
        locked=True,
        metadata={"path_kind": "outer"},
    )
    segment = Segment(
        segment_id="segment_1",
        path_id="path_1",
        type="line",
        params={"start": [0.1234, 0.5678], "end": [10.1234, 10.5678]},
        anchors=("anchor_1", "anchor_2"),
        confidence=0.88,
        locked=True,
        metadata={"segment_kind": "edge"},
    )
    anchor = Anchor(
        anchor_id="anchor_1",
        path_id="path_1",
        position=(0.1234, 0.5678),
        continuity="smooth",
        locked=True,
        out_handle=(1.1234, 1.5678),
        metadata={"anchor_role": "start"},
    )
    constraint = Constraint(
        constraint_id="constraint_1",
        type="lock",
        targets=("object_1", "anchor_1"),
        strength=1.0,
        source="user",
        confidence=0.99,
        locked=True,
        metadata={"constraint_kind": "manual"},
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
    payload_data = json.loads(payload)
    restored = from_json(payload)

    assert set(payload_data) == {
        "anchors",
        "constraints",
        "coordinate_system",
        "document_id",
        "height",
        "metadata",
        "objects",
        "paths",
        "segments",
        "width",
    }
    assert payload_data["coordinate_system"]["precision"] == 4
    assert payload_data["objects"][0]["locked"] is True
    assert payload_data["objects"][0]["confidence"] == pytest.approx(0.97)
    assert payload_data["paths"][0]["topology_status"] == "closed"
    assert payload_data["paths"][0]["style"]["stroke_width"] == pytest.approx(1.25)
    assert payload_data["segments"][0]["confidence"] == pytest.approx(0.88)
    assert payload_data["anchors"][0]["position"] == pytest.approx([0.1234, 0.5678])
    assert payload_data["constraints"][0]["locked"] is True

    assert restored == document
    assert len(restored.objects) == 1
    assert len(restored.paths) == 1
    assert len(restored.segments) == 1
    assert len(restored.anchors) == 1
    assert len(restored.constraints) == 1
    assert restored.paths[0].object_id == "object_1"
    assert restored.segments[0].anchors == ("anchor_1", "anchor_2")
    assert restored.constraints[0].targets == ("object_1", "anchor_1")
    assert restored.coordinate_system.view_box == (0.0, 0.0, 800.0, 600.0)
    assert restored.objects[0].locked is True
    assert restored.objects[0].confidence == pytest.approx(0.97)
    assert restored.paths[0].locked is True
    assert restored.paths[0].topology_status == "closed"
    assert restored.paths[0].style is not None
    assert restored.paths[0].style.fill_color == (12, 34, 56)
    assert restored.paths[0].style.stroke_color == (90, 91, 92)
    assert restored.paths[0].style.stroke_width == pytest.approx(1.25)
    assert restored.segments[0].locked is True
    assert restored.segments[0].confidence == pytest.approx(0.88)
    assert restored.anchors[0].locked is True
    assert restored.anchors[0].position == pytest.approx((0.1234, 0.5678))
    assert restored.constraints[0].confidence == pytest.approx(0.99)
    assert restored.constraints[0].locked is True
    assert restored.coordinate_system.precision == 4
    assert restored.anchors[0].position[0] == pytest.approx(round(restored.anchors[0].position[0], restored.coordinate_system.precision))
    assert restored.anchors[0].position[1] == pytest.approx(round(restored.anchors[0].position[1], restored.coordinate_system.precision))


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
