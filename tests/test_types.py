import ast
from dataclasses import is_dataclass, replace
from pathlib import Path

from core.types import (
    Anchor,
    Constraint,
    CoordinateSystem,
    Object,
    Path as VectorPath,
    Segment,
    SegmentTypes,
    Style,
    VectorDocument,
    updated,
)


def test_core_dataclasses_are_instantiable() -> None:
    style = Style(fill_color=(1, 2, 3), stroke_color=(4, 5, 6))
    anchor = Anchor(
        anchor_id="anchor_1",
        path_id="path_1",
        position=(10, 20),
        continuity="smooth",
        shared_tangent=(0, 1),
        in_handle=(9, 19),
        out_handle=(11, 21),
    )
    segment = Segment(
        segment_id="segment_1",
        path_id="path_1",
        type="line",
        params={"length": 10},
        anchors=("anchor_1", "anchor_2"),
    )
    vector_path = VectorPath(
        path_id="path_1",
        object_id="object_1",
        closed=True,
        segments=("segment_1",),
        style=style,
    )
    vector_object = Object(
        object_id="object_1",
        type="shape",
        paths=("path_1",),
        constraints=("constraint_1",),
    )
    constraint = Constraint(
        constraint_id="constraint_1",
        type="coincident",
        targets=("anchor_1", "anchor_2"),
        strength=0.8,
        source="system",
        confidence=0.95,
    )
    coordinate_system = CoordinateSystem(scale={"px_to_mm": 0.1})
    document = VectorDocument(
        document_id="doc_1",
        width=800,
        height=600,
        coordinate_system=coordinate_system,
        objects=(vector_object,),
        paths=(vector_path,),
        segments=(segment,),
        anchors=(anchor,),
        constraints=(constraint,),
    )

    for value in (
        style,
        anchor,
        segment,
        vector_path,
        vector_object,
        constraint,
        coordinate_system,
        document,
    ):
        assert is_dataclass(value)


def test_segment_supports_required_types() -> None:
    for segment_type in SegmentTypes:
        segment = Segment(segment_id=f"{segment_type}_1", path_id="path_1", type=segment_type)
        assert segment.type == segment_type


def test_anchor_fields_are_preserved() -> None:
    anchor = Anchor(
        anchor_id="anchor_1",
        path_id="path_1",
        position=(1, 2),
        continuity="symmetric",
        shared_tangent=(0.5, 0.5),
        locked=True,
        in_handle=(0, 1),
        out_handle=(2, 3),
    )

    assert anchor.continuity == "symmetric"
    assert anchor.shared_tangent == (0.5, 0.5)
    assert anchor.locked is True
    assert anchor.in_handle == (0.0, 1.0)
    assert anchor.out_handle == (2.0, 3.0)


def test_constraint_fields_are_preserved() -> None:
    constraint = Constraint(
        constraint_id="constraint_1",
        type="tangent",
        targets=("segment_1", "segment_2"),
        strength=0.75,
        source="ai",
        confidence=0.9,
        locked=True,
    )

    assert constraint.type == "tangent"
    assert constraint.targets == ("segment_1", "segment_2")
    assert constraint.strength == 0.75
    assert constraint.source == "ai"
    assert constraint.confidence == 0.9
    assert constraint.locked is True


def test_replace_and_updated_do_not_mutate_original_object() -> None:
    original = Segment(
        segment_id="segment_1",
        path_id="path_1",
        type="arc",
        params={"radius": 5, "direction": "ccw"},
        anchors=("anchor_1", "anchor_2"),
    )

    replaced = replace(original, locked=True)
    modified = updated(original, params={**original.params, "radius": 7})

    assert original.locked is False
    assert replaced.locked is True
    assert original.params["radius"] == 5
    assert modified.params["radius"] == 7
    assert original.params["direction"] == "ccw"


def test_vector_document_snapshot_collections_are_not_polluted() -> None:
    anchor_ids = ["anchor_1", "anchor_2"]
    segment_params = {"radius": 5}
    scale = {"px_to_mm": 0.1}
    doc_metadata = {"source": "initial"}

    anchor = Anchor(anchor_id="anchor_1", path_id="path_1", position=(1, 2))
    segment = Segment(
        segment_id="segment_1",
        path_id="path_1",
        type="circle",
        params=segment_params,
        anchors=anchor_ids,
    )
    vector_path = VectorPath(path_id="path_1", segments=["segment_1"])
    vector_object = Object(object_id="object_1", type="shape", paths=["path_1"])
    coordinate_system = CoordinateSystem(scale=scale)
    document = VectorDocument(
        document_id="doc_1",
        width=100,
        height=200,
        coordinate_system=coordinate_system,
        objects=[vector_object],
        paths=[vector_path],
        segments=[segment],
        anchors=[anchor],
        metadata=doc_metadata,
    )

    anchor_ids.append("anchor_3")
    segment_params["radius"] = 99
    scale["px_to_mm"] = 2.0
    doc_metadata["source"] = "mutated"

    updated_document = updated(document, objects=document.objects + (Object(object_id="object_2", type="shape"),))

    assert document.segments[0].anchors == ("anchor_1", "anchor_2")
    assert document.segments[0].params["radius"] == 5
    assert document.coordinate_system.scale["px_to_mm"] == 0.1
    assert document.metadata["source"] == "initial"
    assert len(document.objects) == 1
    assert len(updated_document.objects) == 2


def test_core_types_has_no_forbidden_dependencies() -> None:
    source_path = Path("core/types.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {
        "cv2",
        "matplotlib",
        "PyQt5",
        "PyQt6",
        "openai",
        "anthropic",
        "services",
        "ui",
        "pathlib",
        "os",
    }

    assert imports.isdisjoint(forbidden_imports)
    assert "open(" not in source
