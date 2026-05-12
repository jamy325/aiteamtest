import ast
import json
from pathlib import Path

from core.document import add_anchor, add_constraint, add_object, add_path, add_segment, create_document, from_json
from core.types import Anchor, Constraint, CoordinateSystem, Object, Path as VectorPath, Segment
from services.json_exporter import JsonExporter


def _document_fixture():
    document = create_document(
        document_id="doc_export",
        width=320.0,
        height=240.0,
        coordinate_system=CoordinateSystem(
            precision=4,
            view_box=(0.0, 0.0, 320.0, 240.0),
            scale={"px_to_mm": 0.2},
            metadata={"space": "vector"},
        ),
        metadata={"source": "unit-test"},
    )
    obj = Object(object_id="object_1", type="shape", paths=(), constraints=(), metadata={"layer": "foreground"})
    path = VectorPath(
        path_id="path_1",
        object_id="object_1",
        closed=True,
        source="binary_contour",
        topology_status="closed",
    )
    segment = Segment(
        segment_id="segment_1",
        path_id="path_1",
        type="line",
        params={"start": [0.0, 0.0], "end": [10.0, 0.0]},
        anchors=("anchor_1", "anchor_2"),
    )
    anchor_1 = Anchor(anchor_id="anchor_1", path_id="path_1", position=(0.0, 0.0))
    anchor_2 = Anchor(anchor_id="anchor_2", path_id="path_1", position=(10.0, 0.0))
    constraint = Constraint(
        constraint_id="constraint_1",
        type="coincident",
        targets=("anchor_1", "anchor_2"),
        source="system",
    )

    document = add_object(document, obj)
    document = add_path(document, path)
    document = add_anchor(document, anchor_1)
    document = add_anchor(document, anchor_2)
    document = add_segment(document, segment)
    document = add_constraint(document, constraint)
    return document


def test_json_exporter_exports_formatted_json_without_mutating_document() -> None:
    document = _document_fixture()
    original_document = document
    exporter = JsonExporter(indent=2, sort_keys=True)

    payload = exporter.export_document(document)
    payload_data = json.loads(payload)

    assert document == original_document
    assert payload.startswith("{\n")
    assert '\n  "anchors": [' in payload
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
    assert payload_data["objects"][0]["object_id"] == "object_1"
    assert payload_data["paths"][0]["path_id"] == "path_1"
    assert payload_data["segments"][0]["segment_id"] == "segment_1"
    assert payload_data["anchors"][0]["anchor_id"] == "anchor_1"
    assert payload_data["constraints"][0]["constraint_id"] == "constraint_1"


def test_json_exporter_output_round_trips_back_to_vector_document() -> None:
    document = _document_fixture()
    exporter = JsonExporter(indent=4, sort_keys=True)

    payload = exporter.export_document(document)
    restored = from_json(payload)
    exported_dict = exporter.export_to_dict(document)

    assert restored == document
    assert exported_dict["document_id"] == "doc_export"
    assert exported_dict["metadata"]["source"] == "unit-test"
    assert exported_dict["objects"][0]["paths"] == ["path_1"]
    assert exported_dict["paths"][0]["segments"] == ["segment_1"]


def test_json_exporter_round_trips_tuple_like_segment_params() -> None:
    document = create_document(
        document_id="doc_tuple_export",
        width=50.0,
        height=50.0,
        coordinate_system=CoordinateSystem(),
    )
    path = VectorPath(path_id="path_tuple_export")
    segment = Segment(
        segment_id="segment_tuple_export",
        path_id="path_tuple_export",
        type="bezier",
        params={
            "start": (0.0, 0.0),
            "control_points": ((1.0, 1.0), (2.0, 2.0)),
            "end": (3.0, 3.0),
        },
    )

    document = add_path(document, path)
    document = add_segment(document, segment)
    exporter = JsonExporter()

    payload = exporter.export_document(document)
    restored = from_json(payload)

    assert document.segments[0].params == {
        "start": [0.0, 0.0],
        "control_points": [[1.0, 1.0], [2.0, 2.0]],
        "end": [3.0, 3.0],
    }
    assert restored == document


def test_json_exporter_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/json_exporter.py")
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
