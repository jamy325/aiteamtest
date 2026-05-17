from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.dxf_exporter import DxfExporter
from tests.snapshot_utils import assert_text_snapshot


def _parse_entities(payload: str) -> list[dict[str, list[str]]]:
    lines = payload.splitlines()
    entities_start = lines.index("ENTITIES") + 1
    entities_end = lines.index("ENDSEC", entities_start)
    entity_lines = lines[entities_start:entities_end]

    entities: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] | None = None
    for index in range(0, len(entity_lines) - 1, 2):
        code = entity_lines[index]
        value = entity_lines[index + 1]
        if code == "0":
            if current is not None:
                entities.append(current)
            current = {"0": [value]}
            continue
        assert current is not None
        current.setdefault(code, []).append(value)
    if current is not None:
        entities.append(current)
    return entities


def _document(*, unit: str = "px", y_axis: str = "down", px_to_mm: float = 0.5) -> object:
    return create_document(
        document_id="doc_dxf",
        width=100.0,
        height=80.0,
        coordinate_system=CoordinateSystem(
            unit=unit,
            y_axis=y_axis,
            precision=4,
            view_box=(0.0, 0.0, 100.0, 80.0),
            scale={"px_to_mm": px_to_mm},
        ),
    )


def _snapshot_dxf_document() -> object:
    document = _document(unit="px", y_axis="down", px_to_mm=0.5)
    document = add_path(document, VectorPath(path_id="p_line", segments=("s_line",)))
    document = add_path(document, VectorPath(path_id="p_arc", segments=("s_arc",)))
    document = add_path(document, VectorPath(path_id="p_circle", segments=("s_circle",)))
    document = add_path(document, VectorPath(path_id="p_ellipse", segments=("s_ellipse",)))
    document = add_segment(document, Segment("s_line", "p_line", "line", {"start": [10.0, 20.0], "end": [30.0, 20.0]}))
    document = add_segment(document, Segment("s_arc", "p_arc", "arc", {"cx": 50.0, "cy": 50.0, "r": 10.0, "start_angle": 0.0, "end_angle": math.pi / 2.0, "direction": "ccw"}))
    document = add_segment(document, Segment("s_circle", "p_circle", "circle", {"cx": 30.0, "cy": 18.0, "r": 7.0}))
    document = add_segment(document, Segment("s_ellipse", "p_ellipse", "ellipse", {"cx": 40.0, "cy": 25.0, "rx": 10.0, "ry": 4.0, "rotation": 0.25}))
    return document


def _snapshot_dxf_document_y_up() -> object:
    document = _document(unit="mm", y_axis="up", px_to_mm=1.0)
    document = add_path(document, VectorPath(path_id="p_arc", segments=("s_arc",)))
    document = add_path(document, VectorPath(path_id="p_circle", segments=("s_circle",)))
    document = add_segment(document, Segment("s_arc", "p_arc", "arc", {"cx": 10.0, "cy": 12.0, "r": 5.0, "start_angle": 0.0, "end_angle": 1.57079632679, "direction": "ccw"}))
    document = add_segment(document, Segment("s_circle", "p_circle", "circle", {"cx": 30.0, "cy": 18.0, "r": 7.0}))
    return document


def test_dxf_exporter_outputs_valid_ascii_structure() -> None:
    document = _document()
    path = VectorPath(path_id="p_line", segments=("s_line",))
    document = add_path(document, path)
    document = add_segment(document, Segment("s_line", "p_line", "line", {"start": [10.0, 20.0], "end": [30.0, 20.0]}))

    payload = DxfExporter().export_document(document)

    assert payload.startswith("0\nSECTION\n2\nHEADER\n")
    assert "\n2\nENTITIES\n" in payload
    assert payload.endswith("0\nEOF\n")


def test_dxf_exporter_applies_px_to_mm_and_y_axis_flip_for_line() -> None:
    document = _document(unit="px", y_axis="down", px_to_mm=0.5)
    path = VectorPath(path_id="p_line", segments=("s_line",))
    document = add_path(document, path)
    document = add_segment(document, Segment("s_line", "p_line", "line", {"start": [10.0, 20.0], "end": [30.0, 20.0]}))

    entities = _parse_entities(DxfExporter().export_document(document))
    line = entities[0]

    assert line["0"] == ["LINE"]
    assert line["10"] == ["5"]
    assert line["20"] == ["30"]
    assert line["11"] == ["15"]
    assert line["21"] == ["30"]


def test_dxf_exporter_writes_closed_rectangle_as_closed_lwpolyline() -> None:
    document = _document(unit="px", y_axis="down", px_to_mm=1.0)
    path = VectorPath(path_id="rect", segments=("s1", "s2", "s3", "s4"), closed=True, topology_status="closed")
    document = add_path(document, path)
    for segment in (
        Segment("s1", "rect", "line", {"start": [10.0, 10.0], "end": [90.0, 10.0]}),
        Segment("s2", "rect", "line", {"start": [90.0, 10.0], "end": [90.0, 90.0]}),
        Segment("s3", "rect", "line", {"start": [90.0, 90.0], "end": [10.0, 90.0]}),
        Segment("s4", "rect", "line", {"start": [10.0, 90.0], "end": [10.0, 10.0]}),
    ):
        document = add_segment(document, segment)

    entities = _parse_entities(DxfExporter().export_document(document))

    assert len(entities) == 1
    polyline = entities[0]
    assert polyline["0"] == ["LWPOLYLINE"]
    assert polyline["70"] == ["1"]
    assert polyline["90"] == ["4"]


def test_dxf_exporter_writes_open_line_path_as_open_lwpolyline() -> None:
    document = _document(unit="px", y_axis="down", px_to_mm=1.0)
    path = VectorPath(path_id="open_poly", segments=("s1", "s2", "s3"), closed=False, topology_status="open")
    document = add_path(document, path)
    for segment in (
        Segment("s1", "open_poly", "line", {"start": [10.0, 10.0], "end": [20.0, 10.0]}),
        Segment("s2", "open_poly", "line", {"start": [20.0, 10.0], "end": [20.0, 20.0]}),
        Segment("s3", "open_poly", "line", {"start": [20.0, 20.0], "end": [30.0, 20.0]}),
    ):
        document = add_segment(document, segment)

    entities = _parse_entities(DxfExporter().export_document(document))

    assert len(entities) == 1
    polyline = entities[0]
    assert polyline["0"] == ["LWPOLYLINE"]
    assert polyline["70"] == ["0"]
    assert polyline["90"] == ["4"]


def test_dxf_exporter_falls_back_when_closed_line_path_is_discontinuous() -> None:
    document = _document(unit="px", y_axis="down", px_to_mm=1.0)
    path = VectorPath(path_id="broken", segments=("s1", "s2", "s3", "s4"), closed=True, topology_status="closed")
    document = add_path(document, path)
    for segment in (
        Segment("s1", "broken", "line", {"start": [10.0, 10.0], "end": [90.0, 10.0]}),
        Segment("s2", "broken", "line", {"start": [90.0, 10.0], "end": [90.0, 90.0]}),
        Segment("s3", "broken", "line", {"start": [90.0, 90.0], "end": [10.0, 90.0]}),
        Segment("s4", "broken", "line", {"start": [12.0, 90.0], "end": [10.0, 10.0]}),
    ):
        document = add_segment(document, segment)

    payload = DxfExporter().export_document(document)
    entities = _parse_entities(payload)
    report = DxfExporter().export_report(document)

    assert len(entities) == 4
    assert all(entity["0"] == ["LINE"] for entity in entities)
    assert "not continuous enough for closed LWPOLYLINE fallback" in payload
    assert report["warnings"]


def test_dxf_exporter_supports_arc_and_circle_entities() -> None:
    document = _document(unit="mm", y_axis="up")
    arc_path = VectorPath(path_id="p_arc", segments=("s_arc",))
    circle_path = VectorPath(path_id="p_circle", segments=("s_circle",))
    document = add_path(document, arc_path)
    document = add_path(document, circle_path)
    document = add_segment(document, Segment("s_arc", "p_arc", "arc", {"cx": 10.0, "cy": 12.0, "r": 5.0, "start_angle": 0.0, "end_angle": 1.57079632679, "direction": "ccw"}))
    document = add_segment(document, Segment("s_circle", "p_circle", "circle", {"cx": 30.0, "cy": 18.0, "r": 7.0}))

    entities = _parse_entities(DxfExporter().export_document(document))
    arc = next(entity for entity in entities if entity["0"] == ["ARC"])
    circle = next(entity for entity in entities if entity["0"] == ["CIRCLE"])

    assert arc["10"] == ["10"]
    assert arc["20"] == ["12"]
    assert arc["40"] == ["5"]
    assert arc["50"] == ["0"]
    assert arc["51"] == ["90"]
    assert circle["10"] == ["30"]
    assert circle["20"] == ["18"]
    assert circle["40"] == ["7"]


def test_dxf_exporter_preserves_ccw_quarter_arc_after_y_down_flip() -> None:
    document = _document(unit="px", y_axis="down", px_to_mm=1.0)
    path = VectorPath(path_id="p_arc_down_ccw", segments=("s_arc_down_ccw",))
    document = add_path(document, path)
    document = add_segment(
        document,
        Segment(
            "s_arc_down_ccw",
            "p_arc_down_ccw",
            "arc",
            {"cx": 50.0, "cy": 50.0, "r": 10.0, "start_angle": 0.0, "end_angle": math.pi / 2.0, "direction": "ccw"},
        ),
    )

    entities = _parse_entities(DxfExporter().export_document(document))
    arc = entities[0]

    assert arc["50"] == ["270"]
    assert arc["51"] == ["0"]


def test_dxf_exporter_preserves_cw_arc_after_y_down_flip() -> None:
    document = _document(unit="px", y_axis="down", px_to_mm=1.0)
    path = VectorPath(path_id="p_arc_down_cw", segments=("s_arc_down_cw",))
    document = add_path(document, path)
    document = add_segment(
        document,
        Segment(
            "s_arc_down_cw",
            "p_arc_down_cw",
            "arc",
            {"cx": 50.0, "cy": 50.0, "r": 10.0, "start_angle": math.pi, "end_angle": math.pi / 2.0, "direction": "cw"},
        ),
    )

    entities = _parse_entities(DxfExporter().export_document(document))
    arc = entities[0]

    assert arc["50"] == ["180"]
    assert arc["51"] == ["270"]


def test_dxf_exporter_uses_polyline_fallback_for_ellipse() -> None:
    document = _document(unit="mm", y_axis="up")
    path = VectorPath(path_id="p_ellipse", segments=("s_ellipse",))
    document = add_path(document, path)
    document = add_segment(document, Segment("s_ellipse", "p_ellipse", "ellipse", {"cx": 40.0, "cy": 25.0, "rx": 10.0, "ry": 4.0, "rotation": 0.25}))

    entities = _parse_entities(DxfExporter().export_document(document))
    polyline = entities[0]

    assert polyline["0"] == ["LWPOLYLINE"]
    assert int(polyline["90"][0]) >= 9
    assert polyline["70"] == ["1"]
    assert len(polyline["10"]) == len(polyline["20"])


def test_dxf_exporter_can_derive_flip_span_from_document_dimensions() -> None:
    document = create_document(
        document_id="doc_no_viewbox",
        width=100.0,
        height=80.0,
        coordinate_system=CoordinateSystem(unit="px", y_axis="down", precision=4, scale={"px_to_mm": 0.5}),
    )
    path = VectorPath(path_id="p_line", segments=("s_line",))
    document = add_path(document, path)
    document = add_segment(document, Segment("s_line", "p_line", "line", {"start": [0.0, 0.0], "end": [0.0, 10.0]}))

    entities = _parse_entities(DxfExporter().export_document(document))
    line = entities[0]

    assert line["10"] == ["0"]
    assert line["20"] == ["40"]
    assert line["11"] == ["0"]
    assert line["21"] == ["35"]


def test_dxf_exporter_matches_golden_snapshots() -> None:
    exporter = DxfExporter()

    assert_text_snapshot(
        actual=exporter.export_document(_snapshot_dxf_document()),
        snapshot_path=Path("tests/golden/dxf/mixed_geometry_y_down_scaled.dxf"),
    )
    assert_text_snapshot(
        actual=exporter.export_document(_snapshot_dxf_document_y_up()),
        snapshot_path=Path("tests/golden/dxf/basic_arc_circle_y_up.dxf"),
    )


def test_dxf_exporter_has_no_forbidden_dependencies() -> None:
    source = Path("services/dxf_exporter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"openai", "anthropic", "PyQt5", "PyQt6", "ui"}
    assert imports.isdisjoint(forbidden_imports)
