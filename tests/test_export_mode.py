from __future__ import annotations

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.dxf_exporter import DxfExporter
from services.svg_exporter import SvgExporter
from tests.test_dxf_exporter import _parse_entities


def _mixed_document() -> object:
    document = create_document(
        document_id="doc_export_mode",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(
            y_axis="down",
            unit="px",
            precision=4,
            view_box=(0.0, 0.0, 100.0, 100.0),
            scale={"px_to_mm": 1.0},
        ),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="binary_outer",
            segments=("b1", "b2", "b3", "b4"),
            closed=True,
            source="binary_contour",
        ),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="binary_inner",
            segments=("i1", "i2", "i3", "i4"),
            closed=True,
            source="binary_contour",
        ),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="centerline",
            segments=("c1", "c2", "c3", "c4"),
            closed=True,
            source="skeleton_contour",
        ),
    )
    segments = (
        Segment("b1", "binary_outer", "line", {"start": [10.0, 10.0], "end": [90.0, 10.0]}),
        Segment("b2", "binary_outer", "line", {"start": [90.0, 10.0], "end": [90.0, 90.0]}),
        Segment("b3", "binary_outer", "line", {"start": [90.0, 90.0], "end": [10.0, 90.0]}),
        Segment("b4", "binary_outer", "line", {"start": [10.0, 90.0], "end": [10.0, 10.0]}),
        Segment("i1", "binary_inner", "line", {"start": [25.0, 25.0], "end": [75.0, 25.0]}),
        Segment("i2", "binary_inner", "line", {"start": [75.0, 25.0], "end": [75.0, 75.0]}),
        Segment("i3", "binary_inner", "line", {"start": [75.0, 75.0], "end": [25.0, 75.0]}),
        Segment("i4", "binary_inner", "line", {"start": [25.0, 75.0], "end": [25.0, 25.0]}),
        Segment("c1", "centerline", "line", {"start": [20.0, 20.0], "end": [80.0, 20.0]}),
        Segment("c2", "centerline", "line", {"start": [80.0, 20.0], "end": [80.0, 80.0]}),
        Segment("c3", "centerline", "line", {"start": [80.0, 80.0], "end": [20.0, 80.0]}),
        Segment("c4", "centerline", "line", {"start": [20.0, 80.0], "end": [20.0, 20.0]}),
    )
    for segment in segments:
        document = add_segment(document, segment)
    return document


def test_svg_exporter_supports_outline_centerline_and_all_debug_modes() -> None:
    document = _mixed_document()
    exporter = SvgExporter()

    outline_svg = exporter.export_document(document, export_mode="outline")
    centerline_svg = exporter.export_document(document, export_mode="centerline")
    all_debug_svg = exporter.export_document(document, export_mode="all_debug")
    outline_report = exporter.export_report(document, export_mode="outline")
    centerline_report = exporter.export_report(document, export_mode="centerline")
    all_debug_report = exporter.export_report(document, export_mode="all_debug")

    assert outline_svg.count("<path") == 2
    assert centerline_svg.count("<path") == 1
    assert all_debug_svg.count("<path") == 3
    assert outline_report["exported_path_count"] == 2
    assert outline_report["skipped_path_count"] == 1
    assert outline_report["exported_by_source"] == {"binary_contour": 2}
    assert centerline_report["exported_by_source"] == {"skeleton_contour": 1}
    assert all_debug_report["exported_by_source"] == {"binary_contour": 2, "skeleton_contour": 1}


def test_dxf_exporter_supports_outline_centerline_and_all_debug_modes() -> None:
    document = _mixed_document()
    exporter = DxfExporter()

    outline_entities = _parse_entities(exporter.export_document(document, export_mode="outline"))
    centerline_entities = _parse_entities(exporter.export_document(document, export_mode="centerline"))
    all_debug_entities = _parse_entities(exporter.export_document(document, export_mode="all_debug"))
    outline_report = exporter.export_report(document, export_mode="outline")
    centerline_report = exporter.export_report(document, export_mode="centerline")
    all_debug_report = exporter.export_report(document, export_mode="all_debug")

    assert len(outline_entities) == 2
    assert len(centerline_entities) == 1
    assert len(all_debug_entities) == 3
    assert all(entity["0"] == ["LWPOLYLINE"] for entity in outline_entities)
    assert all(entity["0"] == ["LWPOLYLINE"] for entity in centerline_entities)
    assert all(entity["0"] == ["LWPOLYLINE"] for entity in all_debug_entities)
    assert outline_report["exported_by_source"] == {"binary_contour": 2}
    assert centerline_report["exported_by_source"] == {"skeleton_contour": 1}
    assert all_debug_report["exported_by_source"] == {"binary_contour": 2, "skeleton_contour": 1}
    assert outline_report["entity_counts"]["LWPOLYLINE"] == 2
    assert centerline_report["entity_counts"]["LWPOLYLINE"] == 1
    assert all_debug_report["entity_counts"]["LWPOLYLINE"] == 3


def test_exporters_report_warning_when_export_mode_has_no_matching_paths() -> None:
    document = create_document(
        document_id="doc_export_empty",
        width=50.0,
        height=50.0,
        coordinate_system=CoordinateSystem(view_box=(0.0, 0.0, 50.0, 50.0)),
    )
    document = add_path(
        document,
        VectorPath(path_id="only_binary", segments=("s1",), closed=False, source="binary_contour"),
    )
    document = add_segment(document, Segment("s1", "only_binary", "line", {"start": [0.0, 0.0], "end": [10.0, 0.0]}))

    svg_exporter = SvgExporter()
    dxf_exporter = DxfExporter()
    svg_report = svg_exporter.export_report(document, export_mode="centerline")
    dxf_report = dxf_exporter.export_report(document, export_mode="centerline")
    svg_payload = svg_exporter.export_document(document, export_mode="centerline")
    dxf_payload = dxf_exporter.export_document(document, export_mode="centerline")

    assert svg_report["exported_path_count"] == 0
    assert dxf_report["exported_path_count"] == 0
    assert svg_report["warning"] is not None
    assert dxf_report["warning"] is not None
    assert "warning: no paths selected for export_mode=centerline" in svg_payload
    assert "warning: no paths selected for export_mode=centerline" in dxf_payload
