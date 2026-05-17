from __future__ import annotations

import ast
from pathlib import Path
from xml.etree import ElementTree as ET

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment, Style
from services.svg_exporter import SvgExporter
from tests.snapshot_utils import assert_text_snapshot


def _document() -> object:
    return create_document(
        document_id="doc_svg",
        width=120.0,
        height=100.0,
        coordinate_system=CoordinateSystem(
            y_axis="down",
            unit="px",
            precision=4,
            view_box=(0.0, 0.0, 120.0, 100.0),
        ),
    )


def _document_y_up() -> object:
    return create_document(
        document_id="doc_svg_up",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(
            y_axis="up",
            unit="px",
            precision=4,
            view_box=(0.0, 0.0, 100.0, 100.0),
        ),
    )


def _snapshot_svg_document() -> object:
    document = _document()
    document = add_path(
        document,
        VectorPath(
            path_id="p1",
            closed=False,
            segments=("line_seg", "bezier_seg"),
            style=Style(stroke_color=(10, 20, 30), stroke_width=2.0, fill_color=None),
        ),
    )
    document = add_segment(document, Segment("line_seg", "p1", "line", {"start": [10.0, 10.0], "end": [30.0, 10.0]}))
    document = add_segment(
        document,
        Segment(
            "bezier_seg",
            "p1",
            "bezier",
            {"start": [30.0, 10.0], "control1": [40.0, 0.0], "control2": [50.0, 20.0], "end": [60.0, 10.0]},
        ),
    )

    document = add_path(document, VectorPath(path_id="circle_path", closed=True, segments=("circle_seg",), style=Style(fill_color=(10, 120, 200))))
    document = add_path(document, VectorPath(path_id="arc_path", closed=False, segments=("arc_seg",), style=Style(stroke_color=(0, 0, 0), stroke_width=1.5, fill_color=None)))
    document = add_path(document, VectorPath(path_id="ellipse_path", closed=True, segments=("ellipse_seg",), style=Style(fill_color=(80, 40, 20))))
    document = add_segment(document, Segment("circle_seg", "circle_path", "circle", {"cx": 25.0, "cy": 25.0, "r": 10.0}))
    document = add_segment(document, Segment("arc_seg", "arc_path", "arc", {"cx": 60.0, "cy": 40.0, "r": 12.0, "start_angle": 0.0, "end_angle": 1.57079632679, "direction": "ccw"}))
    document = add_segment(document, Segment("ellipse_seg", "ellipse_path", "ellipse", {"cx": 90.0, "cy": 60.0, "rx": 16.0, "ry": 8.0, "rotation": 0.4}))
    return document


def _snapshot_compound_document() -> object:
    document = _document()
    outer = VectorPath(
        path_id="outer",
        closed=True,
        child_paths=("inner",),
        segments=("o1", "o2", "o3", "o4"),
        style=Style(
            fill_color=(255, 0, 0),
            fill_alpha=0.5,
            opacity=0.75,
            stroke_color=(32, 16, 8),
            stroke_width=1.25,
        ),
    )
    inner = VectorPath(
        path_id="inner",
        closed=True,
        parent_path="outer",
        segments=("i1", "i2", "i3", "i4"),
    )
    document = add_path(document, outer)
    document = add_path(document, inner)
    for segment in (
        Segment("o1", "outer", "line", {"start": [10.0, 10.0], "end": [110.0, 10.0]}),
        Segment("o2", "outer", "line", {"start": [110.0, 10.0], "end": [110.0, 90.0]}),
        Segment("o3", "outer", "line", {"start": [110.0, 90.0], "end": [10.0, 90.0]}),
        Segment("o4", "outer", "line", {"start": [10.0, 90.0], "end": [10.0, 10.0]}),
        Segment("i1", "inner", "line", {"start": [40.0, 35.0], "end": [80.0, 35.0]}),
        Segment("i2", "inner", "line", {"start": [80.0, 35.0], "end": [80.0, 65.0]}),
        Segment("i3", "inner", "line", {"start": [80.0, 65.0], "end": [40.0, 65.0]}),
        Segment("i4", "inner", "line", {"start": [40.0, 65.0], "end": [40.0, 35.0]}),
    ):
        document = add_segment(document, segment)
    return document


def test_svg_exporter_outputs_valid_svg_with_basic_line_and_bezier_path() -> None:
    document = _document()
    path = VectorPath(
        path_id="p1",
        closed=False,
        segments=("line_seg", "bezier_seg"),
        style=Style(stroke_color=(10, 20, 30), stroke_width=2.0, fill_color=None),
    )
    document = add_path(document, path)
    document = add_segment(document, Segment("line_seg", "p1", "line", {"start": [10.0, 10.0], "end": [30.0, 10.0]}))
    document = add_segment(
        document,
        Segment(
            "bezier_seg",
            "p1",
            "bezier",
            {"start": [30.0, 10.0], "control1": [40.0, 0.0], "control2": [50.0, 20.0], "end": [60.0, 10.0]},
        ),
    )

    svg = SvgExporter(pretty=True).export_document(document)
    root = ET.fromstring(svg)
    path_element = root.find("{http://www.w3.org/2000/svg}path")

    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.attrib["viewBox"] == "0 0 120 100"
    assert path_element is not None
    assert "M 10 10" in path_element.attrib["d"]
    assert "L 30 10" in path_element.attrib["d"]
    assert "C 40 0 50 20 60 10" in path_element.attrib["d"]
    assert path_element.attrib["stroke"] == "rgb(10,20,30)"
    assert path_element.attrib["stroke-width"] == "2"
    assert path_element.attrib["fill"] == "none"


def test_svg_exporter_supports_evenodd_compound_paths() -> None:
    document = _document()
    outer = VectorPath(
        path_id="outer",
        closed=True,
        child_paths=("inner",),
        segments=("o1", "o2", "o3", "o4"),
        style=Style(fill_color=(255, 0, 0), fill_alpha=0.5, opacity=0.75),
    )
    inner = VectorPath(
        path_id="inner",
        closed=True,
        parent_path="outer",
        segments=("i1", "i2", "i3", "i4"),
    )
    document = add_path(document, outer)
    document = add_path(document, inner)
    outer_segments = (
        Segment("o1", "outer", "line", {"start": [10.0, 10.0], "end": [110.0, 10.0]}),
        Segment("o2", "outer", "line", {"start": [110.0, 10.0], "end": [110.0, 90.0]}),
        Segment("o3", "outer", "line", {"start": [110.0, 90.0], "end": [10.0, 90.0]}),
        Segment("o4", "outer", "line", {"start": [10.0, 90.0], "end": [10.0, 10.0]}),
    )
    inner_segments = (
        Segment("i1", "inner", "line", {"start": [40.0, 35.0], "end": [80.0, 35.0]}),
        Segment("i2", "inner", "line", {"start": [80.0, 35.0], "end": [80.0, 65.0]}),
        Segment("i3", "inner", "line", {"start": [80.0, 65.0], "end": [40.0, 65.0]}),
        Segment("i4", "inner", "line", {"start": [40.0, 65.0], "end": [40.0, 35.0]}),
    )
    for segment in outer_segments + inner_segments:
        document = add_segment(document, segment)

    svg = SvgExporter().export_document(document)
    root = ET.fromstring(svg)
    path_element = root.find("{http://www.w3.org/2000/svg}path")

    assert path_element is not None
    assert path_element.attrib["fill-rule"] == "evenodd"
    assert path_element.attrib["fill"] == "rgb(255,0,0)"
    assert path_element.attrib["fill-opacity"] == "0.5"
    assert path_element.attrib["opacity"] == "0.75"
    assert path_element.attrib["d"].count("M ") == 2
    assert path_element.attrib["d"].count("Z") == 2


def test_svg_exporter_supports_circle_arc_ellipse_and_closed_path_commands() -> None:
    document = _document()
    circle_path = VectorPath(path_id="circle_path", closed=True, segments=("circle_seg",), style=Style(fill_color=(10, 120, 200)))
    arc_path = VectorPath(path_id="arc_path", closed=False, segments=("arc_seg",), style=Style(stroke_color=(0, 0, 0), stroke_width=1.5, fill_color=None))
    ellipse_path = VectorPath(path_id="ellipse_path", closed=True, segments=("ellipse_seg",), style=Style(fill_color=(80, 40, 20)))
    document = add_path(document, circle_path)
    document = add_path(document, arc_path)
    document = add_path(document, ellipse_path)
    document = add_segment(document, Segment("circle_seg", "circle_path", "circle", {"cx": 25.0, "cy": 25.0, "r": 10.0}))
    document = add_segment(document, Segment("arc_seg", "arc_path", "arc", {"cx": 60.0, "cy": 40.0, "r": 12.0, "start_angle": 0.0, "end_angle": 1.57079632679, "direction": "ccw"}))
    document = add_segment(document, Segment("ellipse_seg", "ellipse_path", "ellipse", {"cx": 90.0, "cy": 60.0, "rx": 16.0, "ry": 8.0, "rotation": 0.4}))

    svg = SvgExporter().export_document(document)
    root = ET.fromstring(svg)
    circle = root.find("{http://www.w3.org/2000/svg}circle")
    ellipse = root.find("{http://www.w3.org/2000/svg}ellipse")
    paths = root.findall("{http://www.w3.org/2000/svg}path")

    assert circle is not None
    assert circle.attrib["cx"] == "25"
    assert circle.attrib["cy"] == "25"
    assert circle.attrib["r"] == "10"

    assert ellipse is not None
    assert ellipse.attrib["cx"] == "90"
    assert ellipse.attrib["cy"] == "60"
    assert ellipse.attrib["rx"] == "16"
    assert ellipse.attrib["ry"] == "8"
    assert "rotate(" in ellipse.attrib["transform"]

    arc_path_element = next(element for element in paths if element.attrib["id"] == "arc_path")
    assert "A 12 12 0 0 1" in arc_path_element.attrib["d"]
    assert arc_path_element.attrib["stroke"] == "rgb(0,0,0)"
    assert arc_path_element.attrib["fill"] == "none"


def test_svg_exporter_keeps_viewbox_visible_when_y_axis_is_up() -> None:
    document = _document_y_up()
    path = VectorPath(
        path_id="up_path",
        closed=False,
        segments=("up_line",),
        style=Style(stroke_color=(0, 0, 0), stroke_width=1.0, fill_color=None),
    )
    document = add_path(document, path)
    document = add_segment(document, Segment("up_line", "up_path", "line", {"start": [0.0, 0.0], "end": [0.0, 10.0]}))

    svg = SvgExporter().export_document(document)
    root = ET.fromstring(svg)
    path_element = root.find("{http://www.w3.org/2000/svg}path")

    assert root.attrib["viewBox"] == "0 0 100 100"
    assert path_element is not None
    assert path_element.attrib["d"].startswith("M 0 100 L 0 90")


def test_svg_exporter_adds_default_visible_stroke_for_unstyled_centerline_path() -> None:
    document = _document()
    document = add_path(
        document,
        VectorPath(
            path_id="unstyled_centerline",
            closed=False,
            source="skeleton_contour",
            segments=("line_seg",),
        ),
    )
    document = add_segment(
        document,
        Segment("line_seg", "unstyled_centerline", "line", {"start": [10.0, 20.0], "end": [80.0, 20.0]}),
    )

    svg = SvgExporter().export_document(document, export_mode="centerline")
    root = ET.fromstring(svg)
    path_element = root.find("{http://www.w3.org/2000/svg}path")

    assert path_element is not None
    assert path_element.attrib["fill"] == "none"
    assert path_element.attrib["stroke"] == "#000000"
    assert path_element.attrib["stroke-width"] == "1"


def test_svg_exporter_adds_default_visible_stroke_for_unstyled_circle() -> None:
    document = _document()
    document = add_path(
        document,
        VectorPath(
            path_id="unstyled_circle",
            closed=True,
            source="skeleton_contour",
            segments=("circle_seg",),
        ),
    )
    document = add_segment(document, Segment("circle_seg", "unstyled_circle", "circle", {"cx": 25.0, "cy": 25.0, "r": 10.0}))

    svg = SvgExporter().export_document(document, export_mode="centerline")
    root = ET.fromstring(svg)
    circle = root.find("{http://www.w3.org/2000/svg}circle")

    assert circle is not None
    assert circle.attrib["fill"] == "none"
    assert circle.attrib["stroke"] == "#000000"
    assert circle.attrib["stroke-width"] == "1"


def test_svg_exporter_respects_explicit_fill_without_forcing_fallback_stroke() -> None:
    document = _document()
    document = add_path(
        document,
        VectorPath(
            path_id="filled_circle",
            closed=True,
            source="skeleton_contour",
            segments=("circle_seg",),
            style=Style(fill_color=(10, 120, 200)),
        ),
    )
    document = add_segment(document, Segment("circle_seg", "filled_circle", "circle", {"cx": 25.0, "cy": 25.0, "r": 10.0}))

    svg = SvgExporter().export_document(document, export_mode="centerline")
    root = ET.fromstring(svg)
    circle = root.find("{http://www.w3.org/2000/svg}circle")

    assert circle is not None
    assert circle.attrib["fill"] == "rgb(10,120,200)"
    assert circle.attrib["stroke"] == "none"


def test_svg_exporter_matches_golden_snapshots() -> None:
    exporter = SvgExporter(pretty=True)

    assert_text_snapshot(
        actual=exporter.export_document(_snapshot_svg_document()),
        snapshot_path=Path("tests/golden/svg/mixed_geometry.svg"),
    )
    assert_text_snapshot(
        actual=exporter.export_document(_snapshot_compound_document()),
        snapshot_path=Path("tests/golden/svg/compound_evenodd.svg"),
    )


def test_svg_exporter_has_no_forbidden_dependencies() -> None:
    source = open("services/svg_exporter.py", "r", encoding="utf-8").read()
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"openai", "anthropic", "PyQt5", "PyQt6", "ui"}
    assert imports.isdisjoint(forbidden_imports)
