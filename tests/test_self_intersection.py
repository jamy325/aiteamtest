import ast
from pathlib import Path

import pytest

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.self_intersection import SelfIntersectionConfig, SelfIntersectionDetector


def _build_document(path: VectorPath, segments: tuple[Segment, ...]):
    document = create_document(
        document_id=f"doc_{path.path_id}",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(),
    )
    document = add_path(document, path)
    for segment in segments:
        document = add_segment(document, segment)
    return document


def test_self_intersection_detector_detects_line_path_crossing() -> None:
    path = VectorPath(path_id="path_line_cross", closed=False, topology_status="open")
    segments = (
        Segment("seg_1", "path_line_cross", "line", params={"start": [0.0, 0.0], "end": [10.0, 10.0]}),
        Segment("seg_2", "path_line_cross", "line", params={"start": [10.0, 10.0], "end": [0.0, 10.0]}),
        Segment("seg_3", "path_line_cross", "line", params={"start": [0.0, 10.0], "end": [10.0, 0.0]}),
    )
    document = _build_document(path, segments)
    detector = SelfIntersectionDetector()

    result = detector.detect_path_self_intersections(document, "path_line_cross")

    assert result.self_intersection_count == 1
    assert result.self_intersection_points[0][0] == pytest.approx(5.0)
    assert result.self_intersection_points[0][1] == pytest.approx(5.0)
    assert result.document.paths[0].self_intersection_count == 1
    assert result.document.paths[0].topology_status == "self_intersected"
    assert result.document.paths[0].metadata["self_intersection_points"] == [[5.0, 5.0]]
    assert document.paths[0].self_intersection_count == 0
    assert document.paths[0].metadata == {}


def test_self_intersection_detector_detects_non_adjacent_collinear_overlap() -> None:
    path = VectorPath(path_id="path_overlap", closed=False, topology_status="open")
    segments = (
        Segment("seg_1", "path_overlap", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]}),
        Segment("seg_2", "path_overlap", "line", params={"start": [20.0, 0.0], "end": [20.0, 10.0]}),
        Segment("seg_3", "path_overlap", "line", params={"start": [5.0, 0.0], "end": [15.0, 0.0]}),
    )
    document = _build_document(path, segments)
    detector = SelfIntersectionDetector()

    result = detector.detect_path_self_intersections(document, "path_overlap")

    assert result.self_intersection_count == 1
    assert 5.0 <= result.self_intersection_points[0][0] <= 10.0
    assert result.self_intersection_points[0][1] == pytest.approx(0.0)
    assert result.document.paths[0].topology_status == "self_intersected"


def test_self_intersection_detector_ignores_adjacent_collinear_continuous_segments() -> None:
    path = VectorPath(path_id="path_adjacent_collinear", closed=False, topology_status="open")
    segments = (
        Segment("seg_1", "path_adjacent_collinear", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]}),
        Segment("seg_2", "path_adjacent_collinear", "line", params={"start": [10.0, 0.0], "end": [20.0, 0.0]}),
    )
    document = _build_document(path, segments)
    detector = SelfIntersectionDetector()

    result = detector.detect_path_self_intersections(document, "path_adjacent_collinear")

    assert result.self_intersection_count == 0
    assert result.self_intersection_points == ()
    assert result.document.paths[0].topology_status == "open"


def test_self_intersection_detector_ignores_adjacent_shared_endpoints_on_closed_path() -> None:
    path = VectorPath(path_id="path_closed_rect", closed=True, topology_status="closed")
    segments = (
        Segment("seg_1", "path_closed_rect", "line", params={"start": [10.0, 10.0], "end": [30.0, 10.0]}),
        Segment("seg_2", "path_closed_rect", "line", params={"start": [30.0, 10.0], "end": [30.0, 30.0]}),
        Segment("seg_3", "path_closed_rect", "line", params={"start": [30.0, 30.0], "end": [10.0, 30.0]}),
        Segment("seg_4", "path_closed_rect", "line", params={"start": [10.0, 30.0], "end": [10.0, 10.0]}),
    )
    document = _build_document(path, segments)
    detector = SelfIntersectionDetector()

    result = detector.detect_path_self_intersections(document, "path_closed_rect")

    assert result.self_intersection_count == 0
    assert result.self_intersection_points == ()
    assert result.document.paths[0].self_intersection_count == 0
    assert result.document.paths[0].topology_status == "closed"


def test_self_intersection_detector_detects_arc_path_intersections() -> None:
    path = VectorPath(path_id="path_arc_cross", closed=False, topology_status="open")
    segments = (
        Segment("seg_1", "path_arc_cross", "line", params={"start": [-6.0, 0.0], "end": [6.0, 0.0]}),
        Segment(
            "seg_2",
            "path_arc_cross",
            "arc",
            params={
                "cx": 0.0,
                "cy": 0.0,
                "r": 5.0,
                "start_angle": 0.0,
                "end_angle": 0.0,
                "direction": "ccw",
            },
        ),
    )
    document = _build_document(path, segments)
    detector = SelfIntersectionDetector(SelfIntersectionConfig(max_chord_error=0.05, min_segments_per_arc=16, max_segments_per_arc=128))

    result = detector.detect_path_self_intersections(document, "path_arc_cross")

    assert result.self_intersection_count == 2
    assert any(point[0] == pytest.approx(-5.0, abs=0.2) and point[1] == pytest.approx(0.0, abs=0.2) for point in result.self_intersection_points)
    assert any(point[0] == pytest.approx(5.0, abs=0.2) and point[1] == pytest.approx(0.0, abs=0.2) for point in result.self_intersection_points)


def test_self_intersection_detector_sampling_density_improves_arc_detection() -> None:
    path = VectorPath(path_id="path_arc_precision", closed=False, topology_status="open")
    segments = (
        Segment("seg_1", "path_arc_precision", "line", params={"start": [1.0, 10.0], "end": [10.0, 1.0]}),
        Segment(
            "seg_2",
            "path_arc_precision",
            "arc",
            params={
                "cx": 0.0,
                "cy": 0.0,
                "r": 10.0,
                "start_angle": 0.0,
                "end_angle": 90.0,
                "direction": "ccw",
            },
        ),
    )
    document = _build_document(path, segments)
    coarse = SelfIntersectionDetector(
        SelfIntersectionConfig(max_chord_error=50.0, min_segments_per_arc=1, max_segments_per_arc=1),
    )
    dense = SelfIntersectionDetector(
        SelfIntersectionConfig(max_chord_error=0.05, min_segments_per_arc=16, max_segments_per_arc=128),
    )

    coarse_result = coarse.detect_path_self_intersections(document, "path_arc_precision")
    dense_result = dense.detect_path_self_intersections(document, "path_arc_precision")

    assert coarse_result.self_intersection_count == 0
    assert dense_result.self_intersection_count >= 1


def test_self_intersection_detector_detects_bezier_path_intersections() -> None:
    path = VectorPath(path_id="path_bezier_cross", closed=False, topology_status="open")
    segments = (
        Segment("seg_1", "path_bezier_cross", "line", params={"start": [0.0, 7.0], "end": [10.0, 7.0]}),
        Segment(
            "seg_2",
            "path_bezier_cross",
            "bezier",
            params={
                "start": [0.0, 0.0],
                "control1": [0.0, 10.0],
                "control2": [10.0, 10.0],
                "end": [10.0, 0.0],
            },
        ),
    )
    document = _build_document(path, segments)
    detector = SelfIntersectionDetector(SelfIntersectionConfig(bezier_segments=48))

    result = detector.detect_path_self_intersections(document, "path_bezier_cross")

    assert result.self_intersection_count >= 2
    assert all(point[1] == pytest.approx(7.0, abs=0.35) for point in result.self_intersection_points)


def test_self_intersection_detector_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/self_intersection.py")
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
