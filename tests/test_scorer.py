import ast
from pathlib import Path

import pytest

from core.document import add_anchor, add_path, add_segment, create_document
from core.types import Anchor, CoordinateSystem, Path as VectorPath, Segment
from services.edge_error import EdgeErrorResult
from services.scorer import Scorer


def _build_document(
    *,
    coordinate_system: CoordinateSystem | None = None,
    path: VectorPath,
    anchors: tuple[Anchor, ...] = (),
    segments: tuple[Segment, ...] = (),
):
    document = create_document(
        document_id=f"doc_{path.path_id}",
        width=100.0,
        height=100.0,
        coordinate_system=coordinate_system or CoordinateSystem(),
    )
    document = add_path(document, path)
    for anchor in anchors:
        document = add_anchor(document, anchor)
    for segment in segments:
        document = add_segment(document, segment)
    return document


def test_scorer_reads_edge_error_breakdown_and_total_is_lower_for_better_result() -> None:
    path = VectorPath(path_id="path_edge", topology_status="open")
    segment = Segment("seg_1", "path_edge", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]})
    document = _build_document(path=path, segments=(segment,))
    scorer = Scorer()
    good_edge = EdgeErrorResult(0.0, 0.0, 0.0, 2, 2)
    bad_edge = EdgeErrorResult(1.0, 2.0, 3.0, 2, 2)

    good_result = scorer.score_document(document, edge_error=good_edge)
    bad_result = scorer.score_document(document, edge_error=bad_edge)

    assert good_result.breakdown.edge_error_score == pytest.approx(0.0)
    assert bad_result.breakdown.edge_error_score == pytest.approx(3.0)
    assert good_result.total_score < bad_result.total_score


def test_scorer_penalizes_geometry_complexity_and_extra_bezier_control_points() -> None:
    line_path = VectorPath(path_id="path_line")
    bezier_path = VectorPath(path_id="path_bezier")
    line_segment = Segment("seg_line", "path_line", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]})
    bezier_segment = Segment(
        "seg_bezier",
        "path_bezier",
        "bezier",
        params={
            "start": [0.0, 0.0],
            "control1": [2.0, 5.0],
            "control2": [8.0, 5.0],
            "end": [10.0, 0.0],
        },
    )
    line_document = _build_document(path=line_path, segments=(line_segment,))
    bezier_document = _build_document(path=bezier_path, segments=(bezier_segment,))
    scorer = Scorer()

    line_result = scorer.score_document(line_document)
    bezier_result = scorer.score_document(bezier_document)

    assert line_result.breakdown.geometry_complexity_score == pytest.approx(1.0)
    assert bezier_result.breakdown.geometry_complexity_score == pytest.approx(3.5)
    assert line_result.total_score < bezier_result.total_score


def test_scorer_penalizes_topology_errors_and_self_intersections() -> None:
    path = VectorPath(
        path_id="path_topology",
        topology_status="topology_error",
        max_gap=2.5,
        self_intersection_count=2,
    )
    segment = Segment("seg_1", "path_topology", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]})
    document = _build_document(path=path, segments=(segment,))
    scorer = Scorer()

    result = scorer.score_document(document)

    assert result.breakdown.topology_error_score == pytest.approx(27.5)
    assert result.breakdown.self_intersection_score == pytest.approx(40.0)
    assert result.total_score == pytest.approx(
        result.breakdown.edge_error_score
        + result.breakdown.geometry_complexity_score
        + result.breakdown.topology_error_score
        + result.breakdown.self_intersection_score
        + result.breakdown.coordinate_consistency_score
    )


def test_scorer_penalizes_coordinate_inconsistency_without_mutating_document() -> None:
    coordinate_system = CoordinateSystem(internal_space="pixel")
    path = VectorPath(path_id="path_coords", metadata={"coordinate_space": "pixel"})
    anchors = (
        Anchor(anchor_id="a1", path_id="path_coords", position=(0.0, 0.0), metadata={"coordinate_space": "pixel"}),
    )
    segments = (
        Segment(
            "seg_1",
            "path_coords",
            "line",
            params={"start": [0.0, 0.0], "end": [10.0, 0.0]},
            metadata={"coordinate_space": "pixel"},
        ),
    )
    document = _build_document(coordinate_system=coordinate_system, path=path, anchors=anchors, segments=segments)
    original_document = document
    scorer = Scorer()

    result = scorer.score_document(document)

    assert result.breakdown.coordinate_consistency_score == pytest.approx(16.0)
    assert document == original_document


def test_scorer_penalizes_nested_non_vector_coordinate_space_metadata() -> None:
    path = VectorPath(path_id="path_nested")
    document = _build_document(path=path)
    document = document.__class__(
        document_id=document.document_id,
        width=document.width,
        height=document.height,
        coordinate_system=document.coordinate_system,
        objects=document.objects,
        paths=document.paths,
        segments=document.segments,
        anchors=document.anchors,
        constraints=document.constraints,
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [
                        {"coordinate_space": "pixel"},
                    ]
                }
            }
        },
    )
    scorer = Scorer()

    result = scorer.score_document(document)

    assert result.breakdown.coordinate_consistency_score == pytest.approx(2.0)


def test_scorer_does_not_penalize_nested_vector_coordinate_space_metadata() -> None:
    path = VectorPath(path_id="path_nested_vector")
    document = _build_document(path=path)
    document = document.__class__(
        document_id=document.document_id,
        width=document.width,
        height=document.height,
        coordinate_system=document.coordinate_system,
        objects=document.objects,
        paths=document.paths,
        segments=document.segments,
        anchors=document.anchors,
        constraints=document.constraints,
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [
                        {"coordinate_space": "vector"},
                    ]
                }
            }
        },
    )
    scorer = Scorer()

    result = scorer.score_document(document)

    assert result.breakdown.coordinate_consistency_score == pytest.approx(0.0)


def test_scorer_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/scorer.py")
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
