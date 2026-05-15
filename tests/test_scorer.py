import ast
import math
from pathlib import Path

import pytest

from core.document import add_anchor, add_constraint, add_path, add_segment, create_document
from core.types import Anchor, Constraint, CoordinateSystem, Path as VectorPath, Segment
from services.distance_field_diff import DistanceFieldDiffRenderer
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
        + result.breakdown.shared_tangent_violation_score
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


def test_scorer_accumulates_shared_tangent_violation_score() -> None:
    path = VectorPath(path_id="path_shared", segments=("seg_1", "seg_2"))
    anchors = (
        Anchor(anchor_id="a1", path_id="path_shared", position=(10.0, 0.0), shared_tangent=(1.0, 0.0)),
    )
    segments = (
        Segment("seg_1", "path_shared", "line", params={"start": [0.0, 0.0], "end": [10.0, 0.0]}, anchors=("a0", "a1")),
        Segment("seg_2", "path_shared", "line", params={"start": [10.0, 0.0], "end": [20.0, 5.0]}, anchors=("a1", "a2")),
    )
    document = _build_document(path=path, anchors=anchors, segments=segments)
    document = add_constraint(
        document,
        Constraint(
            constraint_id="g1_1",
            type="shared_tangent",
            targets=("seg_1", "seg_2", "a1"),
            locked=False,
        ),
    )
    scorer = Scorer()

    result = scorer.score_document(document)

    assert result.breakdown.shared_tangent_violation_score > 0.0
    assert result.total_score == pytest.approx(
        result.breakdown.edge_error_score
        + result.breakdown.geometry_complexity_score
        + result.breakdown.topology_error_score
        + result.breakdown.self_intersection_score
        + result.breakdown.shared_tangent_violation_score
        + result.breakdown.coordinate_consistency_score
    )


def test_scorer_accepts_edge_error_from_arc_sampling_pipeline() -> None:
    path = VectorPath(path_id="path_arc_score")
    document = create_document(
        document_id="doc_arc_score",
        width=64.0,
        height=64.0,
        coordinate_system=CoordinateSystem(unit="px", precision=4, view_box=(0.0, 0.0, 64.0, 64.0)),
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [
                        {
                            "contour_id": "arc_source",
                            "points": [
                                [32.0 + 12.0 * math.cos(step * (math.pi / 10.0)), 24.0 + 12.0 * math.sin(step * (math.pi / 10.0))]
                                for step in range(6)
                            ],
                            "coordinate_space": "vector",
                            "closed": False,
                        }
                    ],
                    "skeleton_contours": [],
                }
            }
        },
    )
    document = add_path(document, path)
    document = add_segment(
        document,
        Segment(
            "seg_arc_score",
            "path_arc_score",
            "arc",
            params={"cx": 32.0, "cy": 24.0, "r": 12.0, "start_angle": 0.0, "end_angle": math.pi / 2.0, "direction": "ccw"},
        ),
    )
    diff = DistanceFieldDiffRenderer().render_diff(document)
    edge_error = EdgeErrorResult(
        diff.missing_edge_error,
        diff.overdraw_error,
        diff.chamfer_error,
        diff.source_point_count,
        diff.vector_point_count,
    )
    scorer = Scorer()

    result = scorer.score_document(document, edge_error=edge_error)

    assert diff.vector_point_count > 0
    assert result.breakdown.edge_error_score == pytest.approx(diff.chamfer_error)


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
