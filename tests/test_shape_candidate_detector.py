from __future__ import annotations

import math
from pathlib import Path

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.minimal_pipeline import MinimalPipeline
from services.shape_candidate_detector import ShapeCandidateDetector


def _rectangle_document() -> object:
    document = create_document(
        document_id="doc_rectangle",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="rect_path",
            closed=True,
            segments=("rect_0", "rect_1", "rect_2", "rect_3"),
        ),
    )
    corners = ((20.0, 20.0), (120.0, 20.0), (120.0, 80.0), (20.0, 80.0))
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        document = add_segment(
            document,
            Segment(
                segment_id=f"rect_{index}",
                path_id="rect_path",
                type="line",
                params={"start": [start[0], start[1]], "end": [end[0], end[1]]},
            ),
        )
    return document


def _line_document() -> object:
    document = create_document(
        document_id="doc_line",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(
        document,
        VectorPath(path_id="line_path", closed=False, segments=("line_0", "line_1")),
    )
    document = add_segment(
        document,
        Segment(
            segment_id="line_0",
            path_id="line_path",
            type="polyline",
            params={"points": [[10.0, 10.0], [35.0, 10.4], [60.0, 10.1], [85.0, 10.3]]},
        ),
    )
    document = add_segment(
        document,
        Segment(
            segment_id="line_1",
            path_id="line_path",
            type="polyline",
            params={"points": [[85.0, 10.3], [110.0, 10.2], [135.0, 10.1], [160.0, 10.0]]},
        ),
    )
    return document


def _arc_document() -> object:
    document = create_document(
        document_id="doc_arc",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [],
                    "skeleton_contours": [
                        {
                            "contour_id": "arc_contour_0",
                            "source": "skeleton_contour",
                            "points": [
                                [100.0 + 40.0 * math.cos(angle), 100.0 + 40.0 * math.sin(angle)]
                                for angle in [i * (math.pi / 18.0) for i in range(10)]
                            ],
                            "coordinate_space": "vector",
                            "closed": False,
                            "area": 0.0,
                            "depth": 0,
                            "parent_contour": None,
                            "children": [],
                        }
                    ],
                }
            }
        },
    )
    points = tuple(
        (100.0 + 40.0 * math.cos(angle), 100.0 + 40.0 * math.sin(angle))
        for angle in [i * (math.pi / 18.0) for i in range(10)]
    )
    document = add_path(
        document,
        VectorPath(
            path_id="arc_path",
            closed=False,
            segments=("arc_0",),
            source="skeleton_contour",
            metadata={"source_contour_id": "arc_contour_0"},
        ),
    )
    document = add_segment(
        document,
        Segment(
            segment_id="arc_0",
            path_id="arc_path",
            type="polyline",
            params={"points": [[x, y] for x, y in points]},
        ),
    )
    return document


def _noise_document() -> object:
    document = create_document(
        document_id="doc_noise",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    points = (
        (10.0, 10.0),
        (25.0, 40.0),
        (15.0, 70.0),
        (45.0, 20.0),
        (55.0, 65.0),
        (70.0, 15.0),
        (80.0, 75.0),
        (20.0, 85.0),
        (10.0, 10.0),
    )
    document = add_path(
        document,
        VectorPath(path_id="noise_path", closed=True, segments=("noise_0",)),
    )
    document = add_segment(
        document,
        Segment(
            segment_id="noise_0",
            path_id="noise_path",
            type="polyline",
            params={"points": [[x, y] for x, y in points]},
        ),
    )
    return document


def test_shape_candidate_detector_detects_circle_candidate_from_circle_fixture() -> None:
    fixture_path = Path("test_images/circle/test_input_circle.png")
    document = MinimalPipeline(segment_type="line").run_from_file(fixture_path, document_id="circle_fixture").document
    original_document = document

    candidates = ShapeCandidateDetector().detect_candidates(document)

    circle_candidates = [candidate for candidate in candidates if candidate.target_type == "circle"]
    assert circle_candidates
    best = max(circle_candidates, key=lambda item: item.confidence)
    assert best.confidence >= 0.68
    assert best.source == "raw_contour_points"
    assert best.evidence["raw_point_count"] > 0
    assert best.evidence["fit_point_count"] == best.evidence["raw_point_count"]
    assert "bbox" in best.evidence
    assert best.evidence["fit_error"] >= 0.0
    assert document == original_document


def test_shape_candidate_detector_detects_ellipse_candidate_from_ellipse_fixture() -> None:
    fixture_path = Path("test_images/ellipse/test_input_ellipse.png")
    document = MinimalPipeline(segment_type="line").run_from_file(fixture_path, document_id="ellipse_fixture").document

    candidates = ShapeCandidateDetector().detect_candidates(document)

    ellipse_candidates = [candidate for candidate in candidates if candidate.target_type == "ellipse"]
    assert ellipse_candidates
    best = max(ellipse_candidates, key=lambda item: item.confidence)
    assert best.confidence >= 0.7
    assert best.source == "raw_contour_points"
    assert best.evidence["raw_point_count"] > best.evidence["segment_count"]


def test_shape_candidate_detector_detects_rectangle_candidate_from_line_frame_document() -> None:
    candidates = ShapeCandidateDetector().detect_candidates(_rectangle_document())

    rectangle_candidates = [candidate for candidate in candidates if candidate.target_type == "rectangle"]
    assert rectangle_candidates
    best = max(rectangle_candidates, key=lambda item: item.confidence)
    assert best.segment_range == (0, 3)
    assert best.evidence["model_complexity_delta"] == 0


def test_shape_candidate_detector_detects_line_candidate_for_long_straight_range() -> None:
    candidates = ShapeCandidateDetector().detect_candidates(_line_document())

    line_candidates = [candidate for candidate in candidates if candidate.target_type == "line"]
    assert line_candidates
    best = max(line_candidates, key=lambda item: item.confidence)
    assert best.confidence >= 0.72
    assert best.path_id == "line_path"


def test_shape_candidate_detector_detects_arc_candidate_for_curved_range() -> None:
    candidates = ShapeCandidateDetector().detect_candidates(_arc_document())

    arc_candidates = [candidate for candidate in candidates if candidate.target_type == "arc"]
    assert arc_candidates
    best = max(arc_candidates, key=lambda item: item.confidence)
    assert best.confidence >= 0.68
    assert best.evidence["arc_angle_coverage"] < math.tau


def test_shape_candidate_detector_does_not_emit_high_confidence_candidates_for_noise() -> None:
    candidates = ShapeCandidateDetector().detect_candidates(_noise_document())

    assert not candidates or max(candidate.confidence for candidate in candidates) < 0.68
