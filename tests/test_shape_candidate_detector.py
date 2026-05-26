from __future__ import annotations

import math
from pathlib import Path

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.minimal_pipeline import MinimalPipeline
from services.shape_candidate_detector import ShapeCandidateDetector, ShapeCandidateDetectorConfig


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


def _generic_circle_document(*, radius: float = 28.0) -> object:
    document = create_document(
        document_id="generic_circle_doc",
        width=160.0,
        height=160.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [],
                    "skeleton_contours": [
                        {
                            "contour_id": "generic_circle_contour_0",
                            "source": "skeleton_contour",
                            "points": [
                                [80.0 + radius * math.cos(math.tau * index / 96), 80.0 + radius * math.sin(math.tau * index / 96)]
                                for index in range(96)
                            ],
                            "coordinate_space": "vector",
                            "closed": True,
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
    polygon_points = tuple(
        (80.0 + radius * math.cos(math.tau * index / 16), 80.0 + radius * math.sin(math.tau * index / 16))
        for index in range(16)
    )
    segment_ids = tuple(f"generic_circle_seg_{index}" for index in range(len(polygon_points)))
    document = add_path(
        document,
        VectorPath(
            path_id="generic_circle_path",
            closed=True,
            source="skeleton_contour",
            segments=segment_ids,
            metadata={"source_contour_id": "generic_circle_contour_0"},
        ),
    )
    for index in range(len(polygon_points)):
        start = polygon_points[index]
        end = polygon_points[(index + 1) % len(polygon_points)]
        document = add_segment(
            document,
            Segment(
                segment_id=segment_ids[index],
                path_id="generic_circle_path",
                type="line",
                params={"start": [start[0], start[1]], "end": [end[0], end[1]]},
            ),
        )
    return document


def _tiny_line_document() -> object:
    document = create_document(
        document_id="doc_tiny_line",
        width=20.0,
        height=20.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="tiny_line_path", closed=False, segments=("tiny_line_0",)))
    document = add_segment(
        document,
        Segment(
            segment_id="tiny_line_0",
            path_id="tiny_line_path",
            type="polyline",
            params={"points": [[2.0, 2.0], [4.0, 2.05], [6.0, 2.0]]},
        ),
    )
    return document


def _raw_contour_document(
    points: tuple[tuple[float, float], ...],
    *,
    path_id: str,
    contour_id: str,
    closed: bool,
) -> object:
    document = create_document(
        document_id=f"{path_id}_doc",
        width=240.0,
        height=240.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [
                        {
                            "contour_id": contour_id,
                            "source": "binary_contour",
                            "points": [[point[0], point[1]] for point in points],
                            "coordinate_space": "vector",
                            "closed": closed,
                            "area": 0.0,
                            "depth": 0,
                            "parent_contour": None,
                            "children": [],
                        }
                    ],
                    "skeleton_contours": [],
                }
            }
        },
    )
    document = add_path(
        document,
        VectorPath(
            path_id=path_id,
            closed=closed,
            source="binary_contour",
            segments=(f"{path_id}_seg_0",),
            metadata={"source_contour_id": contour_id},
        ),
    )
    document = add_segment(
        document,
        Segment(
            segment_id=f"{path_id}_seg_0",
            path_id=path_id,
            type="polyline",
            params={"points": [[point[0], point[1]] for point in points]},
        ),
    )
    return document


def _open_wave_document() -> object:
    points = tuple(
        (
            12.0 + (index * 2.5),
            60.0 + (14.0 * math.sin((index / 47.0) * math.tau * 1.5)),
        )
        for index in range(48)
    )
    return _raw_contour_document(points, path_id="wave_path", contour_id="wave_contour_0", closed=False)


def _mixed_source_rectangle_document() -> object:
    document = create_document(
        document_id="doc_mixed_sources",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="binary_rect_path",
            closed=True,
            source="binary_contour",
            segments=("binary_0", "binary_1", "binary_2", "binary_3"),
        ),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="skeleton_rect_path",
            closed=True,
            source="skeleton_contour",
            segments=("skeleton_0", "skeleton_1", "skeleton_2", "skeleton_3"),
        ),
    )
    binary_corners = ((20.0, 20.0), (100.0, 20.0), (100.0, 80.0), (20.0, 80.0))
    skeleton_corners = ((120.0, 20.0), (180.0, 20.0), (180.0, 80.0), (120.0, 80.0))
    for index in range(4):
        binary_start = binary_corners[index]
        binary_end = binary_corners[(index + 1) % 4]
        skeleton_start = skeleton_corners[index]
        skeleton_end = skeleton_corners[(index + 1) % 4]
        document = add_segment(
            document,
            Segment(
                segment_id=f"binary_{index}",
                path_id="binary_rect_path",
                type="line",
                params={"start": [binary_start[0], binary_start[1]], "end": [binary_end[0], binary_end[1]]},
            ),
        )
        document = add_segment(
            document,
            Segment(
                segment_id=f"skeleton_{index}",
                path_id="skeleton_rect_path",
                type="line",
                params={"start": [skeleton_start[0], skeleton_start[1]], "end": [skeleton_end[0], skeleton_end[1]]},
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


def test_shape_candidate_detector_filters_paths_by_processing_contour_source() -> None:
    document = _mixed_source_rectangle_document()
    detector = ShapeCandidateDetector()

    all_candidates = detector.detect_candidates(document, contour_source="all")
    skeleton_candidates = detector.detect_candidates(document, contour_source="skeleton")
    binary_candidates = detector.detect_candidates(document, contour_source="binary")

    assert all_candidates
    assert any(candidate.path_id == "binary_rect_path" for candidate in all_candidates)
    assert any(candidate.path_id == "skeleton_rect_path" for candidate in all_candidates)
    assert skeleton_candidates
    assert {candidate.path_id for candidate in skeleton_candidates} == {"skeleton_rect_path"}
    assert binary_candidates
    assert {candidate.path_id for candidate in binary_candidates} == {"binary_rect_path"}


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


def test_shape_candidate_detector_config_can_filter_circle_candidates_more_strictly() -> None:
    fixture_path = Path("test_images/circle/test_input_circle.png")
    document = MinimalPipeline(segment_type="line").run_from_file(fixture_path, document_id="circle_fixture_strict").document

    default_candidates = [
        candidate for candidate in ShapeCandidateDetector().detect_candidates(document) if candidate.target_type == "circle"
    ]
    strict_candidates = [
        candidate
        for candidate in ShapeCandidateDetector(
            ShapeCandidateDetectorConfig(min_circle_confidence=0.95)
        ).detect_candidates(document)
        if candidate.target_type == "circle"
    ]

    assert default_candidates
    assert strict_candidates == []


def test_shape_candidate_detector_config_can_disable_raw_source_preference() -> None:
    document = _generic_circle_document()

    candidates = ShapeCandidateDetector(
        ShapeCandidateDetectorConfig(prefer_raw_source_points=False, min_circle_points=12)
    ).detect_candidates(document)

    circle_candidates = [candidate for candidate in candidates if candidate.target_type == "circle"]
    assert circle_candidates
    best = max(circle_candidates, key=lambda item: item.confidence)
    assert best.source == "segment_samples_fallback"


def test_shape_candidate_detector_config_can_keep_tiny_paths() -> None:
    document = _tiny_line_document()

    default_candidates = ShapeCandidateDetector().detect_candidates(document)
    configured_candidates = ShapeCandidateDetector(
        ShapeCandidateDetectorConfig(
            filter_tiny_paths=False,
            min_path_diagonal=1.0,
            min_line_length=3.0,
            min_line_confidence=0.2,
        )
    ).detect_candidates(document)

    assert default_candidates == ()
    assert any(candidate.target_type == "line" for candidate in configured_candidates)


def test_shape_candidate_detector_can_emit_bezier_fallback_for_open_wave_when_enabled() -> None:
    candidates = ShapeCandidateDetector(
        ShapeCandidateDetectorConfig(
            enable_bezier_fallback=True,
            min_bezier_confidence=0.35,
            bezier_max_error=3.0,
            bezier_max_segments=5,
        )
    ).detect_candidates(_open_wave_document())

    bezier_candidates = [candidate for candidate in candidates if candidate.target_type == "bezier"]
    assert bezier_candidates
    best = max(bezier_candidates, key=lambda item: item.confidence)
    assert best.evidence["fitted_segment_count"] <= 5
    assert best.confidence >= 0.35


def test_shape_candidate_detector_can_emit_bezier_fallback_for_blob_fixture_when_enabled() -> None:
    fixture_path = Path("test_images/bezier/test_input_bezier_fallback.png")
    document = MinimalPipeline(segment_type="line").run_from_file(fixture_path, document_id="blob_fixture").document

    candidates = ShapeCandidateDetector(
        ShapeCandidateDetectorConfig(
            enable_bezier_fallback=True,
            min_bezier_confidence=0.35,
            bezier_max_error=3.5,
            bezier_max_segments=8,
        )
    ).detect_candidates(document)

    bezier_candidates = [candidate for candidate in candidates if candidate.target_type == "bezier"]
    assert bezier_candidates
    best = max(bezier_candidates, key=lambda item: item.confidence)
    assert best.source == "raw_contour_points"
    assert best.evidence["fitted_segment_count"] <= 8
    assert best.evidence["fitted_segment_count"] < (best.evidence["raw_point_count"] // 4)


def test_shape_candidate_detector_does_not_prefer_bezier_fallback_for_circle_or_ellipse() -> None:
    detector = ShapeCandidateDetector(
        ShapeCandidateDetectorConfig(
            enable_bezier_fallback=True,
            min_bezier_confidence=0.2,
            bezier_standard_confidence_threshold=0.7,
        )
    )

    circle_document = MinimalPipeline(segment_type="line").run_from_file(
        Path("test_images/circle/test_input_circle.png"),
        document_id="circle_fixture_bezier_guard",
    ).document
    ellipse_document = MinimalPipeline(segment_type="line").run_from_file(
        Path("test_images/ellipse/test_input_ellipse.png"),
        document_id="ellipse_fixture_bezier_guard",
    ).document

    circle_candidates = detector.detect_candidates(circle_document)
    ellipse_candidates = detector.detect_candidates(ellipse_document)

    circle_shape_candidates = [
        candidate for candidate in circle_candidates if candidate.target_type in {"circle", "ellipse", "bezier"}
    ]
    ellipse_shape_candidates = [
        candidate for candidate in ellipse_candidates if candidate.target_type in {"circle", "ellipse", "bezier"}
    ]

    assert any(candidate.target_type == "circle" for candidate in circle_shape_candidates)
    assert any(candidate.target_type == "ellipse" for candidate in ellipse_shape_candidates)
    assert max(circle_shape_candidates, key=lambda item: item.confidence).target_type == "circle"
    assert max(ellipse_shape_candidates, key=lambda item: item.confidence).target_type == "ellipse"
