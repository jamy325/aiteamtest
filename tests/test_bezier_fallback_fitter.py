from __future__ import annotations

import math
from pathlib import Path

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Point, Segment
from services.bezier_fallback_fitter import BezierFallbackFitter
from services.minimal_pipeline import MinimalPipeline
from services.segment_sampler import SegmentSampler


def _wave_points(count: int = 48) -> tuple[Point, ...]:
    return tuple(
        (
            10.0 + (index * 2.4),
            40.0 + (12.0 * math.sin((index / max(count - 1, 1)) * math.tau * 1.5)),
        )
        for index in range(count)
    )


def _heart_points(count: int = 96) -> tuple[Point, ...]:
    points: list[Point] = []
    for index in range(count):
        angle = (math.tau * index) / count
        x = 16.0 * (math.sin(angle) ** 3)
        y = (
            (13.0 * math.cos(angle))
            - (5.0 * math.cos(2.0 * angle))
            - (2.0 * math.cos(3.0 * angle))
            - math.cos(4.0 * angle)
        )
        points.append((80.0 + (x * 3.0), 90.0 - (y * 3.0)))
    return tuple(points)


def _approximate_max_distance(points: tuple[Point, ...], sampled_points: tuple[Point, ...]) -> float:
    return max(
        min(math.dist(point, sample) for sample in sampled_points)
        for point in points
    )


def _sampled_points(segments: tuple[Segment, ...]) -> tuple[Point, ...]:
    sampler = SegmentSampler()
    sampled: list[Point] = []
    for segment in segments:
        current = tuple(sampler.sample_segment(segment))
        if sampled and current:
            if math.dist(sampled[-1], current[0]) <= 1e-9:
                sampled.extend(current[1:])
            else:
                sampled.extend(current)
        else:
            sampled.extend(current)
    return tuple(sampled)


def _blob_fixture_points() -> tuple[Point, ...]:
    document = MinimalPipeline(segment_type="line").run_from_file(
        Path("test_images/bezier/test_input_bezier_fallback.png"),
        document_id="bezier_fallback_fixture",
    ).document
    pipeline = document.metadata["pipeline"]
    binary_contours = pipeline["source_contours"]["binary_contours"]
    contour = max(binary_contours, key=lambda item: len(item["points"]))
    return tuple((float(point[0]), float(point[1])) for point in contour["points"])


def _synthetic_contour_document(
    points: tuple[Point, ...],
    *,
    path_id: str,
    closed: bool,
    contour_id: str,
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
            segments=(f"{path_id}_segment_0",),
            metadata={"source_contour_id": contour_id},
        ),
    )
    document = add_segment(
        document,
        Segment(
            segment_id=f"{path_id}_segment_0",
            path_id=path_id,
            type="polyline",
            params={"points": [[point[0], point[1]] for point in points]},
        ),
    )
    return document


def test_bezier_fallback_fitter_fits_open_wave_with_few_segments() -> None:
    points = _wave_points()
    fitter = BezierFallbackFitter()

    result = fitter.fit_contour(points, path_id="wave_path", closed=False, max_error=2.5, max_segments=5)

    assert result.closed is False
    assert 2 <= len(result.segments) <= 5
    assert len(result.anchors) == len(result.segments) + 1
    assert all(segment.type == "bezier" for segment in result.segments)
    assert result.confidence >= 0.45
    assert _approximate_max_distance(points, _sampled_points(result.segments)) <= 4.0


def test_bezier_fallback_fitter_fits_closed_heart_curve() -> None:
    points = _heart_points()
    fitter = BezierFallbackFitter()

    result = fitter.fit_contour(points, path_id="heart_path", closed=True, max_error=4.0, max_segments=6)

    assert result.closed is True
    assert 3 <= len(result.segments) <= 6
    assert len(result.anchors) == len(result.segments)
    assert result.confidence >= 0.4
    assert _approximate_max_distance(points, _sampled_points(result.segments)) <= 6.0


def test_bezier_fallback_fitter_reduces_blob_fixture_to_compact_bezier_segments() -> None:
    points = _blob_fixture_points()
    fitter = BezierFallbackFitter()

    result = fitter.fit_contour(points, path_id="blob_path", closed=True, max_error=3.5, max_segments=8)

    assert len(result.segments) <= 8
    assert len(result.segments) < (len(points) // 4)
    assert result.complexity_score > 0.0
    assert _approximate_max_distance(points, _sampled_points(result.segments)) <= 6.0
