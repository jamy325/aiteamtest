from __future__ import annotations

import math

import cv2
import numpy as np

from core.types import CoordinateSystem
from services.document_integrity import DocumentIntegrityValidator
from services.minimal_pipeline import MinimalPipeline
from services.resampler import Resampler, ResamplerConfig


def _wireframe_square_image(size: int = 120, *, thickness: int = 3) -> np.ndarray:
    image = 255 * np.ones((size, size, 3), dtype=np.uint8)
    margin = 20
    cv2.rectangle(image, (margin, margin), (size - margin, size - margin), (0, 0, 0), thickness=thickness)
    return image


def _distance_to_chord(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    segment_length_sq = (segment_x * segment_x) + (segment_y * segment_y)
    if segment_length_sq <= 1e-9:
        return math.dist(point, start)
    projection = ((point[0] - start[0]) * segment_x + (point[1] - start[1]) * segment_y) / segment_length_sq
    projection = max(0.0, min(1.0, projection))
    closest = (start[0] + projection * segment_x, start[1] + projection * segment_y)
    return math.dist(point, closest)


def test_black_square_skeleton_centerline_is_simplified_to_reasonable_segment_count() -> None:
    pipeline = MinimalPipeline(
        coordinate_system=CoordinateSystem(view_box=(0.0, 0.0, 120.0, 120.0), precision=4),
        segment_type="line",
    )

    result = pipeline.run(_wireframe_square_image(), document_id="square_wireframe", debug=True)
    report = DocumentIntegrityValidator().validate(result.document)
    skeleton_paths = [path for path in result.document.paths if path.source == "skeleton_contour"]

    assert skeleton_paths
    assert max(len(path.segments) for path in skeleton_paths) <= 12
    assert len(result.document.segments) <= 32
    assert report.success is True
    assert result.debug_artifacts is not None
    summary = result.debug_artifacts.summary["skeleton_simplification"]
    assert summary["original_point_count"] > summary["simplified_point_count"]
    assert summary["original_segment_count"] > summary["simplified_segment_count"]


def test_skeleton_simplification_preserves_polyline_corners() -> None:
    dense_l_shape = tuple((float(x), 0.0) for x in range(0, 31)) + tuple((30.0, float(y)) for y in range(1, 31))
    resampler = Resampler(
        ResamplerConfig(
            skeleton_rdp_epsilon_ratio=0.03,
            duplicate_epsilon=1e-6,
        )
    )

    simplified = resampler.simplify_linear_contour(dense_l_shape, closed=False)

    assert simplified[0] == (0.0, 0.0)
    assert simplified[-1] == (30.0, 30.0)
    assert (30.0, 0.0) in simplified
    assert len(simplified) <= 4


def test_skeleton_simplification_does_not_collapse_arc_into_single_line() -> None:
    arc_points = tuple(
        (
            30.0 + 20.0 * math.cos(math.pi * step / 39.0),
            30.0 + 20.0 * math.sin(math.pi * step / 39.0),
        )
        for step in range(40)
    )
    resampler = Resampler(
        ResamplerConfig(
            skeleton_rdp_epsilon_ratio=0.02,
            duplicate_epsilon=1e-6,
        )
    )

    simplified = resampler.simplify_linear_contour(arc_points, closed=False)

    assert len(simplified) > 2
    assert any(
        _distance_to_chord(point, simplified[0], simplified[-1]) > 1.0
        for point in simplified[1:-1]
    )
