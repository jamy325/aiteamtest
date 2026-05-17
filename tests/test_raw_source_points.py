from __future__ import annotations

import math

import pytest

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.command_executor import CommandExecutor


def _circle_points(
    *,
    cx: float,
    cy: float,
    radius: float,
    count: int,
) -> tuple[tuple[float, float], ...]:
    points = []
    for index in range(count):
        angle = math.tau * index / count
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    points.append(points[0])
    return tuple(points)


def _polygon_from_circle(
    *,
    cx: float,
    cy: float,
    radius: float,
    count: int,
) -> tuple[tuple[float, float], ...]:
    points = []
    for index in range(count):
        angle = math.tau * index / count
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return tuple(points)


def _document_with_polygon_circle(
    *,
    with_raw_points: bool,
    source: str = "skeleton_contour",
) -> object:
    raw_points = _circle_points(cx=64.0, cy=64.0, radius=28.0, count=96)
    polygon_points = _polygon_from_circle(cx=64.0, cy=64.0, radius=28.0, count=16)
    document = create_document(
        document_id="doc_raw_circle",
        width=160.0,
        height=160.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [],
                    "skeleton_contours": [
                        {
                            "contour_id": "skeleton_contour_0",
                            "source": source,
                            "points": [[x, y] for x, y in raw_points],
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

    segment_ids = tuple(f"path_circle_seg_{index}" for index in range(len(polygon_points)))
    path_metadata = {"source_contour_id": "skeleton_contour_0"} if with_raw_points else {}
    document = add_path(
        document,
        VectorPath(
            path_id="path_circle",
            closed=True,
            source=source,
            segments=segment_ids,
            metadata=path_metadata,
        ),
    )
    for index in range(len(polygon_points)):
        start = polygon_points[index]
        end = polygon_points[(index + 1) % len(polygon_points)]
        document = add_segment(
            document,
            Segment(
                segment_id=segment_ids[index],
                path_id="path_circle",
                type="line",
                params={"start": [start[0], start[1]], "end": [end[0], end[1]]},
            ),
        )
    return document


def test_circle_replacement_uses_raw_source_points_for_simplified_polygon() -> None:
    document = _document_with_polygon_circle(with_raw_points=True)
    executor = CommandExecutor()

    result = executor.execute(
        {
            "command_id": "replace_circle_raw",
            "tool": "propose_replace_path_with_circle",
            "path_id": "path_circle",
            "reason": "manual circle test",
            "confidence": 0.95,
            "requires_user_confirmation": True,
        },
        document,
    )

    assert result.success is True
    assert result.fitting_source == "raw_contour_points"
    assert any(segment.type == "circle" for segment in result.document.segments)


def test_circle_replacement_fallback_records_segment_samples_source_when_raw_missing() -> None:
    document = _document_with_polygon_circle(with_raw_points=False)
    executor = CommandExecutor()

    result = executor.execute(
        {
            "command_id": "replace_circle_fallback",
            "tool": "propose_replace_path_with_circle",
            "path_id": "path_circle",
            "reason": "manual circle test",
            "confidence": 0.95,
            "requires_user_confirmation": True,
        },
        document,
    )

    assert result.fitting_source == "segment_samples_fallback"
    if result.success:
        assert any(segment.type == "circle" for segment in result.document.segments)
    else:
        assert result.reason
