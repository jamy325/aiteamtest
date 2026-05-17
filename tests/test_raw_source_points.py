from __future__ import annotations

import math
from pathlib import Path

import pytest
import cv2
import numpy as np

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.command_executor import CommandExecutor
from services.minimal_pipeline import MinimalPipeline


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
                },
                "resampled_contours": {
                    "binary_contours": [],
                    "skeleton_contours": [
                        {
                            "contour_id": "skeleton_contour_0",
                            "source": source,
                            "points": [[x, y] for x, y in raw_points[::6]],
                            "coordinate_space": "vector",
                            "closed": True,
                            "area": 0.0,
                            "depth": 0,
                            "parent_contour": None,
                            "children": [],
                        }
                    ],
                },
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
    replacement = next(segment for segment in result.document.segments if segment.type == "circle")
    assert replacement.metadata["executor"]["raw_source_point_count"] == 96
    assert replacement.metadata["executor"]["fitting_point_count"] == 96
    assert replacement.metadata["executor"]["support_point_count"] < 96


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


def test_circle_replacement_prefers_pipeline_source_contours_over_resampled_contours(tmp_path: Path) -> None:
    image_path = tmp_path / "circle.png"
    image = np.full((160, 160, 3), 255, dtype=np.uint8)
    cv2.circle(image, (80, 80), 42, (0, 0, 0), thickness=3)
    cv2.imwrite(str(image_path), image)

    document = MinimalPipeline(segment_type="line").run_from_file(image_path, document_id="doc_pipeline_circle").document
    candidate_path = max(
        (
            path
            for path in document.paths
            if path.closed and path.source == "skeleton_contour" and path.metadata.get("source_contour_id")
        ),
        key=lambda item: len(item.segments),
    )
    contour_id = str(candidate_path.metadata["source_contour_id"])
    source_entry = next(
        contour
        for contour in document.metadata["pipeline"]["source_contours"]["skeleton_contours"]
        if contour["contour_id"] == contour_id
    )
    resampled_entry = next(
        contour
        for contour in document.metadata["pipeline"]["resampled_contours"]["skeleton_contours"]
        if contour["contour_id"] == contour_id
    )

    result = CommandExecutor().execute(
        {
            "command_id": "replace_circle_pipeline_raw",
            "tool": "propose_replace_path_with_circle",
            "path_id": candidate_path.path_id,
            "reason": "manual circle test",
            "confidence": 0.95,
            "requires_user_confirmation": True,
        },
        document,
    )

    assert result.success is True
    assert result.fitting_source == "raw_contour_points"
    replacement = next(segment for segment in result.document.segments if segment.type == "circle")
    executor_metadata = replacement.metadata["executor"]
    source_count = len(source_entry["points"])
    if source_entry["points"] and source_entry["points"][0] == source_entry["points"][-1]:
        source_count -= 1
    assert executor_metadata["raw_source_point_count"] == source_count
    assert executor_metadata["fitting_point_count"] == executor_metadata["raw_source_point_count"]
    assert executor_metadata["fitting_point_count"] > len(resampled_entry["points"])
    assert executor_metadata["support_point_count"] <= len(resampled_entry["points"])
