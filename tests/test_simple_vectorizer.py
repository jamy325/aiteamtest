import ast
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import cv2
import numpy as np

from core.document import add_anchor, add_path, add_segment, create_document, from_json, to_json
from core.types import CoordinateSystem
from services.simple_vectorizer import SimpleVectorizer, VectorizationResult

Point = tuple[float, float]


def _visual_artifacts_enabled() -> bool:
    return os.environ.get("GENERATE_VISUAL_ARTIFACTS") == "1"


def _visual_artifact_dir() -> Path:
    root = os.environ.get("VISUAL_ARTIFACT_DIR")
    if root:
        return Path(root)
    return Path(tempfile.gettempdir()) / "aiteamtest_visual_artifacts" / "simple_vectorizer"


def _cubic_bezier_point(start: Point, control1: Point, control2: Point, end: Point, t: float) -> Point:
    one_minus_t = 1.0 - t
    x = (
        (one_minus_t ** 3) * start[0]
        + 3.0 * (one_minus_t ** 2) * t * control1[0]
        + 3.0 * one_minus_t * (t ** 2) * control2[0]
        + (t ** 3) * end[0]
    )
    y = (
        (one_minus_t ** 3) * start[1]
        + 3.0 * (one_minus_t ** 2) * t * control1[1]
        + 3.0 * one_minus_t * (t ** 2) * control2[1]
        + (t ** 3) * end[1]
    )
    return (x, y)


def _draw_wrapped_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    *,
    font_scale: float = 0.45,
    color: tuple[int, int, int] = (30, 30, 30),
    line_height: int = 18,
) -> int:
    x, y = origin
    words = text.split()
    if not words:
        return y

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        if candidate_width <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)

    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_height
    return y


def _closed_parametric_points(
    *,
    center: Point,
    radius_x: float,
    radius_y: float,
    point_count: int,
) -> list[Point]:
    points = [
        (
            center[0] + radius_x * math.cos((2.0 * math.pi * index) / point_count),
            center[1] + radius_y * math.sin((2.0 * math.pi * index) / point_count),
        )
        for index in range(point_count)
    ]
    points.append(points[0])
    return points


def _write_vectorizer_overlay(
    *,
    test_name: str,
    points: Sequence[Point],
    result: VectorizationResult,
) -> None:
    if not _visual_artifacts_enabled():
        return

    try:
        width = 1200
        drawing_height = 760
        metadata_height = 220
        height = drawing_height + metadata_height
        padding = 60
        drawable_width = width - (padding * 2)
        drawable_height = drawing_height - (padding * 2)

        bounds_points: list[Point] = [(float(x), float(y)) for x, y in points]
        bounds_points.extend(anchor.position for anchor in result.anchors)
        for anchor in result.anchors:
            if anchor.in_handle is not None:
                bounds_points.append(anchor.in_handle)
            if anchor.out_handle is not None:
                bounds_points.append(anchor.out_handle)
        for segment in result.segments:
            for key in ("start", "control1", "control2", "end"):
                value = segment.params.get(key)
                if value is not None:
                    bounds_points.append((float(value[0]), float(value[1])))

        min_x = min(point[0] for point in bounds_points)
        max_x = max(point[0] for point in bounds_points)
        min_y = min(point[1] for point in bounds_points)
        max_y = max(point[1] for point in bounds_points)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        scale = min(drawable_width / span_x, drawable_height / span_y)
        offset_x = padding + (drawable_width - (span_x * scale)) / 2.0
        offset_y = padding + (drawable_height - (span_y * scale)) / 2.0

        image = np.full((height, width, 3), 255, dtype=np.uint8)
        image[drawing_height:, :] = (246, 246, 246)

        def to_image(point: Point) -> tuple[int, int]:
            x = offset_x + (point[0] - min_x) * scale
            y = offset_y + (point[1] - min_y) * scale
            return (int(round(x)), int(round(y)))

        contour = np.array([to_image((float(x), float(y))) for x, y in points], dtype=np.int32)
        if len(contour) >= 2:
            cv2.polylines(image, [contour], isClosed=result.path.closed, color=(210, 210, 210), thickness=2)

        for index, point in enumerate(points):
            pixel = to_image((float(point[0]), float(point[1])))
            cv2.circle(image, pixel, 6, (0, 140, 255), thickness=2)
            cv2.putText(
                image,
                str(index),
                (pixel[0] + 7, pixel[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

        for segment in result.segments:
            if segment.type == "line":
                start = to_image(segment.params["start"])
                end = to_image(segment.params["end"])
                cv2.line(image, start, end, (0, 180, 0), thickness=2, lineType=cv2.LINE_AA)
                continue

            if segment.type == "bezier":
                start = segment.params["start"]
                control1 = segment.params["control1"]
                control2 = segment.params["control2"]
                end = segment.params["end"]
                sampled = np.array(
                    [
                        to_image(_cubic_bezier_point(start, control1, control2, end, step / 24.0))
                        for step in range(25)
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(image, [sampled], isClosed=False, color=(180, 0, 180), thickness=2)

        for anchor in result.anchors:
            anchor_pixel = to_image(anchor.position)
            if anchor.in_handle is not None:
                handle_pixel = to_image(anchor.in_handle)
                cv2.line(image, anchor_pixel, handle_pixel, (160, 80, 160), thickness=1, lineType=cv2.LINE_AA)
                cv2.circle(image, handle_pixel, 3, (160, 80, 160), thickness=-1)
            if anchor.out_handle is not None:
                handle_pixel = to_image(anchor.out_handle)
                cv2.line(image, anchor_pixel, handle_pixel, (160, 80, 160), thickness=1, lineType=cv2.LINE_AA)
                cv2.circle(image, handle_pixel, 3, (160, 80, 160), thickness=-1)
            cv2.circle(image, anchor_pixel, 3, (40, 40, 220), thickness=-1)

        cv2.rectangle(image, (0, drawing_height), (width - 1, height - 1), (220, 220, 220), thickness=1)
        legend_top = drawing_height + 28
        segment_type = result.segments[0].type if result.segments else "none"
        metadata_lines = [
            f"test: {test_name}",
            f"path_id: {result.path.path_id}",
            f"segments: {len(result.segments)}  anchors: {len(result.anchors)}  type: {segment_type}  closed: {result.path.closed}  topology_status: {result.path.topology_status}",
            "layers: gray=input contour, orange=input points, green=line, magenta=bezier, red=anchors, purple=bezier handles",
        ]
        current_y = legend_top
        for line in metadata_lines:
            current_y = _draw_wrapped_text(
                image,
                line,
                (padding, current_y),
                width - (padding * 2),
                font_scale=0.58,
                color=(20, 20, 20),
                line_height=24,
            )
            current_y += 4

        output_dir = _visual_artifact_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{test_name}.png"
        cv2.imwrite(str(output_path), image)
        print(f"wrote visual artifact: {output_path}")
    except Exception as exc:
        print(f"visual artifact generation skipped for {test_name}: {exc}")


def test_simple_vectorizer_creates_closed_line_path_segments_and_anchors() -> None:
    points = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0, 10.0),
        (0.0, 0.0),
    ]
    vectorizer = SimpleVectorizer(segment_type="line")

    result = vectorizer.vectorize_contour(points, path_id="path_line", closed=True, source="binary_contour")

    assert result.path.closed is True
    assert result.path.topology_status == "closed"
    assert result.path.source == "binary_contour"
    assert result.path.metadata["coordinate_space"] == "vector"
    assert len(result.anchors) == 4
    assert len(result.segments) == 4
    assert result.path.segments == tuple(segment.segment_id for segment in result.segments)
    assert all(anchor.in_handle is None and anchor.out_handle is None for anchor in result.anchors)
    assert all(anchor.continuity == "corner" for anchor in result.anchors)
    assert all(segment.type == "line" for segment in result.segments)
    assert result.segments[0].anchors == ("path_line_anchor_0", "path_line_anchor_1")
    assert result.segments[-1].anchors == ("path_line_anchor_3", "path_line_anchor_0")
    assert result.segments[0].params["start"] == [0.0, 0.0]
    assert result.segments[0].params["end"] == [10.0, 0.0]
    _write_vectorizer_overlay(
        test_name="test_simple_vectorizer_creates_closed_line_path_segments_and_anchors",
        points=points,
        result=result,
    )


def test_simple_vectorizer_creates_closed_bezier_segments_with_handles() -> None:
    points = [
        (0.0, 0.0),
        (6.0, 1.0),
        (10.0, 6.0),
        (5.0, 11.0),
        (0.0, 7.0),
        (0.0, 0.0),
    ]
    vectorizer = SimpleVectorizer(segment_type="bezier")

    result = vectorizer.vectorize_contour(points, path_id="path_bezier", closed=True, source="skeleton_contour")

    assert result.path.closed is True
    assert result.path.metadata["initial_segment_type"] == "bezier"
    assert len(result.anchors) == 5
    assert len(result.segments) == 5
    assert all(anchor.continuity == "smooth" for anchor in result.anchors)
    assert all(anchor.in_handle is not None and anchor.out_handle is not None for anchor in result.anchors)
    assert all(anchor.shared_tangent is not None for anchor in result.anchors)
    assert all(segment.type == "bezier" for segment in result.segments)
    assert all(segment.anchors[0] != segment.anchors[1] for segment in result.segments)
    assert set(result.segments[0].params) == {"start", "control1", "control2", "end"}
    assert result.segments[0].params["start"] == [0.0, 0.0]
    assert isinstance(result.segments[0].params["control1"], list)
    assert isinstance(result.segments[0].params["control2"], list)
    assert isinstance(result.segments[0].params["end"], list)
    assert result.segments[-1].anchors == ("path_bezier_anchor_4", "path_bezier_anchor_0")
    _write_vectorizer_overlay(
        test_name="test_simple_vectorizer_creates_closed_bezier_segments_with_handles",
        points=points,
        result=result,
    )


def test_simple_vectorizer_result_is_writable_to_vector_document() -> None:
    document = create_document(
        document_id="doc_vectorizer",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(),
    )
    points = [
        (0.0, 0.0),
        (8.0, 0.0),
        (8.0, 8.0),
        (0.0, 8.0),
        (0.0, 0.0),
    ]
    result = SimpleVectorizer(segment_type="line").vectorize_contour(points, path_id="path_doc", closed=True)

    document = add_path(document, result.path)
    for anchor in result.anchors:
        document = add_anchor(document, anchor)
    for segment in result.segments:
        document = add_segment(document, segment)

    assert len(document.paths) == 1
    assert len(document.anchors) == 4
    assert len(document.segments) == 4
    assert document.paths[0].segments == tuple(segment.segment_id for segment in result.segments)
    assert document.segments[0].anchors == ("path_doc_anchor_0", "path_doc_anchor_1")
    assert document.segments[0].params["start"] == [0.0, 0.0]
    assert from_json(to_json(document)) == document
    _write_vectorizer_overlay(
        test_name="test_simple_vectorizer_result_is_writable_to_vector_document",
        points=points,
        result=result,
    )


def test_simple_vectorizer_visualizes_closed_circle_like_bezier_contour() -> None:
    point_count = 16
    points = _closed_parametric_points(center=(50.0, 50.0), radius_x=30.0, radius_y=30.0, point_count=point_count)
    result = SimpleVectorizer(segment_type="bezier").vectorize_contour(
        points,
        path_id="path_circle_bezier",
        closed=True,
    )

    assert result.path.closed is True
    assert len(result.anchors) == point_count
    assert len(result.segments) == point_count
    assert all(segment.type == "bezier" for segment in result.segments)
    assert result.segments[-1].anchors == (
        f"{result.path.path_id}_anchor_{point_count - 1}",
        f"{result.path.path_id}_anchor_0",
    )
    _write_vectorizer_overlay(
        test_name="test_simple_vectorizer_visualizes_closed_circle_like_bezier_contour",
        points=points,
        result=result,
    )


def test_simple_vectorizer_visualizes_closed_ellipse_like_bezier_contour() -> None:
    point_count = 18
    points = _closed_parametric_points(center=(60.0, 40.0), radius_x=45.0, radius_y=20.0, point_count=point_count)
    result = SimpleVectorizer(segment_type="bezier").vectorize_contour(
        points,
        path_id="path_ellipse_bezier",
        closed=True,
    )

    assert result.path.closed is True
    assert len(result.anchors) == point_count
    assert len(result.segments) == point_count
    assert all(segment.type == "bezier" for segment in result.segments)
    assert result.segments[-1].anchors == (
        f"{result.path.path_id}_anchor_{point_count - 1}",
        f"{result.path.path_id}_anchor_0",
    )
    _write_vectorizer_overlay(
        test_name="test_simple_vectorizer_visualizes_closed_ellipse_like_bezier_contour",
        points=points,
        result=result,
    )


def test_simple_vectorizer_visualizes_open_wave_bezier_contour() -> None:
    points = [
        (float(x), 40.0 + 15.0 * math.sin(x / 10.0))
        for x in range(0, 121, 8)
    ]
    result = SimpleVectorizer(segment_type="bezier").vectorize_contour(
        points,
        path_id="path_wave_bezier",
        closed=False,
    )

    assert result.path.closed is False
    assert result.path.topology_status == "open"
    assert len(result.anchors) == len(points)
    assert len(result.segments) == len(points) - 1
    assert result.segments[0].anchors == ("path_wave_bezier_anchor_0", "path_wave_bezier_anchor_1")
    assert result.segments[-1].anchors == (
        f"path_wave_bezier_anchor_{len(points) - 2}",
        f"path_wave_bezier_anchor_{len(points) - 1}",
    )
    assert result.anchors[0].in_handle is None
    assert result.anchors[-1].out_handle is None
    _write_vectorizer_overlay(
        test_name="test_simple_vectorizer_visualizes_open_wave_bezier_contour",
        points=points,
        result=result,
    )


def test_simple_vectorizer_visualizes_closed_circle_like_line_contour() -> None:
    point_count = 16
    points = _closed_parametric_points(center=(50.0, 50.0), radius_x=30.0, radius_y=30.0, point_count=point_count)
    result = SimpleVectorizer(segment_type="line").vectorize_contour(
        points,
        path_id="path_circle_line",
        closed=True,
    )

    assert result.path.closed is True
    assert len(result.anchors) == point_count
    assert len(result.segments) == point_count
    assert all(segment.type == "line" for segment in result.segments)
    assert result.segments[-1].anchors == (
        f"{result.path.path_id}_anchor_{point_count - 1}",
        f"{result.path.path_id}_anchor_0",
    )
    _write_vectorizer_overlay(
        test_name="test_simple_vectorizer_visualizes_closed_circle_like_line_contour",
        points=points,
        result=result,
    )


def test_simple_vectorizer_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/simple_vectorizer.py")
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
