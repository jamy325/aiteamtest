import ast
import os
from pathlib import Path
import tempfile
from typing import Sequence

import cv2
import numpy as np

from core.document import add_anchor, add_path, add_segment, create_document
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


def _write_vectorizer_overlay(
    *,
    test_name: str,
    points: Sequence[Point],
    result: VectorizationResult,
) -> None:
    if not _visual_artifacts_enabled():
        return

    try:
        width = 800
        height = 600
        padding = 40
        legend_height = 110
        drawable_width = width - (padding * 2)
        drawable_height = height - (padding * 2) - legend_height

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

        image = np.full((height, width, 3), 255, dtype=np.uint8)

        def to_image(point: Point) -> tuple[int, int]:
            x = padding + (point[0] - min_x) * scale
            y = padding + (point[1] - min_y) * scale
            return (int(round(x)), int(round(y)))

        contour = np.array([to_image((float(x), float(y))) for x, y in points], dtype=np.int32)
        if len(contour) >= 2:
            cv2.polylines(image, [contour], isClosed=result.path.closed, color=(200, 200, 200), thickness=2)

        for index, point in enumerate(points):
            pixel = to_image((float(point[0]), float(point[1])))
            cv2.circle(image, pixel, 4, (220, 120, 20), thickness=-1)
            cv2.putText(
                image,
                str(index),
                (pixel[0] + 5, pixel[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
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
            cv2.circle(image, anchor_pixel, 5, (30, 30, 220), thickness=-1)

        legend_top = height - legend_height + 20
        segment_type = result.segments[0].type if result.segments else "none"
        legend_lines = [
            f"test: {test_name}",
            f"path_id: {result.path.path_id}",
            f"segments: {len(result.segments)}  anchors: {len(result.anchors)}  type: {segment_type}",
            "layers: gray=input contour, orange=input points, green=line, magenta=bezier, red=anchor, purple=handles",
        ]
        for index, line in enumerate(legend_lines):
            cv2.putText(
                image,
                line,
                (padding, legend_top + (index * 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

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
    assert result.segments[0].params["start"] == (0.0, 0.0)
    assert result.segments[0].params["end"] == (10.0, 0.0)
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
    _write_vectorizer_overlay(
        test_name="test_simple_vectorizer_result_is_writable_to_vector_document",
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
