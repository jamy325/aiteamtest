import ast
import math
from pathlib import Path

import cv2
import numpy as np

from core.document import add_anchor, add_path, add_segment, create_document
from core.types import Anchor, CoordinateSystem, Path as VectorPath, Segment
from services.renderer import Renderer
from services.segment_sampler import SegmentSampler


def _renderer_document(y_axis: str = "down"):
    document = create_document(
        document_id="doc_render",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(
            y_axis=y_axis,
            unit="px",
            precision=4,
            view_box=(0.0, 0.0, 100.0, 100.0),
        ),
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [
                        {
                            "contour_id": "binary_1",
                            "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 30.0], [10.0, 30.0]],
                            "coordinate_space": "vector",
                            "closed": True,
                        }
                    ],
                    "skeleton_contours": [
                        {
                            "contour_id": "skeleton_1",
                            "points": [[50.0, 20.0], [60.0, 20.0], [70.0, 25.0]],
                            "coordinate_space": "vector",
                            "closed": False,
                        }
                    ],
                }
            }
        },
    )
    line_path = VectorPath(path_id="path_line")
    bezier_path = VectorPath(path_id="path_bezier")
    line_start = Anchor(anchor_id="line_a1", path_id="path_line", position=(10.0, 10.0))
    line_end = Anchor(anchor_id="line_a2", path_id="path_line", position=(40.0, 10.0))
    bezier_start = Anchor(anchor_id="bezier_a1", path_id="path_bezier", position=(50.0, 20.0), out_handle=(60.0, 0.0))
    bezier_end = Anchor(anchor_id="bezier_a2", path_id="path_bezier", position=(80.0, 20.0), in_handle=(70.0, 40.0))
    line_segment = Segment(
        segment_id="line_seg",
        path_id="path_line",
        type="line",
        params={"start": [10.0, 10.0], "end": [40.0, 10.0]},
        anchors=("line_a1", "line_a2"),
    )
    bezier_segment = Segment(
        segment_id="bezier_seg",
        path_id="path_bezier",
        type="bezier",
        params={
            "start": [50.0, 20.0],
            "control1": [60.0, 0.0],
            "control2": [70.0, 40.0],
            "end": [80.0, 20.0],
        },
        anchors=("bezier_a1", "bezier_a2"),
    )

    document = add_path(document, line_path)
    document = add_path(document, bezier_path)
    for anchor in (line_start, line_end, bezier_start, bezier_end):
        document = add_anchor(document, anchor)
    for segment in (line_segment, bezier_segment):
        document = add_segment(document, segment)
    return document


def _curved_renderer_document() -> object:
    document = create_document(
        document_id="doc_render_curves",
        width=120.0,
        height=120.0,
        coordinate_system=CoordinateSystem(
            y_axis="down",
            unit="px",
            precision=4,
            view_box=(0.0, 0.0, 120.0, 120.0),
        ),
    )
    path = VectorPath(path_id="path_curves")
    document = add_path(document, path)
    for segment in (
        Segment(
            segment_id="arc_seg",
            path_id="path_curves",
            type="arc",
            params={"cx": 30.0, "cy": 30.0, "r": 12.0, "start_angle": 0.0, "end_angle": math.pi / 2.0, "direction": "ccw"},
        ),
        Segment(
            segment_id="circle_seg",
            path_id="path_curves",
            type="circle",
            params={"cx": 70.0, "cy": 75.0, "r": 10.0},
        ),
        Segment(
            segment_id="ellipse_seg",
            path_id="path_curves",
            type="ellipse",
            params={"cx": 90.0, "cy": 35.0, "rx": 14.0, "ry": 6.0, "rotation": math.pi / 6.0},
        ),
    ):
        document = add_segment(document, segment)
    return document


def test_renderer_renders_overlay_with_source_contours_segments_and_controls() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    document = _renderer_document()
    renderer = Renderer()

    overlay = renderer.render_overlay(document, image)
    encoded = renderer.export_overlay_png(document, image)

    assert overlay.shape == (100, 100, 3)
    assert not np.array_equal(overlay, image)
    assert overlay[10, 10, 2] > 0
    assert overlay[20, 50].sum() > 0
    assert overlay[0, 60].sum() > 0
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_renderer_uses_vector_to_pixel_for_y_axis_flip() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    document = _renderer_document(y_axis="up")
    renderer = Renderer()

    overlay = renderer.render_overlay(document, image)

    assert overlay[90, 10, 2] > 0
    assert overlay[10, 10].sum() == 0


def test_renderer_does_not_mutate_document() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    document = _renderer_document()
    original_document = document
    renderer = Renderer()

    _ = renderer.render_overlay(document, image)

    assert document == original_document


def test_renderer_draws_arc_circle_and_ellipse_curves_not_just_anchors() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    document = _curved_renderer_document()
    renderer = Renderer()
    sampler = SegmentSampler()

    overlay = renderer.render_overlay(document, image)

    for segment in document.segments:
        sampled_points = sampler.sample_segment(segment)
        mid_point = sampled_points[len(sampled_points) // 2]
        pixel = (int(round(mid_point[0])), int(round(mid_point[1])))
        assert overlay[pixel[1], pixel[0]].sum() > 0


def test_renderer_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/renderer.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"PyQt5", "PyQt6", "openai", "anthropic", "ui"}

    assert imports.isdisjoint(forbidden_imports)
    assert "core.document" not in source
