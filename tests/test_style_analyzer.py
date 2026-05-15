from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.style_analyzer import AlphaAwareStyleAnalyzer


def _rectangle_document(*, width: int = 80, height: int = 80) -> object:
    document = create_document(
        document_id="doc_style_rect",
        width=float(width),
        height=float(height),
        coordinate_system=CoordinateSystem(
            y_axis="down",
            unit="px",
            precision=4,
            view_box=(0.0, 0.0, float(width), float(height)),
        ),
    )
    path = VectorPath(path_id="rect", closed=True, segments=("s0", "s1", "s2", "s3"))
    segments = (
        Segment("s0", "rect", "line", {"start": [20.0, 20.0], "end": [60.0, 20.0]}),
        Segment("s1", "rect", "line", {"start": [60.0, 20.0], "end": [60.0, 60.0]}),
        Segment("s2", "rect", "line", {"start": [60.0, 60.0], "end": [20.0, 60.0]}),
        Segment("s3", "rect", "line", {"start": [20.0, 60.0], "end": [20.0, 20.0]}),
    )
    document = add_path(document, path)
    for segment in segments:
        document = add_segment(document, segment)
    return document


def _circle_document(*, width: int = 96, height: int = 96) -> object:
    document = create_document(
        document_id="doc_style_circle",
        width=float(width),
        height=float(height),
        coordinate_system=CoordinateSystem(
            y_axis="down",
            unit="px",
            precision=4,
            view_box=(0.0, 0.0, float(width), float(height)),
        ),
    )
    path = VectorPath(path_id="circle", closed=True, segments=("circle_seg",))
    segment = Segment("circle_seg", "circle", "circle", {"cx": 48.0, "cy": 48.0, "r": 18.0})
    document = add_path(document, path)
    document = add_segment(document, segment)
    return document


def _supersampled_circle_image(
    *,
    size: int,
    center: tuple[int, int],
    radius: int,
    fill_bgr: tuple[int, int, int],
    background_bgr: tuple[int, int, int],
    scale: int = 4,
) -> np.ndarray:
    high_size = size * scale
    image = np.zeros((high_size, high_size, 3), dtype=np.uint8)
    image[:] = background_bgr
    cv2.circle(
        image,
        (center[0] * scale, center[1] * scale),
        radius * scale,
        fill_bgr,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def test_style_analyzer_samples_pure_solid_color_from_closed_circle() -> None:
    document = _circle_document()
    image = np.zeros((96, 96, 4), dtype=np.uint8)
    cv2.circle(image, (48, 48), 18, (60, 180, 20, 255), thickness=-1)

    style = AlphaAwareStyleAnalyzer().analyze_path_style(document, "circle", image)

    assert style.fill_color == (20, 180, 60)
    assert style.fill_alpha == pytest.approx(1.0)
    assert style.color_confidence is not None and style.color_confidence > 0.95
    assert style.color_variance == pytest.approx(0.0)
    assert style.alpha_variance == pytest.approx(0.0)
    assert style.paint_type == "solid"


def test_style_analyzer_handles_transparent_background_and_reports_fill_alpha() -> None:
    document = _rectangle_document()
    image = np.zeros((80, 80, 4), dtype=np.uint8)
    image[20:61, 20:61] = (25, 200, 40, 128)

    style = AlphaAwareStyleAnalyzer().analyze_path_style(document, "rect", image)

    assert style.fill_color == (40, 200, 25)
    assert style.fill_alpha == pytest.approx(128.0 / 255.0, abs=0.01)
    assert style.color_confidence is not None and style.color_confidence > 0.85
    assert style.paint_type == "solid"


def test_style_analyzer_avoids_antialiased_edge_contamination() -> None:
    document = _circle_document()
    image = _supersampled_circle_image(
        size=96,
        center=(48, 48),
        radius=18,
        fill_bgr=(30, 200, 20),
        background_bgr=(20, 30, 220),
    )

    style = AlphaAwareStyleAnalyzer().analyze_path_style(document, "circle", image)

    assert style.fill_color is not None
    red, green, blue = style.fill_color
    assert green > 180
    assert red < 40
    assert blue < 60
    assert style.color_confidence is not None and style.color_confidence > 0.85
    assert style.paint_type == "solid"


def test_style_analyzer_supports_images_without_alpha_channel() -> None:
    document = _rectangle_document()
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = (200, 40, 30)
    cv2.rectangle(image, (20, 20), (60, 60), (90, 150, 15), thickness=-1)

    style = AlphaAwareStyleAnalyzer().analyze_path_style(document, "rect", image)

    assert style.fill_color == (15, 150, 90)
    assert style.fill_alpha == pytest.approx(1.0)
    assert style.color_confidence is not None and style.color_confidence > 0.95
    assert style.paint_type == "solid"


def test_style_analyzer_marks_high_color_variance_as_gradient_candidate() -> None:
    document = _rectangle_document()
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = (10, 10, 10)
    image[20:61, 20:41] = (0, 0, 255)
    image[20:61, 41:61] = (255, 0, 0)

    style = AlphaAwareStyleAnalyzer().analyze_path_style(document, "rect", image)

    assert style.color_variance is not None and style.color_variance > 28.0
    assert style.alpha_variance == pytest.approx(0.0)
    assert style.paint_type == "gradient_candidate"


def test_style_analyzer_marks_high_alpha_variance_as_transparency_candidate() -> None:
    document = _rectangle_document()
    image = np.zeros((80, 80, 4), dtype=np.uint8)
    image[20:61, 20:61, :3] = (40, 160, 60)
    image[20:61, 20:41, 3] = 255
    image[20:61, 41:61, 3] = 64

    style = AlphaAwareStyleAnalyzer().analyze_path_style(document, "rect", image)

    assert style.color_variance is not None and style.color_variance < 8.0
    assert style.alpha_variance is not None and style.alpha_variance > 0.12
    assert style.paint_type == "transparency_candidate"


def test_style_analyzer_marks_intermediate_variance_as_unknown() -> None:
    document = _rectangle_document()
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = (0, 0, 0)
    image[20:61, 20:41] = (90, 120, 60)
    image[20:61, 41:61] = (100, 130, 70)

    style = AlphaAwareStyleAnalyzer().analyze_path_style(document, "rect", image)

    assert style.color_variance is not None
    assert 8.0 < style.color_variance < 28.0
    assert style.alpha_variance == pytest.approx(0.0)
    assert style.paint_type == "unknown"


def test_style_analyzer_has_no_forbidden_dependencies() -> None:
    source = Path("services/style_analyzer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"openai", "anthropic", "PyQt5", "PyQt6", "ui"}
    assert imports.isdisjoint(forbidden_imports)
    assert "VectorDocument(" not in source
