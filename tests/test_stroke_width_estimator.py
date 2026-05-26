from __future__ import annotations

import math

import cv2
import numpy as np

from core.document import create_document
from core.types import CoordinateSystem
from services.contour_extractor import BinaryContour
from services.stroke_width_estimator import StrokeWidthEstimator


def _document(width: int = 64, height: int = 64):
    return create_document(
        document_id="stroke_width_doc",
        width=float(width),
        height=float(height),
        coordinate_system=CoordinateSystem(view_box=(0.0, 0.0, float(width), float(height))),
    )


def _line_contour(width: int = 64, y: int = 32) -> BinaryContour:
    points = tuple((float(x), float(y)) for x in range(10, width - 10, 4))
    return BinaryContour(
        contour_id="skeleton_contour_0",
        source="skeleton_contour",
        points=points,
        coordinate_space="vector",
        closed=False,
        area=0.0,
        depth=0,
        parent_contour=None,
        children=(),
    )


def _rgba_white_background_black_line(width: int = 64, height: int = 64) -> np.ndarray:
    image = np.full((height, width, 4), 255, dtype=np.uint8)
    cv2.line(image, (10, height // 2), (width - 10, height // 2), (0, 0, 0, 255), thickness=5)
    return image


def test_stroke_width_estimator_falls_back_from_full_alpha_rgba_to_grayscale() -> None:
    estimator = StrokeWidthEstimator()

    estimate = estimator.estimate_for_contour(_document(), _line_contour(), _rgba_white_background_black_line())

    assert math.isfinite(estimate.stroke_width)
    assert estimate.stroke_width > 0.0
    assert estimate.reason == "distance_transform_estimate"
    assert estimate.sample_count >= 3


def test_stroke_width_estimator_rejects_all_opaque_white_rgba_as_invalid_alpha_mask() -> None:
    estimator = StrokeWidthEstimator()
    image = np.full((64, 64, 4), 255, dtype=np.uint8)

    estimate = estimator.estimate_for_contour(_document(), _line_contour(), image)

    assert estimate.reason == "invalid_alpha_mask"
    assert math.isfinite(estimate.stroke_width)
    assert estimate.stroke_width > 0.0
    assert estimate.confidence == 0.0


def test_stroke_width_estimator_rejects_all_foreground_mask(monkeypatch) -> None:
    estimator = StrokeWidthEstimator()
    monkeypatch.setattr(
        estimator,
        "_foreground_mask",
        lambda _image: (np.full((32, 32), 255, dtype=np.uint8), "grayscale_otsu"),
    )

    estimate = estimator.estimate_for_contour(_document(), _line_contour(), np.zeros((32, 32), dtype=np.uint8))

    assert estimate.reason == "all_foreground_mask"
    assert math.isfinite(estimate.stroke_width)
    assert estimate.stroke_width > 0.0


def test_stroke_width_estimator_rejects_all_background_mask(monkeypatch) -> None:
    estimator = StrokeWidthEstimator()
    monkeypatch.setattr(
        estimator,
        "_foreground_mask",
        lambda _image: (np.zeros((32, 32), dtype=np.uint8), "grayscale_otsu"),
    )

    estimate = estimator.estimate_for_contour(_document(), _line_contour(), np.zeros((32, 32), dtype=np.uint8))

    assert estimate.reason == "all_background_mask"
    assert math.isfinite(estimate.stroke_width)
    assert estimate.stroke_width > 0.0


def test_stroke_width_estimator_handles_non_finite_distance_transform(monkeypatch) -> None:
    estimator = StrokeWidthEstimator()
    monkeypatch.setattr(
        estimator,
        "_foreground_mask",
        lambda _image: (cv2.rectangle(np.zeros((32, 32), dtype=np.uint8), (8, 8), (24, 24), 255, thickness=-1), "grayscale_otsu"),
    )
    monkeypatch.setattr(
        cv2,
        "distanceTransform",
        lambda *_args, **_kwargs: np.full((32, 32), np.inf, dtype=np.float32),
    )

    estimate = estimator.estimate_for_contour(_document(32, 32), _line_contour(32, 16), np.zeros((32, 32), dtype=np.uint8))

    assert estimate.reason == "non_finite_distance_transform"
    assert math.isfinite(estimate.stroke_width)
    assert estimate.stroke_width > 0.0
