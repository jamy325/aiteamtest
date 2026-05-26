from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.types import Point, VectorDocument
from services.contour_extractor import BinaryContour


@dataclass(frozen=True, slots=True)
class StrokeWidthEstimate:
    stroke_width: float
    confidence: float
    reason: str
    sample_count: int


@dataclass(frozen=True, slots=True)
class StrokeWidthEstimatorConfig:
    alpha_threshold: int = 8
    max_default_stroke_width: float = 1.0
    min_valid_samples: int = 3


class StrokeWidthEstimator:
    def __init__(self, config: StrokeWidthEstimatorConfig | None = None) -> None:
        self.config = config or StrokeWidthEstimatorConfig()

    def estimate_for_contour(
        self,
        document: VectorDocument,
        contour: BinaryContour,
        image: np.ndarray,
    ) -> StrokeWidthEstimate:
        transformer = CoordinateTransformer(document.coordinate_system)
        foreground_mask = self._foreground_mask(image)
        if not np.any(foreground_mask):
            return self._default_estimate("foreground_mask_unavailable")

        distance = cv2.distanceTransform(foreground_mask, cv2.DIST_L2, 3)
        radii: list[float] = []
        for point in contour.points:
            px, py = self._point_to_pixel(transformer, point)
            if py < 0 or px < 0 or py >= distance.shape[0] or px >= distance.shape[1]:
                continue
            radius = float(distance[py, px])
            if radius > 0.0:
                radii.append(radius)

        if len(radii) < self.config.min_valid_samples:
            return self._default_estimate("insufficient_valid_samples", sample_count=len(radii))

        widths = [max(1.0, radius * 2.0) for radius in radii]
        stroke_width = float(np.median(np.asarray(widths, dtype=np.float32)))
        normalized_std = float(np.std(np.asarray(widths, dtype=np.float32)) / max(stroke_width, 1e-6))
        sample_ratio = min(1.0, len(radii) / max(len(contour.points), 1))
        confidence = max(0.0, min(1.0, (0.7 * sample_ratio) + (0.3 * max(0.0, 1.0 - normalized_std))))
        return StrokeWidthEstimate(
            stroke_width=stroke_width,
            confidence=confidence,
            reason="distance_transform_estimate",
            sample_count=len(radii),
        )

    def _foreground_mask(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            grayscale = image.astype(np.uint8)
            _, mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return self._normalize_foreground_mask(mask)

        if image.ndim != 3:
            return np.zeros((1, 1), dtype=np.uint8)

        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            if np.count_nonzero(alpha > self.config.alpha_threshold) > 0:
                return (alpha > self.config.alpha_threshold).astype(np.uint8) * 255
            grayscale = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return self._normalize_foreground_mask(mask)

        grayscale = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._normalize_foreground_mask(mask)

    def _normalize_foreground_mask(self, mask: np.ndarray) -> np.ndarray:
        white_ratio = float(np.count_nonzero(mask)) / float(mask.size or 1)
        if white_ratio > 0.5:
            mask = cv2.bitwise_not(mask)
        return mask.astype(np.uint8)

    def _default_estimate(self, reason: str, *, sample_count: int = 0) -> StrokeWidthEstimate:
        return StrokeWidthEstimate(
            stroke_width=self.config.max_default_stroke_width,
            confidence=0.0,
            reason=reason,
            sample_count=sample_count,
        )

    def _point_to_pixel(self, transformer: CoordinateTransformer, point: Point) -> tuple[int, int]:
        pixel = transformer.vector_to_pixel((float(point[0]), float(point[1])))
        return (int(round(pixel[0])), int(round(pixel[1])))


__all__ = [
    "StrokeWidthEstimate",
    "StrokeWidthEstimator",
    "StrokeWidthEstimatorConfig",
]
