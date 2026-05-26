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
    alpha_foreground_fallback_ratio: float = 0.98
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
        foreground_mask, mask_source = self._foreground_mask(image)
        mask_failure_reason = self._mask_failure_reason(foreground_mask, mask_source)
        if mask_failure_reason is not None:
            return self._default_estimate(mask_failure_reason)

        distance = cv2.distanceTransform(foreground_mask, cv2.DIST_L2, 3)
        if not bool(np.all(np.isfinite(distance))):
            return self._default_estimate("non_finite_distance_transform")

        radii: list[float] = []
        saw_non_finite_radius = False
        for point in contour.points:
            px, py = self._point_to_pixel(transformer, point)
            if py < 0 or px < 0 or py >= distance.shape[0] or px >= distance.shape[1]:
                continue
            radius = float(distance[py, px])
            if not math.isfinite(radius):
                saw_non_finite_radius = True
                continue
            if radius > 0.0:
                radii.append(radius)

        if not radii and saw_non_finite_radius:
            return self._default_estimate("non_finite_distance_transform")

        if len(radii) < self.config.min_valid_samples:
            return self._default_estimate("insufficient_valid_samples", sample_count=len(radii))

        widths = np.asarray(
            [radius * 2.0 for radius in radii if math.isfinite(radius) and radius > 0.0],
            dtype=np.float64,
        )
        widths = widths[np.isfinite(widths) & (widths > 0.0)]
        if int(widths.size) < self.config.min_valid_samples:
            reason = "non_finite_distance_transform" if saw_non_finite_radius else "insufficient_valid_samples"
            return self._default_estimate(reason, sample_count=int(widths.size))

        stroke_width = float(np.median(widths))
        if not math.isfinite(stroke_width) or stroke_width <= 0.0:
            reason = "non_finite_distance_transform" if saw_non_finite_radius else "insufficient_valid_samples"
            return self._default_estimate(reason, sample_count=int(widths.size))

        normalized_std = float(np.std(widths, dtype=np.float64) / max(stroke_width, 1e-6))
        if not math.isfinite(normalized_std):
            normalized_std = 1.0
        sample_ratio = min(1.0, len(radii) / max(len(contour.points), 1))
        confidence = max(0.0, min(1.0, (0.7 * sample_ratio) + (0.3 * max(0.0, 1.0 - normalized_std))))
        if not math.isfinite(confidence):
            confidence = 0.0
        return StrokeWidthEstimate(
            stroke_width=stroke_width,
            confidence=confidence,
            reason="distance_transform_estimate",
            sample_count=len(radii),
        )

    def _foreground_mask(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        if image.ndim == 2:
            grayscale = image.astype(np.uint8)
            return (self._grayscale_foreground_mask(grayscale), "grayscale_otsu")

        if image.ndim != 3:
            return (np.zeros((1, 1), dtype=np.uint8), "foreground_mask_unavailable")

        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            alpha_mask = (alpha > self.config.alpha_threshold).astype(np.uint8) * 255
            alpha_foreground_ratio = float(np.count_nonzero(alpha_mask)) / float(alpha_mask.size or 1)
            if 0.0 < alpha_foreground_ratio < self.config.alpha_foreground_fallback_ratio:
                return (alpha_mask.astype(np.uint8), "alpha_mask")
            grayscale = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
            return (self._grayscale_foreground_mask(grayscale), "invalid_alpha_mask")

        grayscale = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        return (self._grayscale_foreground_mask(grayscale), "grayscale_otsu")

    def _grayscale_foreground_mask(self, grayscale: np.ndarray) -> np.ndarray:
        _, mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._normalize_foreground_mask(mask)

    def _normalize_foreground_mask(self, mask: np.ndarray) -> np.ndarray:
        white_ratio = float(np.count_nonzero(mask)) / float(mask.size or 1)
        if white_ratio > 0.5:
            mask = cv2.bitwise_not(mask)
        return mask.astype(np.uint8)

    def _mask_failure_reason(self, mask: np.ndarray, mask_source: str) -> str | None:
        if mask.size == 0:
            return "foreground_mask_unavailable"
        if not bool(np.any(mask)):
            return "invalid_alpha_mask" if mask_source == "invalid_alpha_mask" else "all_background_mask"
        if bool(np.all(mask > 0)):
            return "invalid_alpha_mask" if mask_source == "invalid_alpha_mask" else "all_foreground_mask"
        return None

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
