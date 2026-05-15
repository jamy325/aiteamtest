from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.types import Path, Point, Style, VectorDocument
from services.segment_sampler import SegmentSampler, SegmentSamplerConfig


@dataclass(frozen=True, slots=True)
class AlphaAwareStyleAnalyzerConfig:
    alpha_threshold: int = 8
    erosion_kernel_size: int = 3
    min_eroded_pixels: int = 9
    line_sample_step: float = 1.0
    bezier_segments: int = 48
    circle_segments: int = 96
    ellipse_segments: int = 96
    min_segments_per_arc: int = 16
    max_segments_per_arc: int = 192
    max_chord_error: float = 0.2


class AlphaAwareStyleAnalyzer:
    def __init__(self, config: AlphaAwareStyleAnalyzerConfig | None = None) -> None:
        self.config = config or AlphaAwareStyleAnalyzerConfig()
        self.segment_sampler = SegmentSampler(
            SegmentSamplerConfig(
                line_sample_step=self.config.line_sample_step,
                bezier_segments=self.config.bezier_segments,
                circle_segments=self.config.circle_segments,
                ellipse_segments=self.config.ellipse_segments,
                min_segments_per_arc=self.config.min_segments_per_arc,
                max_segments_per_arc=self.config.max_segments_per_arc,
                max_chord_error=self.config.max_chord_error,
            )
        )

    def analyze_path_style(self, document: VectorDocument, path_id: str, image: np.ndarray) -> Style:
        path = self._path_by_id(document, path_id)
        if not path.closed:
            raise ValueError("style analysis requires a closed path")

        mask = self._build_path_mask(document, path, image.shape[:2])
        if not np.any(mask):
            raise ValueError("path mask is empty")

        rgb, alpha = self._extract_rgb_and_alpha(image)
        sample_mask = self._build_sample_mask(mask, alpha)
        if not np.any(sample_mask):
            raise ValueError("no reliable pixels available for style sampling")

        sampled_rgb = rgb[sample_mask]
        sampled_alpha = alpha[sample_mask].astype(np.float32) / 255.0
        median_rgb = np.median(sampled_rgb, axis=0)
        fill_color = tuple(int(round(float(channel))) for channel in median_rgb)
        fill_alpha = float(np.median(sampled_alpha))

        color_variance = self._color_variance(sampled_rgb, median_rgb)
        alpha_variance = float(np.std(sampled_alpha))
        color_confidence = self._color_confidence(
            color_variance=color_variance,
            alpha_variance=alpha_variance,
            sample_ratio=float(np.count_nonzero(sample_mask)) / float(np.count_nonzero(mask)),
        )

        return Style(
            fill_color=fill_color,
            fill_alpha=fill_alpha,
            color_confidence=color_confidence,
            color_variance=color_variance,
            alpha_variance=alpha_variance,
            metadata={
                "sample_pixel_count": int(np.count_nonzero(sample_mask)),
                "mask_pixel_count": int(np.count_nonzero(mask)),
            },
        )

    def _path_by_id(self, document: VectorDocument, path_id: str) -> Path:
        for path in document.paths:
            if path.path_id == path_id:
                return path
        raise ValueError(f"unknown path_id: {path_id}")

    def _build_path_mask(self, document: VectorDocument, path: Path, image_shape: tuple[int, int]) -> np.ndarray:
        polygon = self._path_polygon(document, path)
        if len(polygon) < 3:
            raise ValueError("path polygon requires at least three points")

        transformer = CoordinateTransformer(document.coordinate_system)
        pixels = np.array(
            [[int(round(point[0])), int(round(point[1]))] for point in self._polygon_to_pixels(transformer, polygon)],
            dtype=np.int32,
        )
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [pixels], 255)
        return mask > 0

    def _path_polygon(self, document: VectorDocument, path: Path) -> tuple[Point, ...]:
        segment_by_id = {segment.segment_id: segment for segment in document.segments}
        polygon: list[Point] = []

        for segment_id in path.segments:
            segment = segment_by_id.get(segment_id)
            if segment is None:
                continue
            sampled = list(self.segment_sampler.sample_segment(segment))
            if not sampled:
                continue
            if polygon and self._points_close(polygon[-1], sampled[0]):
                sampled = sampled[1:]
            polygon.extend(sampled)

        if len(polygon) > 1 and self._points_close(polygon[0], polygon[-1]):
            polygon.pop()
        return tuple(polygon)

    def _polygon_to_pixels(
        self,
        transformer: CoordinateTransformer,
        polygon: tuple[Point, ...],
    ) -> tuple[Point, ...]:
        return tuple(transformer.vector_to_pixel(point) for point in polygon)

    def _extract_rgb_and_alpha(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim == 2:
            grayscale = image.astype(np.uint8)
            rgb = np.stack([grayscale, grayscale, grayscale], axis=-1)
            alpha = np.full(grayscale.shape, 255, dtype=np.uint8)
            return rgb, alpha

        if image.ndim != 3:
            raise ValueError("image must be grayscale, BGR, or BGRA")

        if image.shape[2] == 3:
            bgr = image.astype(np.uint8)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
            return rgb, alpha

        if image.shape[2] == 4:
            bgra = image.astype(np.uint8)
            rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
            alpha = bgra[:, :, 3]
            return rgb, alpha

        raise ValueError("image must have 1, 3, or 4 channels")

    def _build_sample_mask(self, mask: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        reliable = mask & (alpha > int(self.config.alpha_threshold))
        if not np.any(reliable):
            return reliable

        kernel_size = max(1, int(self.config.erosion_kernel_size))
        if kernel_size == 1:
            return reliable

        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(reliable.astype(np.uint8) * 255, kernel, iterations=1) > 0
        if int(np.count_nonzero(eroded)) >= int(self.config.min_eroded_pixels):
            return eroded
        return reliable

    @staticmethod
    def _color_variance(sampled_rgb: np.ndarray, median_rgb: np.ndarray) -> float:
        deviations = np.linalg.norm(sampled_rgb.astype(np.float32) - median_rgb.astype(np.float32), axis=1)
        return float(np.mean(deviations)) if len(deviations) else 0.0

    @staticmethod
    def _color_confidence(*, color_variance: float, alpha_variance: float, sample_ratio: float) -> float:
        color_term = max(0.0, 1.0 - (float(color_variance) / 64.0))
        alpha_term = max(0.0, 1.0 - (float(alpha_variance) / 0.25))
        ratio_term = max(0.0, min(1.0, float(sample_ratio)))
        return max(0.0, min(1.0, (0.6 * color_term) + (0.2 * alpha_term) + (0.2 * ratio_term)))

    @staticmethod
    def _points_close(left: Point, right: Point) -> bool:
        return math.dist(left, right) <= 1e-6


__all__ = ["AlphaAwareStyleAnalyzer", "AlphaAwareStyleAnalyzerConfig"]
