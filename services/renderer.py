from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.types import Point, VectorDocument
from services.segment_sampler import SegmentSampler, SegmentSamplerConfig


@dataclass(frozen=True, slots=True)
class OverlayRenderOptions:
    show_original_contours: bool = True
    show_control_points: bool = True
    contour_color: tuple[int, int, int] = (120, 220, 220)
    line_color: tuple[int, int, int] = (0, 180, 0)
    bezier_color: tuple[int, int, int] = (180, 0, 180)
    anchor_color: tuple[int, int, int] = (0, 0, 255)
    control_color: tuple[int, int, int] = (255, 120, 0)
    handle_color: tuple[int, int, int] = (200, 120, 0)
    contour_thickness: int = 1
    segment_thickness: int = 2
    anchor_radius: int = 3
    control_radius: int = 2
    bezier_samples: int = 24
    max_chord_error: float = 0.25
    min_segments_per_arc: int = 8
    max_segments_per_arc: int = 128
    circle_segments: int = 64
    ellipse_segments: int = 64


class Renderer:
    def __init__(self, options: OverlayRenderOptions | None = None) -> None:
        self.options = options or OverlayRenderOptions()
        self.segment_sampler = SegmentSampler(
            SegmentSamplerConfig(
                max_chord_error=self.options.max_chord_error,
                min_segments_per_arc=self.options.min_segments_per_arc,
                max_segments_per_arc=self.options.max_segments_per_arc,
                circle_segments=self.options.circle_segments,
                ellipse_segments=self.options.ellipse_segments,
                bezier_segments=self.options.bezier_samples,
            )
        )

    def render_overlay(self, document: VectorDocument, image: np.ndarray) -> np.ndarray:
        canvas = self._to_bgr_canvas(image)
        transformer = CoordinateTransformer(document.coordinate_system)

        if self.options.show_original_contours:
            self._draw_source_contours(canvas, document, transformer)
        self._draw_segments(canvas, document, transformer)
        if self.options.show_control_points:
            self._draw_controls(canvas, document, transformer)
        self._draw_anchors(canvas, document, transformer)
        return canvas

    def export_overlay_png(self, document: VectorDocument, image: np.ndarray) -> bytes:
        overlay = self.render_overlay(document, image)
        success, encoded = cv2.imencode(".png", overlay)
        if not success:
            raise ValueError("failed to encode overlay image")
        return encoded.tobytes()

    def _draw_source_contours(
        self,
        canvas: np.ndarray,
        document: VectorDocument,
        transformer: CoordinateTransformer,
    ) -> None:
        pipeline = document.metadata.get("pipeline", {})
        source_contours = pipeline.get("source_contours", {})
        for group in ("binary_contours", "skeleton_contours"):
            for contour in source_contours.get(group, ()):
                points = contour.get("points", ())
                coordinate_space = contour.get("coordinate_space", "vector")
                if len(points) < 2:
                    continue
                polyline = np.array(
                    [self._point_to_pixel(transformer, tuple(point), coordinate_space) for point in points],
                    dtype=np.int32,
                )
                cv2.polylines(
                    canvas,
                    [polyline],
                    bool(contour.get("closed", False)),
                    self.options.contour_color,
                    self.options.contour_thickness,
                    cv2.LINE_AA,
                )

    def _draw_segments(
        self,
        canvas: np.ndarray,
        document: VectorDocument,
        transformer: CoordinateTransformer,
    ) -> None:
        for segment in document.segments:
            sampled_points = self.segment_sampler.sample_segment(segment)
            if len(sampled_points) < 2:
                continue
            sampled = np.array([self._point_to_pixel(transformer, point) for point in sampled_points], dtype=np.int32)
            cv2.polylines(
                canvas,
                [sampled],
                self.segment_sampler.is_closed(segment),
                self._segment_color(segment.type),
                self.options.segment_thickness,
                cv2.LINE_AA,
            )

    def _draw_controls(
        self,
        canvas: np.ndarray,
        document: VectorDocument,
        transformer: CoordinateTransformer,
    ) -> None:
        for anchor in document.anchors:
            anchor_pixel = self._point_to_pixel(transformer, anchor.position)
            if anchor.in_handle is not None:
                in_handle = self._point_to_pixel(transformer, anchor.in_handle)
                cv2.line(canvas, anchor_pixel, in_handle, self.options.handle_color, 1, cv2.LINE_AA)
                cv2.circle(canvas, in_handle, self.options.control_radius, self.options.control_color, -1)
            if anchor.out_handle is not None:
                out_handle = self._point_to_pixel(transformer, anchor.out_handle)
                cv2.line(canvas, anchor_pixel, out_handle, self.options.handle_color, 1, cv2.LINE_AA)
                cv2.circle(canvas, out_handle, self.options.control_radius, self.options.control_color, -1)

    def _draw_anchors(
        self,
        canvas: np.ndarray,
        document: VectorDocument,
        transformer: CoordinateTransformer,
    ) -> None:
        for anchor in document.anchors:
            cv2.circle(
                canvas,
                self._point_to_pixel(transformer, anchor.position),
                self.options.anchor_radius,
                self.options.anchor_color,
                -1,
            )

    def _point_to_pixel(
        self,
        transformer: CoordinateTransformer,
        point: Point | list[float],
        coordinate_space: str = "vector",
    ) -> tuple[int, int]:
        raw = (float(point[0]), float(point[1]))
        pixel = raw if coordinate_space == "pixel" else transformer.vector_to_pixel(raw)
        return (int(round(pixel[0])), int(round(pixel[1])))

    def _segment_color(self, segment_type: str) -> tuple[int, int, int]:
        if segment_type == "bezier":
            return self.options.bezier_color
        return self.options.line_color

    def _to_bgr_canvas(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy()
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError("image must be grayscale, BGR, or BGRA")


__all__ = ["OverlayRenderOptions", "Renderer"]
