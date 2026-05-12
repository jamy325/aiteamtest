from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.types import Point, VectorDocument


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


class Renderer:
    def __init__(self, options: OverlayRenderOptions | None = None) -> None:
        self.options = options or OverlayRenderOptions()

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
            if segment.type == "line":
                start = self._point_to_pixel(transformer, segment.params["start"])
                end = self._point_to_pixel(transformer, segment.params["end"])
                cv2.line(canvas, start, end, self.options.line_color, self.options.segment_thickness, cv2.LINE_AA)
                continue

            if segment.type == "bezier":
                sampled = np.array(
                    [
                        self._point_to_pixel(
                            transformer,
                            self._sample_cubic_bezier(
                                segment.params["start"],
                                segment.params["control1"],
                                segment.params["control2"],
                                segment.params["end"],
                                step / self.options.bezier_samples,
                            ),
                        )
                        for step in range(self.options.bezier_samples + 1)
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(canvas, [sampled], False, self.options.bezier_color, self.options.segment_thickness, cv2.LINE_AA)

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

    def _sample_cubic_bezier(
        self,
        start: Point | list[float],
        control1: Point | list[float],
        control2: Point | list[float],
        end: Point | list[float],
        t: float,
    ) -> Point:
        p0 = (float(start[0]), float(start[1]))
        p1 = (float(control1[0]), float(control1[1]))
        p2 = (float(control2[0]), float(control2[1]))
        p3 = (float(end[0]), float(end[1]))
        one_minus_t = 1.0 - t
        x = (
            (one_minus_t ** 3) * p0[0]
            + 3.0 * (one_minus_t ** 2) * t * p1[0]
            + 3.0 * one_minus_t * (t ** 2) * p2[0]
            + (t ** 3) * p3[0]
        )
        y = (
            (one_minus_t ** 3) * p0[1]
            + 3.0 * (one_minus_t ** 2) * t * p1[1]
            + 3.0 * one_minus_t * (t ** 2) * p2[1]
            + (t ** 3) * p3[1]
        )
        return (x, y)

    def _to_bgr_canvas(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy()
        raise ValueError("image must be grayscale or BGR")


__all__ = ["OverlayRenderOptions", "Renderer"]
