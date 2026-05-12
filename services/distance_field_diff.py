from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.types import Point, VectorDocument
from services.edge_error import EdgeErrorCalculator, EdgeErrorResult


@dataclass(frozen=True, slots=True)
class DistanceFieldDiffOptions:
    sample_step: float = 1.0
    missing_color: tuple[int, int, int] = (0, 0, 255)
    overdraw_color: tuple[int, int, int] = (255, 0, 0)
    overlap_color: tuple[int, int, int] = (80, 80, 80)
    max_visual_distance: float = 8.0
    line_thickness: int = 1


@dataclass(frozen=True, slots=True)
class DistanceFieldDiffResult:
    image: np.ndarray
    missing_edge_error: float
    overdraw_error: float
    chamfer_error: float
    source_point_count: int
    vector_point_count: int


class DistanceFieldDiffRenderer:
    def __init__(
        self,
        options: DistanceFieldDiffOptions | None = None,
        edge_error_calculator: EdgeErrorCalculator | None = None,
    ) -> None:
        self.options = options or DistanceFieldDiffOptions()
        self.edge_error_calculator = edge_error_calculator or EdgeErrorCalculator()

    def render_diff(self, document: VectorDocument) -> DistanceFieldDiffResult:
        transformer = CoordinateTransformer(document.coordinate_system)
        height, width = self._canvas_size(document)
        source_vector_points, source_polylines = self._source_contours(document, transformer)
        vector_points, vector_polylines = self._vector_segments(document, transformer)

        source_mask = self._rasterize_polylines(source_polylines, width, height)
        vector_mask = self._rasterize_polylines(vector_polylines, width, height)
        diff_image = self._build_diff_image(source_mask, vector_mask)
        edge_error = self.edge_error_calculator.calculate(source_vector_points, vector_points)

        return DistanceFieldDiffResult(
            image=diff_image,
            missing_edge_error=edge_error.missing_edge_error,
            overdraw_error=edge_error.overdraw_error,
            chamfer_error=edge_error.chamfer_error,
            source_point_count=edge_error.source_point_count,
            vector_point_count=edge_error.vector_point_count,
        )

    def export_diff_png(self, document: VectorDocument) -> bytes:
        result = self.render_diff(document)
        success, encoded = cv2.imencode(".png", result.image)
        if not success:
            raise ValueError("failed to encode distance field diff image")
        return encoded.tobytes()

    def _canvas_size(self, document: VectorDocument) -> tuple[int, int]:
        if document.coordinate_system.view_box is not None:
            _, _, width, height = document.coordinate_system.view_box
            return (max(1, int(round(height))), max(1, int(round(width))))
        return (max(1, int(round(document.height))), max(1, int(round(document.width))))

    def _source_contours(
        self,
        document: VectorDocument,
        transformer: CoordinateTransformer,
    ) -> tuple[tuple[Point, ...], tuple[np.ndarray, ...]]:
        vector_points: list[Point] = []
        polylines: list[np.ndarray] = []
        pipeline = document.metadata.get("pipeline", {})
        source_contours = pipeline.get("source_contours", {})

        for group in ("binary_contours", "skeleton_contours"):
            for contour in source_contours.get(group, ()):
                points = contour.get("points", ())
                coordinate_space = contour.get("coordinate_space", "vector")
                if len(points) < 2:
                    continue
                contour_vector_points = tuple(
                    self._coerce_vector_point(transformer, point, coordinate_space) for point in points
                )
                vector_points.extend(contour_vector_points)
                polylines.append(
                    np.array(
                        [self._point_to_pixel(transformer, point) for point in contour_vector_points],
                        dtype=np.int32,
                    )
                )

        return (tuple(vector_points), tuple(polylines))

    def _vector_segments(
        self,
        document: VectorDocument,
        transformer: CoordinateTransformer,
    ) -> tuple[tuple[Point, ...], tuple[np.ndarray, ...]]:
        sampled_points: list[Point] = []
        polylines: list[np.ndarray] = []

        for segment in document.segments:
            if segment.type == "line":
                segment_points = self._sample_line(segment.params["start"], segment.params["end"])
            elif segment.type == "bezier":
                segment_points = self._sample_bezier(
                    segment.params["start"],
                    segment.params["control1"],
                    segment.params["control2"],
                    segment.params["end"],
                )
            else:
                continue

            if len(segment_points) < 2:
                continue
            sampled_points.extend(segment_points)
            polylines.append(
                np.array([self._point_to_pixel(transformer, point) for point in segment_points], dtype=np.int32)
            )

        return (tuple(sampled_points), tuple(polylines))

    def _sample_line(self, start: Point | list[float], end: Point | list[float]) -> tuple[Point, ...]:
        p0 = self._coerce_point(start)
        p1 = self._coerce_point(end)
        length = math.dist(p0, p1)
        sample_count = self._sample_count(length)
        return tuple(
            (
                p0[0] + (p1[0] - p0[0]) * t,
                p0[1] + (p1[1] - p0[1]) * t,
            )
            for t in self._parameter_steps(sample_count)
        )

    def _sample_bezier(
        self,
        start: Point | list[float],
        control1: Point | list[float],
        control2: Point | list[float],
        end: Point | list[float],
    ) -> tuple[Point, ...]:
        p0 = self._coerce_point(start)
        p1 = self._coerce_point(control1)
        p2 = self._coerce_point(control2)
        p3 = self._coerce_point(end)
        control_polygon_length = math.dist(p0, p1) + math.dist(p1, p2) + math.dist(p2, p3)
        sample_count = self._sample_count(control_polygon_length)
        return tuple(self._cubic_bezier_point(p0, p1, p2, p3, t) for t in self._parameter_steps(sample_count))

    def _sample_count(self, length: float) -> int:
        step = max(float(self.options.sample_step), 1e-6)
        return max(2, int(math.ceil(length / step)) + 1)

    def _parameter_steps(self, sample_count: int) -> tuple[float, ...]:
        if sample_count <= 1:
            return (0.0,)
        return tuple(index / (sample_count - 1) for index in range(sample_count))

    def _cubic_bezier_point(
        self,
        p0: Point,
        p1: Point,
        p2: Point,
        p3: Point,
        t: float,
    ) -> Point:
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

    def _rasterize_polylines(
        self,
        polylines: tuple[np.ndarray, ...],
        width: int,
        height: int,
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for polyline in polylines:
            if len(polyline) < 2:
                continue
            cv2.polylines(mask, [polyline], False, 255, self.options.line_thickness, cv2.LINE_8)
        return mask

    def _build_diff_image(self, source_mask: np.ndarray, vector_mask: np.ndarray) -> np.ndarray:
        image = np.zeros((source_mask.shape[0], source_mask.shape[1], 3), dtype=np.uint8)
        overlap_mask = (source_mask > 0) & (vector_mask > 0)
        missing_mask = (source_mask > 0) & ~overlap_mask
        overdraw_mask = (vector_mask > 0) & ~overlap_mask

        image[overlap_mask] = np.array(self.options.overlap_color, dtype=np.uint8)

        vector_distance = cv2.distanceTransform(255 - vector_mask, cv2.DIST_L2, 3)
        source_distance = cv2.distanceTransform(255 - source_mask, cv2.DIST_L2, 3)

        self._apply_distance_color(image, missing_mask, vector_distance, self.options.missing_color)
        self._apply_distance_color(image, overdraw_mask, source_distance, self.options.overdraw_color)
        return image

    def _apply_distance_color(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        distance_field: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        if not np.any(mask):
            return
        distances = distance_field[mask]
        max_distance = max(float(self.options.max_visual_distance), 1e-6)
        normalized = np.clip(distances / max_distance, 0.0, 1.0)
        intensity = np.maximum((normalized * 255.0).astype(np.uint8), 48)
        color_array = np.array(color, dtype=np.float32)
        colored_pixels = np.clip((intensity[:, None] / 255.0) * color_array[None, :], 0.0, 255.0).astype(np.uint8)
        image[mask] = colored_pixels

    def _point_to_pixel(self, transformer: CoordinateTransformer, point: Point) -> tuple[int, int]:
        pixel = transformer.vector_to_pixel(point)
        return (int(round(pixel[0])), int(round(pixel[1])))

    def _coerce_vector_point(
        self,
        transformer: CoordinateTransformer,
        point: Point | list[float],
        coordinate_space: str,
    ) -> Point:
        raw = self._coerce_point(point)
        if coordinate_space == "pixel":
            return transformer.pixel_to_vector(raw)
        return raw

    def _coerce_point(self, point: Point | list[float]) -> Point:
        return (float(point[0]), float(point[1]))


__all__ = [
    "DistanceFieldDiffOptions",
    "DistanceFieldDiffRenderer",
    "DistanceFieldDiffResult",
]
