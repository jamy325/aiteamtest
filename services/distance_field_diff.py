from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.types import Point, VectorDocument
from services.edge_error import EdgeErrorCalculator, EdgeErrorResult
from services.segment_sampler import SegmentSampler, SegmentSamplerConfig


@dataclass(frozen=True, slots=True)
class DistanceFieldDiffOptions:
    sample_step: float = 1.0
    max_chord_error: float = 0.25
    min_segments_per_arc: int = 8
    max_segments_per_arc: int = 128
    circle_segments: int = 64
    ellipse_segments: int = 64
    bezier_segments: int = 24
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


@dataclass(frozen=True, slots=True)
class RasterPolyline:
    points: np.ndarray
    closed: bool = False


class DistanceFieldDiffRenderer:
    def __init__(
        self,
        options: DistanceFieldDiffOptions | None = None,
        edge_error_calculator: EdgeErrorCalculator | None = None,
    ) -> None:
        self.options = options or DistanceFieldDiffOptions()
        self.edge_error_calculator = edge_error_calculator or EdgeErrorCalculator()
        self.segment_sampler = SegmentSampler(
            SegmentSamplerConfig(
                max_chord_error=self.options.max_chord_error,
                min_segments_per_arc=self.options.min_segments_per_arc,
                max_segments_per_arc=self.options.max_segments_per_arc,
                circle_segments=self.options.circle_segments,
                ellipse_segments=self.options.ellipse_segments,
                bezier_segments=self.options.bezier_segments,
                line_sample_step=self.options.sample_step,
            )
        )

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
        return (max(1, int(round(document.height))), max(1, int(round(document.width))))

    def _source_contours(
        self,
        document: VectorDocument,
        transformer: CoordinateTransformer,
    ) -> tuple[tuple[Point, ...], tuple[RasterPolyline, ...]]:
        vector_points: list[Point] = []
        polylines: list[RasterPolyline] = []
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
                    RasterPolyline(
                        points=np.array(
                            [self._point_to_pixel(transformer, point) for point in contour_vector_points],
                            dtype=np.int32,
                        ),
                        closed=bool(contour.get("closed", False)),
                    )
                )

        return (tuple(vector_points), tuple(polylines))

    def _vector_segments(
        self,
        document: VectorDocument,
        transformer: CoordinateTransformer,
    ) -> tuple[tuple[Point, ...], tuple[RasterPolyline, ...]]:
        sampled_points: list[Point] = []
        polylines: list[RasterPolyline] = []

        for segment in document.segments:
            segment_points = self.segment_sampler.sample_segment(segment)
            if len(segment_points) < 2:
                continue
            sampled_points.extend(segment_points)
            polylines.append(
                RasterPolyline(
                    points=np.array([self._point_to_pixel(transformer, point) for point in segment_points], dtype=np.int32),
                    closed=self.segment_sampler.is_closed(segment),
                )
            )

        return (tuple(sampled_points), tuple(polylines))

    def _rasterize_polylines(
        self,
        polylines: tuple[RasterPolyline, ...],
        width: int,
        height: int,
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for polyline in polylines:
            if len(polyline.points) < 2:
                continue
            cv2.polylines(mask, [polyline.points], polyline.closed, 255, self.options.line_thickness, cv2.LINE_8)
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
    "RasterPolyline",
]
