from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from core.precision import PrecisionUtility
from core.types import Point


@dataclass(frozen=True, slots=True)
class EdgeErrorResult:
    missing_edge_error: float
    overdraw_error: float
    chamfer_error: float
    source_point_count: int
    vector_point_count: int


class EdgeErrorCalculator:
    """Approximate edge error from sampled Vector Space points.

    This intentionally stays at the sampled-point level for MVP scoring.
    It does not build a distance field or require any raster-space rendering.
    """

    def calculate(
        self,
        source_edge_points: Sequence[Point] | Sequence[Sequence[float]],
        vector_edge_points: Sequence[Point] | Sequence[Sequence[float]],
    ) -> EdgeErrorResult:
        source_points = self._coerce_points(source_edge_points)
        vector_points = self._coerce_points(vector_edge_points)

        missing_edge_error = self._mean_nearest_distance(source_points, vector_points)
        overdraw_error = self._mean_nearest_distance(vector_points, source_points)
        chamfer_error = self._combine_directional_errors(missing_edge_error, overdraw_error)

        return EdgeErrorResult(
            missing_edge_error=missing_edge_error,
            overdraw_error=overdraw_error,
            chamfer_error=chamfer_error,
            source_point_count=len(source_points),
            vector_point_count=len(vector_points),
        )

    def _coerce_points(
        self,
        points: Sequence[Point] | Sequence[Sequence[float]],
    ) -> tuple[Point, ...]:
        return tuple(self._coerce_point(point) for point in points)

    def _coerce_point(self, point: Point | Sequence[float]) -> Point:
        x, y = point
        return (float(x), float(y))

    def _mean_nearest_distance(
        self,
        points: tuple[Point, ...],
        reference_points: tuple[Point, ...],
    ) -> float:
        if not points:
            return 0.0
        if not reference_points:
            return math.inf

        total_distance = 0.0
        for point in points:
            total_distance += self._nearest_distance(point, reference_points)
        return total_distance / len(points)

    def _nearest_distance(
        self,
        point: Point,
        reference_points: Iterable[Point],
    ) -> float:
        nearest_distance = math.inf
        for reference in reference_points:
            distance = PrecisionUtility.distance_between_points(point, reference)
            if distance < nearest_distance:
                nearest_distance = distance
        return nearest_distance

    def _combine_directional_errors(self, missing_edge_error: float, overdraw_error: float) -> float:
        if math.isinf(missing_edge_error) or math.isinf(overdraw_error):
            return math.inf
        return missing_edge_error + overdraw_error


__all__ = ["EdgeErrorCalculator", "EdgeErrorResult"]
