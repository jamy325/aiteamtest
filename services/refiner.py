from __future__ import annotations

import math
import random
from dataclasses import dataclass

from core.precision import PrecisionUtility
from core.types import Point


@dataclass(frozen=True, slots=True)
class RansacLineConfig:
    iterations: int = 64
    inlier_threshold: float = 0.2
    min_inlier_ratio: float = 0.5
    random_seed: int = 0


@dataclass(frozen=True, slots=True)
class RansacLineResult:
    params: dict[str, object]
    inlier_ratio: float
    fit_error: float
    inlier_indexes: tuple[int, ...]
    outlier_indexes: tuple[int, ...]


class RansacLineFitter:
    def __init__(self, config: RansacLineConfig | None = None) -> None:
        self.config = config or RansacLineConfig()

    def fit(self, points: tuple[Point, ...] | list[Point]) -> RansacLineResult:
        point_sequence = tuple((float(x), float(y)) for x, y in points)
        if len(point_sequence) < 2:
            raise ValueError("at least two Vector Space points are required")

        rng = random.Random(self.config.random_seed)
        best: tuple[tuple[int, ...], tuple[float, float, float], float] | None = None

        for first_index, second_index in self._sample_pairs(len(point_sequence), rng):
            line = self._line_from_points(point_sequence[first_index], point_sequence[second_index])
            if line is None:
                continue

            inlier_indexes = tuple(
                index
                for index, point in enumerate(point_sequence)
                if self._point_line_distance(point, line) <= self.config.inlier_threshold
            )
            if len(inlier_indexes) < 2:
                continue

            fit_error = self._fit_error(point_sequence, inlier_indexes, line)
            candidate = (inlier_indexes, line, fit_error)
            if best is None or self._is_better_candidate(candidate, best, len(point_sequence)):
                best = candidate

        if best is None:
            raise ValueError("unable to fit a robust line from the provided points")

        inlier_indexes, line, fit_error = best
        inlier_ratio = len(inlier_indexes) / len(point_sequence)
        if inlier_ratio < self.config.min_inlier_ratio:
            raise ValueError("insufficient inlier ratio for robust line fit")

        params = self._segment_params(point_sequence, inlier_indexes, line)
        outlier_indexes = tuple(index for index in range(len(point_sequence)) if index not in set(inlier_indexes))

        return RansacLineResult(
            params=params,
            inlier_ratio=inlier_ratio,
            fit_error=fit_error,
            inlier_indexes=inlier_indexes,
            outlier_indexes=outlier_indexes,
        )

    def _sample_pairs(self, point_count: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
        if point_count == 2:
            return ((0, 1),)

        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        max_unique_pairs = point_count * (point_count - 1) // 2
        target_iterations = min(self.config.iterations, max_unique_pairs)

        while len(pairs) < target_iterations:
            first_index, second_index = sorted(rng.sample(range(point_count), 2))
            pair = (first_index, second_index)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)

        return tuple(pairs)

    def _line_from_points(self, first: Point, second: Point) -> tuple[float, float, float] | None:
        if PrecisionUtility.points_close(first, second):
            return None
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = math.hypot(dx, dy)
        if PrecisionUtility.near_zero(length):
            return None
        a = dy / length
        b = -dx / length
        c = -((a * first[0]) + (b * first[1]))
        return (a, b, c)

    def _point_line_distance(self, point: Point, line: tuple[float, float, float]) -> float:
        a, b, c = line
        return abs((a * point[0]) + (b * point[1]) + c)

    def _fit_error(
        self,
        points: tuple[Point, ...],
        inlier_indexes: tuple[int, ...],
        line: tuple[float, float, float],
    ) -> float:
        distances = [self._point_line_distance(points[index], line) for index in inlier_indexes]
        if not distances:
            return math.inf
        return sum(distances) / len(distances)

    def _is_better_candidate(
        self,
        candidate: tuple[tuple[int, ...], tuple[float, float, float], float],
        current_best: tuple[tuple[int, ...], tuple[float, float, float], float],
        point_count: int,
    ) -> bool:
        candidate_inliers, _, candidate_error = candidate
        best_inliers, _, best_error = current_best
        if len(candidate_inliers) != len(best_inliers):
            return len(candidate_inliers) > len(best_inliers)
        if not PrecisionUtility.almost_equal(candidate_error, best_error):
            return candidate_error < best_error
        return (len(candidate_inliers) / point_count) > (len(best_inliers) / point_count)

    def _segment_params(
        self,
        points: tuple[Point, ...],
        inlier_indexes: tuple[int, ...],
        line: tuple[float, float, float],
    ) -> dict[str, object]:
        inlier_points = tuple(points[index] for index in inlier_indexes)
        direction = self._principal_direction(inlier_points)
        origin = inlier_points[0]
        projected = tuple(
            ((point[0] - origin[0]) * direction[0]) + ((point[1] - origin[1]) * direction[1])
            for point in inlier_points
        )
        start_offset = min(projected)
        end_offset = max(projected)
        start = (origin[0] + (direction[0] * start_offset), origin[1] + (direction[1] * start_offset))
        end = (origin[0] + (direction[0] * end_offset), origin[1] + (direction[1] * end_offset))
        a, b, c = line

        return {
            "start": [start[0], start[1]],
            "end": [end[0], end[1]],
            "direction": [direction[0], direction[1]],
            "line": {"a": a, "b": b, "c": c},
        }

    def _principal_direction(self, points: tuple[Point, ...]) -> tuple[float, float]:
        if len(points) == 2:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            direction = PrecisionUtility.normalize_vector((dx, dy))
            if direction is None:
                raise ValueError("degenerate inlier set")
            return direction

        mean_x = sum(point[0] for point in points) / len(points)
        mean_y = sum(point[1] for point in points) / len(points)
        sxx = sum((point[0] - mean_x) ** 2 for point in points)
        syy = sum((point[1] - mean_y) ** 2 for point in points)
        sxy = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points)
        angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
        direction = PrecisionUtility.normalize_vector((math.cos(angle), math.sin(angle)))
        if direction is None:
            raise ValueError("degenerate inlier set")
        return direction


__all__ = [
    "RansacLineConfig",
    "RansacLineFitter",
    "RansacLineResult",
]
