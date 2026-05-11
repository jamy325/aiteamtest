from __future__ import annotations

from dataclasses import dataclass
import math

from core.precision import PrecisionUtility


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class ResamplerConfig:
    straight_spacing: float = 4.0
    curve_spacing: float = 1.5
    corner_angle_degrees: float = 35.0
    curvature_threshold: float = 0.08
    noise_distance_threshold: float = 3.0
    duplicate_epsilon: float = 1e-6


class Resampler:
    def __init__(self, config: ResamplerConfig | None = None) -> None:
        self.config = config or ResamplerConfig()

    def resample(self, points: tuple[Point, ...] | list[Point], closed: bool = False) -> tuple[Point, ...]:
        normalized = self._normalize_points(points, closed=closed)
        if len(normalized) <= 2:
            if closed and len(normalized) == 2 and normalized[0] != normalized[-1]:
                return normalized + (normalized[0],)
            return normalized

        filtered = self._filter_noise_points(normalized, closed=closed)
        if len(filtered) <= 2:
            if closed and len(filtered) == 2 and filtered[0] != filtered[-1]:
                return filtered + (filtered[0],)
            return filtered

        high_curvature = self._classify_curvature(filtered, closed=closed)
        # MVP note: this is adaptive decimation over existing Vector Space points,
        # not a full interpolation-based uniform resampling pass.
        sampled = self._sample_by_spacing(filtered, high_curvature, closed=closed)

        if closed:
            if sampled[0] != sampled[-1]:
                sampled = sampled + (sampled[0],)
        return sampled

    def _normalize_points(self, points: tuple[Point, ...] | list[Point], closed: bool) -> tuple[Point, ...]:
        normalized: list[Point] = []
        for point in points:
            candidate = (float(point[0]), float(point[1]))
            if normalized and PrecisionUtility.points_close(normalized[-1], candidate, epsilon=self.config.duplicate_epsilon):
                continue
            normalized.append(candidate)

        if closed and len(normalized) > 1 and PrecisionUtility.points_close(
            normalized[0], normalized[-1], epsilon=self.config.duplicate_epsilon
        ):
            normalized.pop()

        return tuple(normalized)

    def _filter_noise_points(self, points: tuple[Point, ...], closed: bool) -> tuple[Point, ...]:
        if len(points) <= 2:
            return points

        filtered: list[Point] = []
        count = len(points)

        for index, point in enumerate(points):
            if not closed and index in (0, count - 1):
                filtered.append(point)
                continue

            prev_point = points[(index - 1) % count]
            next_point = points[(index + 1) % count]
            bridge_length = PrecisionUtility.distance_between_points(prev_point, next_point)
            deviation = self._distance_to_segment(point, prev_point, next_point)

            if deviation > self.config.noise_distance_threshold and bridge_length <= self.config.noise_distance_threshold * 2.0:
                continue

            filtered.append(point)

        return tuple(filtered)

    def _classify_curvature(self, points: tuple[Point, ...], closed: bool) -> tuple[bool, ...]:
        flags: list[bool] = []
        count = len(points)
        angle_threshold = math.radians(self.config.corner_angle_degrees)

        for index in range(count):
            if not closed and index in (0, count - 1):
                flags.append(True)
                continue

            prev_index = (index - 1) % count
            next_index = (index + 1) % count
            prev_vector = (points[index][0] - points[prev_index][0], points[index][1] - points[prev_index][1])
            next_vector = (points[next_index][0] - points[index][0], points[next_index][1] - points[index][1])
            prev_length = math.hypot(prev_vector[0], prev_vector[1])
            next_length = math.hypot(next_vector[0], next_vector[1])
            if PrecisionUtility.near_zero(prev_length, self.config.duplicate_epsilon) or PrecisionUtility.near_zero(
                next_length, self.config.duplicate_epsilon
            ):
                flags.append(False)
                continue

            unit_prev = PrecisionUtility.normalize_vector(prev_vector)
            unit_next = PrecisionUtility.normalize_vector(next_vector)
            if unit_prev is None or unit_next is None:
                flags.append(False)
                continue

            dot = max(-1.0, min(1.0, unit_prev[0] * unit_next[0] + unit_prev[1] * unit_next[1]))
            turn_angle = math.acos(dot)
            average_length = (prev_length + next_length) / 2.0
            curvature = turn_angle / average_length
            flags.append(turn_angle >= angle_threshold or curvature >= self.config.curvature_threshold)

        return tuple(flags)

    def _sample_by_spacing(self, points: tuple[Point, ...], high_curvature: tuple[bool, ...], closed: bool) -> tuple[Point, ...]:
        sampled: list[Point] = [points[0]]
        distance_since_keep = 0.0

        for index in range(1, len(points)):
            segment_length = PrecisionUtility.distance_between_points(points[index - 1], points[index])
            distance_since_keep += segment_length
            spacing = self.config.curve_spacing if high_curvature[index] else self.config.straight_spacing

            if high_curvature[index] or distance_since_keep >= spacing:
                if not PrecisionUtility.points_close(sampled[-1], points[index], epsilon=self.config.duplicate_epsilon):
                    sampled.append(points[index])
                distance_since_keep = 0.0

        if not closed and not PrecisionUtility.points_close(sampled[-1], points[-1], epsilon=self.config.duplicate_epsilon):
            sampled.append(points[-1])

        if closed and not PrecisionUtility.points_close(sampled[-1], points[-1], epsilon=self.config.duplicate_epsilon):
            sampled.append(points[-1])

        return tuple(sampled)

    def _distance_to_segment(self, point: Point, start: Point, end: Point) -> float:
        segment = (end[0] - start[0], end[1] - start[1])
        segment_length_sq = segment[0] ** 2 + segment[1] ** 2
        if PrecisionUtility.near_zero(segment_length_sq, self.config.duplicate_epsilon):
            return PrecisionUtility.distance_between_points(point, start)

        projection = ((point[0] - start[0]) * segment[0] + (point[1] - start[1]) * segment[1]) / segment_length_sq
        projection = max(0.0, min(1.0, projection))
        closest = (start[0] + projection * segment[0], start[1] + projection * segment[1])
        return PrecisionUtility.distance_between_points(point, closest)


__all__ = ["Point", "Resampler", "ResamplerConfig"]
