from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from core.precision import PrecisionUtility


Point = tuple[float, float]
NoiseThresholdMode = Literal["absolute", "bbox_diagonal", "average_segment_length"]


@dataclass(frozen=True, slots=True)
class ResamplerConfig:
    straight_spacing: float = 4.0
    curve_spacing: float = 1.5
    corner_angle_degrees: float = 35.0
    curvature_threshold: float = 0.08
    noise_distance_threshold: float = 3.0
    duplicate_epsilon: float = 1e-6
    enable_uniform_resampling: bool = False
    target_spacing: float | None = None
    preserve_corners: bool = True
    noise_threshold_mode: NoiseThresholdMode = "absolute"
    noise_scale_ratio: float = 0.08


class Resampler:
    def __init__(self, config: ResamplerConfig | None = None) -> None:
        self.config = config or ResamplerConfig()

    def resample(self, points: tuple[Point, ...] | list[Point], closed: bool = False) -> tuple[Point, ...]:
        normalized = self._normalize_points(points, closed=closed)
        if len(normalized) <= 2:
            return self._resample_simple_path(normalized, closed=closed)

        filtered = self._filter_noise_points(normalized, closed=closed)
        if len(filtered) <= 2:
            return self._resample_simple_path(filtered, closed=closed)

        high_curvature = self._classify_curvature(filtered, closed=closed)
        if self.config.enable_uniform_resampling:
            sampled = self._uniform_resample(filtered, high_curvature, closed=closed)
        else:
            # MVP note: this is adaptive decimation over existing Vector Space points,
            # not a full interpolation-based uniform resampling pass.
            sampled = self._sample_by_spacing(filtered, high_curvature, closed=closed)

        if closed and sampled and sampled[0] != sampled[-1]:
            sampled = sampled + (sampled[0],)
        return sampled

    def _resample_simple_path(self, points: tuple[Point, ...], closed: bool) -> tuple[Point, ...]:
        if len(points) <= 1:
            return points
        if not self.config.enable_uniform_resampling:
            if closed and len(points) == 2 and points[0] != points[-1]:
                return points + (points[0],)
            return points

        sampled = self._resample_polyline(points, self._target_spacing(), include_start=True, include_end=True)
        if closed and sampled and sampled[0] != sampled[-1]:
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
            threshold = self._noise_threshold(points, prev_point, next_point)
            bridge_length = PrecisionUtility.distance_between_points(prev_point, next_point)
            deviation = self._distance_to_segment(point, prev_point, next_point)

            if deviation > threshold and bridge_length <= threshold * 2.0:
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

            turn_angle, curvature = self._turn_metrics(points, index=index, closed=closed)
            if turn_angle is None or curvature is None:
                flags.append(False)
                continue
            flags.append(turn_angle >= angle_threshold or curvature >= self.config.curvature_threshold)

        return tuple(flags)

    def _detect_corners(self, points: tuple[Point, ...], closed: bool) -> tuple[bool, ...]:
        count = len(points)
        if count == 0:
            return ()

        angle_threshold = math.radians(self.config.corner_angle_degrees)
        corners: list[bool] = []
        for index in range(count):
            if not closed and index in (0, count - 1):
                corners.append(True)
                continue

            turn_angle, _ = self._turn_metrics(points, index=index, closed=closed)
            corners.append(turn_angle is not None and turn_angle >= angle_threshold)

        return tuple(corners)

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

    def _uniform_resample(self, points: tuple[Point, ...], high_curvature: tuple[bool, ...], closed: bool) -> tuple[Point, ...]:
        if len(points) <= 1:
            return points

        if not self.config.preserve_corners:
            return self._uniform_resample_without_corners(points, closed=closed)

        corners = self._detect_corners(points, closed=closed)
        if closed:
            if not any(corners):
                return self._uniform_resample_without_corners(points, closed=True)
            return self._uniform_resample_closed_with_corners(points, corners)

        return self._uniform_resample_open_with_corners(points, corners)

    def _uniform_resample_without_corners(self, points: tuple[Point, ...], closed: bool) -> tuple[Point, ...]:
        if closed:
            sampled = self._resample_closed_loop(points, self._target_spacing())
            return sampled + ((sampled[0],) if sampled else ())
        return self._resample_polyline(points, self._target_spacing(), include_start=True, include_end=True)

    def _uniform_resample_open_with_corners(self, points: tuple[Point, ...], corners: tuple[bool, ...]) -> tuple[Point, ...]:
        corner_indices = [index for index, is_corner in enumerate(corners) if is_corner]
        if len(corner_indices) < 2:
            return self._resample_polyline(points, self._target_spacing(), include_start=True, include_end=True)

        sampled: list[Point] = []
        for segment_index, (start_index, end_index) in enumerate(zip(corner_indices, corner_indices[1:])):
            segment_points = points[start_index : end_index + 1]
            segment_sampled = self._resample_polyline(
                segment_points,
                self._target_spacing(),
                include_start=segment_index == 0,
                include_end=True,
            )
            sampled.extend(segment_sampled)

        return tuple(sampled)

    def _uniform_resample_closed_with_corners(self, points: tuple[Point, ...], corners: tuple[bool, ...]) -> tuple[Point, ...]:
        corner_indices = [index for index, is_corner in enumerate(corners) if is_corner]
        if not corner_indices:
            sampled = self._resample_closed_loop(points, self._target_spacing())
            return sampled + ((sampled[0],) if sampled else ())

        sampled: list[Point] = []
        count = len(points)
        for segment_index, start_index in enumerate(corner_indices):
            end_index = corner_indices[(segment_index + 1) % len(corner_indices)]
            segment_points = self._slice_closed_segment(points, start_index, end_index)
            segment_sampled = self._resample_polyline(
                segment_points,
                self._target_spacing(),
                include_start=segment_index == 0,
                include_end=True,
            )
            sampled.extend(segment_sampled)

        sampled_tuple = tuple(sampled)
        if sampled_tuple and not PrecisionUtility.points_close(
            sampled_tuple[0], sampled_tuple[-1], epsilon=self.config.duplicate_epsilon
        ):
            sampled_tuple = sampled_tuple + (sampled_tuple[0],)
        return sampled_tuple

    def _slice_closed_segment(self, points: tuple[Point, ...], start_index: int, end_index: int) -> tuple[Point, ...]:
        count = len(points)
        segment = [points[start_index]]
        index = start_index
        while index != end_index:
            index = (index + 1) % count
            segment.append(points[index])
        return tuple(segment)

    def _resample_closed_loop(self, points: tuple[Point, ...], spacing: float) -> tuple[Point, ...]:
        if len(points) <= 1:
            return points

        wrapped_points = points + (points[0],)
        cumulative_lengths = self._cumulative_lengths(wrapped_points)
        total_length = cumulative_lengths[-1]
        if PrecisionUtility.near_zero(total_length, self.config.duplicate_epsilon):
            return (points[0],)

        distances = [0.0]
        current = spacing
        while current < total_length - self.config.duplicate_epsilon:
            distances.append(current)
            current += spacing

        return tuple(self._interpolate_along_polyline(wrapped_points, cumulative_lengths, distance) for distance in distances)

    def _resample_polyline(
        self,
        points: tuple[Point, ...],
        spacing: float,
        *,
        include_start: bool,
        include_end: bool,
    ) -> tuple[Point, ...]:
        if len(points) <= 1:
            return points if include_start else ()

        cumulative_lengths = self._cumulative_lengths(points)
        total_length = cumulative_lengths[-1]
        if PrecisionUtility.near_zero(total_length, self.config.duplicate_epsilon):
            if include_start and include_end and points[0] != points[-1]:
                return (points[0], points[-1])
            return (points[0],) if include_start or include_end else ()

        sampled: list[Point] = []
        if include_start:
            sampled.append(points[0])

        current = spacing
        while current < total_length - self.config.duplicate_epsilon:
            point = self._interpolate_along_polyline(points, cumulative_lengths, current)
            if not sampled or not PrecisionUtility.points_close(sampled[-1], point, epsilon=self.config.duplicate_epsilon):
                sampled.append(point)
            current += spacing

        if include_end:
            end_point = points[-1]
            if not sampled or not PrecisionUtility.points_close(sampled[-1], end_point, epsilon=self.config.duplicate_epsilon):
                sampled.append(end_point)

        return tuple(sampled)

    def _interpolate_along_polyline(
        self,
        points: tuple[Point, ...],
        cumulative_lengths: tuple[float, ...],
        target_distance: float,
    ) -> Point:
        for index in range(1, len(points)):
            if target_distance <= cumulative_lengths[index] + self.config.duplicate_epsilon:
                start = points[index - 1]
                end = points[index]
                segment_start = cumulative_lengths[index - 1]
                segment_length = cumulative_lengths[index] - segment_start
                if PrecisionUtility.near_zero(segment_length, self.config.duplicate_epsilon):
                    return end

                ratio = max(0.0, min(1.0, (target_distance - segment_start) / segment_length))
                return (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )

        return points[-1]

    def _cumulative_lengths(self, points: tuple[Point, ...]) -> tuple[float, ...]:
        lengths = [0.0]
        total = 0.0
        for index in range(1, len(points)):
            total += PrecisionUtility.distance_between_points(points[index - 1], points[index])
            lengths.append(total)
        return tuple(lengths)

    def _turn_metrics(self, points: tuple[Point, ...], *, index: int, closed: bool) -> tuple[float | None, float | None]:
        count = len(points)
        if not closed and index in (0, count - 1):
            return (None, None)

        prev_index = (index - 1) % count
        next_index = (index + 1) % count
        prev_vector = (points[index][0] - points[prev_index][0], points[index][1] - points[prev_index][1])
        next_vector = (points[next_index][0] - points[index][0], points[next_index][1] - points[index][1])
        prev_length = math.hypot(prev_vector[0], prev_vector[1])
        next_length = math.hypot(next_vector[0], next_vector[1])
        if PrecisionUtility.near_zero(prev_length, self.config.duplicate_epsilon) or PrecisionUtility.near_zero(
            next_length, self.config.duplicate_epsilon
        ):
            return (None, None)

        unit_prev = PrecisionUtility.normalize_vector(prev_vector)
        unit_next = PrecisionUtility.normalize_vector(next_vector)
        if unit_prev is None or unit_next is None:
            return (None, None)

        dot = max(-1.0, min(1.0, unit_prev[0] * unit_next[0] + unit_prev[1] * unit_next[1]))
        turn_angle = math.acos(dot)
        average_length = (prev_length + next_length) / 2.0
        curvature = turn_angle / average_length
        return (turn_angle, curvature)

    def _noise_threshold(self, points: tuple[Point, ...], prev_point: Point, next_point: Point) -> float:
        if self.config.noise_threshold_mode == "absolute":
            return self.config.noise_distance_threshold

        if self.config.noise_threshold_mode == "bbox_diagonal":
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            return max(self.config.duplicate_epsilon, diagonal * self.config.noise_scale_ratio)

        average_segment_length = self._average_segment_length(points)
        return max(self.config.duplicate_epsilon, average_segment_length * self.config.noise_scale_ratio)

    def _average_segment_length(self, points: tuple[Point, ...]) -> float:
        if len(points) <= 1:
            return 0.0
        lengths = [
            PrecisionUtility.distance_between_points(points[index - 1], points[index])
            for index in range(1, len(points))
        ]
        if not lengths:
            return 0.0
        return sum(lengths) / len(lengths)

    def _target_spacing(self) -> float:
        spacing = self.config.target_spacing if self.config.target_spacing is not None else self.config.straight_spacing
        return max(self.config.duplicate_epsilon, float(spacing))

    def _distance_to_segment(self, point: Point, start: Point, end: Point) -> float:
        segment = (end[0] - start[0], end[1] - start[1])
        segment_length_sq = segment[0] ** 2 + segment[1] ** 2
        if PrecisionUtility.near_zero(segment_length_sq, self.config.duplicate_epsilon):
            return PrecisionUtility.distance_between_points(point, start)

        projection = ((point[0] - start[0]) * segment[0] + (point[1] - start[1]) * segment[1]) / segment_length_sq
        projection = max(0.0, min(1.0, projection))
        closest = (start[0] + projection * segment[0], start[1] + projection * segment[1])
        return PrecisionUtility.distance_between_points(point, closest)


__all__ = ["NoiseThresholdMode", "Point", "Resampler", "ResamplerConfig"]
