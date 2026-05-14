from __future__ import annotations

from dataclasses import dataclass
import math

from core.precision import PrecisionUtility
from core.types import Point, Segment


@dataclass(frozen=True, slots=True)
class SegmentSamplerConfig:
    max_chord_error: float = 0.25
    min_segments_per_arc: int = 8
    max_segments_per_arc: int = 128
    circle_segments: int = 64
    ellipse_segments: int = 64
    bezier_segments: int = 24
    line_sample_step: float | None = None
    epsilon: float = 1e-9


class SegmentSampler:
    def __init__(self, config: SegmentSamplerConfig | None = None) -> None:
        self.config = config or SegmentSamplerConfig()

    def sample_segment(self, segment: Segment) -> tuple[Point, ...]:
        if segment.type == "line":
            return self._sample_line(segment)
        if segment.type == "bezier":
            return self._sample_bezier(segment)
        if segment.type == "arc":
            return self._sample_arc(segment)
        if segment.type == "circle":
            return self._sample_circle(segment)
        if segment.type == "ellipse":
            return self._sample_ellipse(segment)
        if segment.type == "polyline":
            if "points" in segment.params:
                return tuple(self._coerce_point(point) for point in segment.params["points"])
            return (
                self._coerce_point(segment.params["start"]),
                self._coerce_point(segment.params["end"]),
            )
        raise ValueError(f"unsupported segment type for sampling: {segment.type}")

    def is_closed(self, segment: Segment) -> bool:
        return segment.type in {"circle", "ellipse"}

    def _sample_bezier(self, segment: Segment) -> tuple[Point, ...]:
        start = self._coerce_point(segment.params["start"])
        control1 = self._coerce_point(segment.params["control1"])
        control2 = self._coerce_point(segment.params["control2"])
        end = self._coerce_point(segment.params["end"])
        segment_count = max(1, int(self.config.bezier_segments))
        return tuple(
            self._cubic_bezier_point(start, control1, control2, end, step / segment_count)
            for step in range(segment_count + 1)
        )

    def _sample_line(self, segment: Segment) -> tuple[Point, ...]:
        start = self._coerce_point(segment.params["start"])
        end = self._coerce_point(segment.params["end"])
        if self.config.line_sample_step is None:
            return (start, end)
        length = math.dist(start, end)
        step = max(float(self.config.line_sample_step), self.config.epsilon)
        segment_count = max(1, int(math.ceil(length / step)))
        return tuple(
            (
                start[0] + (end[0] - start[0]) * (index / segment_count),
                start[1] + (end[1] - start[1]) * (index / segment_count),
            )
            for index in range(segment_count + 1)
        )

    def _sample_arc(self, segment: Segment) -> tuple[Point, ...]:
        center = (float(segment.params["cx"]), float(segment.params["cy"]))
        radius = abs(float(segment.params["r"]))
        if PrecisionUtility.near_zero(radius, epsilon=self.config.epsilon):
            raise ValueError("arc radius must be positive for sampling")

        start_angle = float(segment.params["start_angle"])
        end_angle = float(segment.params["end_angle"])
        direction = str(segment.params.get("direction", "ccw")).lower()
        signed_sweep = self._signed_arc_sweep(start_angle, end_angle, direction)
        sweep = abs(signed_sweep)

        min_segments = max(1, int(self.config.min_segments_per_arc))
        max_segments = max(min_segments, int(self.config.max_segments_per_arc))
        chord_error = max(float(self.config.max_chord_error), self.config.epsilon)
        if chord_error >= radius:
            segment_count = max_segments
        else:
            max_angle = 2.0 * math.acos(max(-1.0, min(1.0, 1.0 - (chord_error / radius))))
            if PrecisionUtility.near_zero(max_angle, epsilon=self.config.epsilon):
                segment_count = max_segments
            else:
                segment_count = int(math.ceil(sweep / max_angle))
        segment_count = min(max(max(segment_count, min_segments), 1), max_segments)

        return tuple(
            (
                center[0] + radius * math.cos(start_angle + signed_sweep * (step / segment_count)),
                center[1] + radius * math.sin(start_angle + signed_sweep * (step / segment_count)),
            )
            for step in range(segment_count + 1)
        )

    def _sample_circle(self, segment: Segment) -> tuple[Point, ...]:
        center = (float(segment.params["cx"]), float(segment.params["cy"]))
        radius = abs(float(segment.params["r"]))
        if PrecisionUtility.near_zero(radius, epsilon=self.config.epsilon):
            raise ValueError("circle radius must be positive for sampling")
        segment_count = max(8, int(self.config.circle_segments))
        return tuple(
            (
                center[0] + radius * math.cos((math.tau * step) / segment_count),
                center[1] + radius * math.sin((math.tau * step) / segment_count),
            )
            for step in range(segment_count + 1)
        )

    def _sample_ellipse(self, segment: Segment) -> tuple[Point, ...]:
        cx = float(segment.params["cx"])
        cy = float(segment.params["cy"])
        rx = abs(float(segment.params["rx"]))
        ry = abs(float(segment.params["ry"]))
        rotation = float(segment.params.get("rotation", 0.0))
        if PrecisionUtility.near_zero(rx, epsilon=self.config.epsilon) or PrecisionUtility.near_zero(
            ry,
            epsilon=self.config.epsilon,
        ):
            raise ValueError("ellipse axes must be positive for sampling")
        segment_count = max(8, int(self.config.ellipse_segments))
        cos_theta = math.cos(rotation)
        sin_theta = math.sin(rotation)
        return tuple(
            (
                cx + (rx * math.cos((math.tau * step) / segment_count) * cos_theta)
                - (ry * math.sin((math.tau * step) / segment_count) * sin_theta),
                cy + (rx * math.cos((math.tau * step) / segment_count) * sin_theta)
                + (ry * math.sin((math.tau * step) / segment_count) * cos_theta),
            )
            for step in range(segment_count + 1)
        )

    def _signed_arc_sweep(self, start_angle: float, end_angle: float, direction: str) -> float:
        if PrecisionUtility.almost_equal(start_angle, end_angle, epsilon=self.config.epsilon):
            return -math.tau if direction == "cw" else math.tau

        if direction == "cw":
            sweep = end_angle - start_angle
            if sweep >= 0.0:
                sweep -= math.tau
            return sweep

        sweep = end_angle - start_angle
        if sweep <= 0.0:
            sweep += math.tau
        return sweep

    def _cubic_bezier_point(
        self,
        start: Point,
        control1: Point,
        control2: Point,
        end: Point,
        t: float,
    ) -> Point:
        one_minus_t = 1.0 - t
        x = (
            (one_minus_t ** 3) * start[0]
            + 3.0 * (one_minus_t ** 2) * t * control1[0]
            + 3.0 * one_minus_t * (t ** 2) * control2[0]
            + (t ** 3) * end[0]
        )
        y = (
            (one_minus_t ** 3) * start[1]
            + 3.0 * (one_minus_t ** 2) * t * control1[1]
            + 3.0 * one_minus_t * (t ** 2) * control2[1]
            + (t ** 3) * end[1]
        )
        return (x, y)

    def _coerce_point(self, value: Point | list[float]) -> Point:
        return (float(value[0]), float(value[1]))


__all__ = ["SegmentSampler", "SegmentSamplerConfig"]
