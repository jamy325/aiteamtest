from __future__ import annotations

from dataclasses import dataclass
from typing import overload

from core.precision import PrecisionUtility
from core.types import CoordinateSystem, Point


@dataclass(frozen=True, slots=True)
class CoordinateTransformer:
    coordinate_system: CoordinateSystem

    def pixel_to_vector(self, point: Point) -> Point:
        x = float(point[0])
        y = float(point[1])
        if self.coordinate_system.unit == "mm":
            x = self.px_to_mm(x)
            y = self.px_to_mm(y)
        if self.coordinate_system.y_axis == "up":
            x, y = self.y_axis_flip((x, y))
        return (x, y)

    def vector_to_pixel(self, point: Point) -> Point:
        x = float(point[0])
        y = float(point[1])
        if self.coordinate_system.y_axis == "up":
            x, y = self.y_axis_flip((x, y))
        if self.coordinate_system.unit == "mm":
            x = self.mm_to_px(x)
            y = self.mm_to_px(y)
        return (x, y)

    def vector_to_svg(self, point: Point) -> Point:
        x = float(point[0])
        y = float(point[1])
        if self.coordinate_system.y_axis == "up":
            x, y = self.y_axis_flip((x, y))
        return self.precision_rounding((x, y))

    def vector_to_dxf(self, point: Point) -> Point:
        x = float(point[0])
        y = float(point[1])
        if self.coordinate_system.y_axis != "up":
            x, y = self.y_axis_flip((x, y))
        if self.coordinate_system.unit == "px":
            x = self.px_to_mm(x)
            y = self.px_to_mm(y)
        return self.precision_rounding((x, y))

    def px_to_mm(self, value: float) -> float:
        return float(value) * self._px_to_mm_scale()

    def mm_to_px(self, value: float) -> float:
        scale = self._px_to_mm_scale()
        if PrecisionUtility.near_zero(scale):
            raise ValueError("px_to_mm scale must be non-zero")
        return float(value) / scale

    def y_axis_flip(self, point: Point, span: float | None = None) -> Point:
        y_span = self._vertical_span() if span is None else float(span)
        return (float(point[0]), y_span - float(point[1]))

    @overload
    def precision_rounding(self, value: float, precision: int | None = None) -> float: ...

    @overload
    def precision_rounding(self, value: Point, precision: int | None = None) -> Point: ...

    def precision_rounding(self, value: float | Point, precision: int | None = None) -> float | Point:
        digits = self.coordinate_system.precision if precision is None else precision
        if isinstance(value, tuple):
            return (round(float(value[0]), digits), round(float(value[1]), digits))
        return round(float(value), digits)

    def _px_to_mm_scale(self) -> float:
        return float(self.coordinate_system.scale.get("px_to_mm", 1.0))

    def _vertical_span(self) -> float:
        if self.coordinate_system.view_box is not None:
            return float(self.coordinate_system.view_box[3])
        raise ValueError("view_box is required for Y-axis flipping")


__all__ = ["CoordinateTransformer"]
