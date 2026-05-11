from __future__ import annotations

import math
from typing import Final

EPSILON: Final[float] = 1e-9
Vector2 = tuple[float, float]


class PrecisionUtility:
    EPSILON: Final[float] = EPSILON

    @staticmethod
    def near_zero(value: float, epsilon: float | None = None) -> bool:
        threshold = PrecisionUtility.EPSILON if epsilon is None else abs(epsilon)
        return abs(value) <= threshold

    @staticmethod
    def almost_equal(left: float, right: float, epsilon: float | None = None) -> bool:
        threshold = PrecisionUtility.EPSILON if epsilon is None else abs(epsilon)
        return abs(left - right) <= threshold

    @staticmethod
    def compare(left: float, right: float, epsilon: float | None = None) -> int:
        if PrecisionUtility.almost_equal(left, right, epsilon=epsilon):
            return 0
        return -1 if left < right else 1

    @staticmethod
    def distance_between_points(first: Vector2, second: Vector2) -> float:
        return math.dist(first, second)

    @staticmethod
    def points_close(first: Vector2, second: Vector2, epsilon: float | None = None) -> bool:
        threshold = PrecisionUtility.EPSILON if epsilon is None else abs(epsilon)
        return PrecisionUtility.distance_between_points(first, second) <= threshold

    @staticmethod
    def normalize_vector(vector: Vector2, epsilon: float | None = None) -> Vector2 | None:
        threshold = PrecisionUtility.EPSILON if epsilon is None else abs(epsilon)
        length = math.hypot(vector[0], vector[1])
        if length <= threshold:
            return None
        return (vector[0] / length, vector[1] / length)

    @staticmethod
    def angle_close(left_radians: float, right_radians: float, epsilon: float | None = None) -> bool:
        threshold = PrecisionUtility.EPSILON if epsilon is None else abs(epsilon)
        delta = (left_radians - right_radians + math.pi) % (2.0 * math.pi) - math.pi
        return abs(delta) <= threshold


__all__ = ["EPSILON", "PrecisionUtility", "Vector2"]
