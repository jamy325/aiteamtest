import math

from core.precision import EPSILON, PrecisionUtility


def test_global_epsilon_is_exposed() -> None:
    assert EPSILON == PrecisionUtility.EPSILON
    assert EPSILON > 0.0


def test_float_comparison_uses_epsilon() -> None:
    left = 1.0
    right = 1.0 + EPSILON / 2.0

    assert PrecisionUtility.almost_equal(left, right)
    assert PrecisionUtility.compare(left, right) == 0
    assert PrecisionUtility.compare(1.0, 2.0) == -1
    assert PrecisionUtility.compare(2.0, 1.0) == 1


def test_point_distance_and_near_zero_helpers() -> None:
    assert PrecisionUtility.near_zero(EPSILON / 2.0)
    assert PrecisionUtility.distance_between_points((0.0, 0.0), (3.0, 4.0)) == 5.0
    assert PrecisionUtility.points_close((0.0, 0.0), (EPSILON / 2.0, 0.0))


def test_normalize_vector_handles_zero_safely() -> None:
    assert PrecisionUtility.normalize_vector((0.0, 0.0)) is None

    normalized = PrecisionUtility.normalize_vector((3.0, 4.0))
    assert normalized is not None
    assert math.isclose(normalized[0], 0.6)
    assert math.isclose(normalized[1], 0.8)


def test_angle_comparison_wraps_across_full_rotation() -> None:
    assert PrecisionUtility.angle_close(0.0, 2.0 * math.pi)
    assert not PrecisionUtility.angle_close(0.0, math.pi / 2.0)
