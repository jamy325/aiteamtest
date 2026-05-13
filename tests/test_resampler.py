import math

import pytest

from services.resampler import Resampler, ResamplerConfig


def _distances(points: tuple[tuple[float, float], ...]) -> list[float]:
    return [
        math.dist(points[index - 1], points[index])
        for index in range(1, len(points))
    ]


def test_resampler_reduces_straight_line_points_in_vector_space() -> None:
    points = [(float(x), 0.0) for x in range(40)]
    resampler = Resampler(ResamplerConfig(straight_spacing=5.0, curve_spacing=1.5))

    sampled = resampler.resample(points, closed=False)

    assert len(sampled) < len(points)
    assert sampled[0] == (0.0, 0.0)
    assert sampled[-1] == (39.0, 0.0)
    assert all(isinstance(point[0], float) and isinstance(point[1], float) for point in sampled)


def test_resampler_keeps_more_points_for_curved_shapes() -> None:
    line_points = [(float(x), 0.0) for x in range(40)]
    arc_points = [
        (10.0 * math.cos(theta), 10.0 * math.sin(theta))
        for theta in [math.pi * step / 39.0 for step in range(40)]
    ]
    resampler = Resampler(ResamplerConfig(straight_spacing=5.0, curve_spacing=1.0, curvature_threshold=0.05))

    sampled_line = resampler.resample(line_points, closed=False)
    sampled_arc = resampler.resample(arc_points, closed=False)

    assert len(sampled_arc) > len(sampled_line)
    assert sampled_arc[0] == arc_points[0]
    assert sampled_arc[-1] == arc_points[-1]


def test_resampler_preserves_corner_features() -> None:
    points = [
        (0.0, 0.0),
        (2.0, 0.0),
        (4.0, 0.0),
        (6.0, 0.0),
        (8.0, 0.0),
        (10.0, 0.0),
        (10.0, 2.0),
        (10.0, 4.0),
        (10.0, 6.0),
        (10.0, 8.0),
        (10.0, 10.0),
    ]
    resampler = Resampler(ResamplerConfig(straight_spacing=6.0, curve_spacing=1.0, corner_angle_degrees=20.0))

    sampled = resampler.resample(points, closed=False)

    assert any(point == (10.0, 0.0) for point in sampled)
    assert sampled[0] == (0.0, 0.0)
    assert sampled[-1] == (10.0, 10.0)


def test_resampler_preserves_closed_contour_start_end_relationship() -> None:
    square = [
        (0.0, 0.0),
        (4.0, 0.0),
        (8.0, 0.0),
        (8.0, 4.0),
        (8.0, 8.0),
        (4.0, 8.0),
        (0.0, 8.0),
        (0.0, 4.0),
        (0.0, 0.0),
    ]
    resampler = Resampler(ResamplerConfig(straight_spacing=4.0, curve_spacing=1.0))

    sampled = resampler.resample(square, closed=True)

    assert sampled[0] == sampled[-1]
    assert len(sampled) >= 5


def test_resampler_filters_single_spike_noise_point() -> None:
    points = [(float(x), 0.0) for x in range(10)] + [(10.0, 10.0)] + [(float(x), 0.0) for x in range(11, 21)]
    resampler = Resampler(ResamplerConfig(straight_spacing=5.0, curve_spacing=1.0, noise_distance_threshold=3.0))

    sampled = resampler.resample(points, closed=False)

    assert (10.0, 10.0) not in sampled
    assert sampled[0] == (0.0, 0.0)
    assert sampled[-1] == (20.0, 0.0)


def test_resampler_removes_duplicate_and_near_duplicate_points() -> None:
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (1.0 + 1e-7, 0.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (4.0, 0.0),
    ]
    resampler = Resampler(ResamplerConfig(straight_spacing=2.0, curve_spacing=1.0, duplicate_epsilon=1e-6))

    sampled = resampler.resample(points, closed=False)

    assert sampled[0] == (0.0, 0.0)
    assert sampled[-1] == (4.0, 0.0)
    assert sampled.count((1.0, 0.0)) <= 1


def test_uniform_resampler_inserts_evenly_spaced_points_on_open_line() -> None:
    points = [(0.0, 0.0), (100.0, 0.0)]
    resampler = Resampler(
        ResamplerConfig(
            enable_uniform_resampling=True,
            target_spacing=10.0,
        )
    )

    sampled = resampler.resample(points, closed=False)

    expected = tuple((float(step), 0.0) for step in range(0, 101, 10))
    assert sampled == expected
    assert all(distance == pytest.approx(10.0) for distance in _distances(sampled))


def test_uniform_resampler_handles_non_divisible_open_line_and_keeps_endpoints() -> None:
    points = [(0.0, 0.0), (95.0, 0.0)]
    resampler = Resampler(
        ResamplerConfig(
            enable_uniform_resampling=True,
            target_spacing=10.0,
        )
    )

    sampled = resampler.resample(points, closed=False)
    distances = _distances(sampled)

    assert sampled[0] == (0.0, 0.0)
    assert sampled[-1] == (95.0, 0.0)
    assert all(distance == pytest.approx(10.0) for distance in distances[:-1])
    assert distances[-1] <= 10.0


def test_uniform_resampler_preserves_closed_square_corners_and_spacing() -> None:
    square = (
        (0.0, 0.0),
        (10.0, 0.0),
        (20.0, 0.0),
        (20.0, 10.0),
        (20.0, 20.0),
        (10.0, 20.0),
        (0.0, 20.0),
        (0.0, 10.0),
        (0.0, 0.0),
    )
    resampler = Resampler(
        ResamplerConfig(
            enable_uniform_resampling=True,
            target_spacing=5.0,
            preserve_corners=True,
            corner_angle_degrees=30.0,
        )
    )

    sampled = resampler.resample(square, closed=True)
    distances = _distances(sampled[:-1])

    assert sampled[0] == sampled[-1]
    for corner in ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)):
        assert corner in sampled
    assert all(distance == pytest.approx(5.0) for distance in distances)


def test_uniform_resampler_keeps_corner_on_open_l_shape() -> None:
    points = (
        (0.0, 0.0),
        (10.0, 0.0),
        (20.0, 0.0),
        (20.0, 10.0),
        (20.0, 20.0),
    )
    resampler = Resampler(
        ResamplerConfig(
            enable_uniform_resampling=True,
            target_spacing=5.0,
            preserve_corners=True,
            corner_angle_degrees=30.0,
        )
    )

    sampled = resampler.resample(points, closed=False)

    assert (20.0, 0.0) in sampled
    assert sampled[0] == (0.0, 0.0)
    assert sampled[-1] == (20.0, 20.0)


def test_uniform_resampler_distributes_circle_points_by_arclength() -> None:
    circle_points = tuple(
        (
            10.0 * math.cos(2.0 * math.pi * step / 8.0),
            10.0 * math.sin(2.0 * math.pi * step / 8.0),
        )
        for step in range(8)
    ) + ((10.0, 0.0),)
    resampler = Resampler(
        ResamplerConfig(
            enable_uniform_resampling=True,
            target_spacing=4.0,
            preserve_corners=False,
        )
    )

    sampled = resampler.resample(circle_points, closed=True)
    distances = _distances(sampled[:-1])

    assert sampled[0] == sampled[-1]
    assert len(sampled) > len(circle_points)
    assert max(distances) - min(distances) < 1.5


def test_scale_aware_noise_filtering_removes_relative_spikes_at_small_and_large_scales() -> None:
    small_points = tuple((float(x), 0.0) for x in range(10)) + ((10.0, 2.0),) + tuple((float(x), 0.0) for x in range(11, 21))
    large_points = tuple((float(10 * x), 0.0) for x in range(10)) + ((100.0, 20.0),) + tuple(
        (float(10 * x), 0.0) for x in range(11, 21)
    )
    config = ResamplerConfig(
        straight_spacing=100.0,
        curve_spacing=10.0,
        noise_threshold_mode="bbox_diagonal",
        noise_scale_ratio=0.05,
    )
    resampler = Resampler(config)

    sampled_small = resampler.resample(small_points, closed=False)
    sampled_large = resampler.resample(large_points, closed=False)

    assert (10.0, 2.0) not in sampled_small
    assert (100.0, 20.0) not in sampled_large


def test_scale_aware_noise_filtering_keeps_large_scale_real_corner() -> None:
    points = (
        (0.0, 0.0),
        (40.0, 0.0),
        (80.0, 0.0),
        (80.0, 40.0),
        (80.0, 80.0),
    )
    resampler = Resampler(
        ResamplerConfig(
            noise_threshold_mode="bbox_diagonal",
            noise_scale_ratio=0.05,
            straight_spacing=40.0,
            curve_spacing=5.0,
            corner_angle_degrees=25.0,
        )
    )

    sampled = resampler.resample(points, closed=False)

    assert (80.0, 0.0) in sampled
