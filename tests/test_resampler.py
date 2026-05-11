import math

from services.resampler import Resampler, ResamplerConfig


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
