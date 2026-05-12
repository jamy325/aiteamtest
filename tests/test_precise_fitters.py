import ast
import math
from pathlib import Path

import pytest

from services.refiner import (
    PreciseArcFitter,
    PreciseCircleFitter,
    PreciseLineFitter,
    RansacArcConfig,
    RansacArcFitter,
    RansacCircleConfig,
    RansacCircleFitter,
    RansacLineConfig,
    RansacLineFitter,
)


def test_precise_line_fitter_refits_ransac_inliers() -> None:
    points = tuple(
        (
            float(index),
            (2.0 * float(index)) + 1.0 + (0.04 if index % 2 == 0 else -0.03),
        )
        for index in range(10)
    ) + ((2.0, 30.0), (11.0, -5.0))
    ransac = RansacLineFitter(
        RansacLineConfig(
            iterations=80,
            inlier_threshold=0.15,
            min_inlier_ratio=0.7,
            random_seed=3,
        )
    ).fit(points)
    inlier_points = tuple(points[index] for index in ransac.inlier_indexes)

    result = PreciseLineFitter().fit(inlier_points, ransac.params)

    assert result.mse < 0.005
    assert result.rmse < 0.08
    assert result.params["start"] == pytest.approx([0.0, 1.005], abs=0.08)
    assert result.params["end"] == pytest.approx([9.0, 18.995], abs=0.08)
    assert result.params["direction"] == pytest.approx([0.4472135955, 0.8944271910], rel=1e-3)
    assert result.parameter_delta["direction_angle"] < 0.02
    assert result.parameter_delta["start_distance"] < 0.08
    assert result.parameter_delta["end_distance"] < 0.08


def test_precise_circle_fitter_refits_ransac_inliers() -> None:
    center = (8.0, -4.0)
    radius = 6.0
    points = tuple(
        (
            center[0] + radius * math.cos(angle) + (0.03 if index % 2 == 0 else -0.02),
            center[1] + radius * math.sin(angle) + (0.02 if index % 3 == 0 else -0.01),
        )
        for index, angle in enumerate(index * (math.pi / 8.0) for index in range(12))
    ) + ((20.0, 20.0), (-10.0, 30.0))
    ransac = RansacCircleFitter(
        RansacCircleConfig(
            iterations=128,
            inlier_threshold=0.2,
            min_inlier_ratio=0.75,
            random_seed=5,
        )
    ).fit(points)
    inlier_points = tuple(points[index] for index in ransac.inlier_indexes)

    result = PreciseCircleFitter().fit(inlier_points, ransac.params)

    assert result.mse < 0.005
    assert result.rmse < 0.08
    assert result.params["cx"] == pytest.approx(center[0], abs=0.08)
    assert result.params["cy"] == pytest.approx(center[1], abs=0.08)
    assert result.params["r"] == pytest.approx(radius, abs=0.08)
    assert result.parameter_delta["center_distance"] < 0.08
    assert result.parameter_delta["radius_delta"] < 0.08


def test_precise_arc_fitter_refits_ransac_inliers() -> None:
    center = (15.0, -6.0)
    radius = 9.0
    angles = [math.radians(20.0 + (index * 10.0)) for index in range(11)]
    points = tuple(
        (
            center[0] + radius * math.cos(angle) + (0.025 if index % 2 == 0 else -0.015),
            center[1] + radius * math.sin(angle) + (0.02 if index % 3 == 0 else -0.01),
        )
        for index, angle in enumerate(angles)
    ) + ((50.0, 50.0), (5.0, -30.0))
    ransac = RansacArcFitter(
        RansacArcConfig(
            iterations=128,
            inlier_threshold=0.2,
            min_inlier_ratio=0.75,
            random_seed=19,
            min_arc_angle=math.radians(30.0),
            max_radial_error=0.08,
        )
    ).fit(points)
    inlier_points = tuple(points[index] for index in ransac.inlier_indexes)

    result = PreciseArcFitter().fit(inlier_points, ransac.params)

    assert result.mse < 0.005
    assert result.rmse < 0.08
    assert result.params["cx"] == pytest.approx(center[0], abs=0.08)
    assert result.params["cy"] == pytest.approx(center[1], abs=0.08)
    assert result.params["r"] == pytest.approx(radius, abs=0.08)
    assert result.params["direction"] == "ccw"
    assert result.params["start_angle"] == pytest.approx(angles[0], abs=0.08)
    assert result.params["end_angle"] == pytest.approx(angles[-1], abs=0.08)
    assert result.parameter_delta["center_distance"] < 0.08
    assert result.parameter_delta["radius_delta"] < 0.08
    assert result.parameter_delta["start_angle_delta"] < 0.08
    assert result.parameter_delta["end_angle_delta"] < 0.08
    assert result.parameter_delta["direction_changed"] is False


def test_precise_fitters_have_no_forbidden_dependencies() -> None:
    source_path = Path("services/refiner.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic", "ui"}

    assert imports.isdisjoint(forbidden_imports)
    assert "open(" not in source
