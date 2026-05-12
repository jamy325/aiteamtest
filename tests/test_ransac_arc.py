import ast
import math
from pathlib import Path

import pytest

from services.refiner import RansacArcConfig, RansacArcFitter


def test_ransac_arc_fitter_handles_noisy_arc_with_outliers() -> None:
    center_x = 20.0
    center_y = -10.0
    radius = 12.0
    angles = [math.radians(30.0 + (index * 12.0)) for index in range(12)]
    noisy_points = tuple(
        (
            center_x + radius * math.cos(angle) + (0.03 if index % 2 == 0 else -0.02),
            center_y + radius * math.sin(angle) + (0.02 if index % 3 == 0 else -0.01),
        )
        for index, angle in enumerate(angles)
    )
    outliers = ((50.0, 20.0), (-15.0, -40.0), (5.0, 45.0))
    points = noisy_points + outliers
    fitter = RansacArcFitter(
        RansacArcConfig(
            iterations=128,
            inlier_threshold=0.2,
            min_inlier_ratio=0.6,
            random_seed=13,
            min_arc_angle=math.radians(30.0),
            max_radial_error=0.08,
        )
    )

    result = fitter.fit(points)

    assert result.inlier_ratio == pytest.approx(12.0 / 15.0)
    assert result.fit_error < 0.08
    assert len(result.inlier_indexes) == 12
    assert set(result.outlier_indexes) == {12, 13, 14}
    assert result.params["cx"] == pytest.approx(center_x, abs=0.08)
    assert result.params["cy"] == pytest.approx(center_y, abs=0.08)
    assert result.params["r"] == pytest.approx(radius, abs=0.08)
    assert result.params["direction"] == "ccw"
    assert result.params["start_angle"] == pytest.approx(angles[0], abs=0.08)
    assert result.params["end_angle"] == pytest.approx(angles[-1], abs=0.08)


def test_ransac_arc_fitter_rejects_short_arc_below_minimum_angle() -> None:
    center = (5.0, 8.0)
    radius = 10.0
    angles = [0.05 + (index * 0.02) for index in range(5)]
    points = tuple(
        (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
        for angle in angles
    )
    fitter = RansacArcFitter(
        RansacArcConfig(
            iterations=16,
            inlier_threshold=0.05,
            min_inlier_ratio=0.8,
            random_seed=5,
            min_arc_angle=0.25,
            max_radial_error=0.1,
        )
    )

    with pytest.raises(ValueError, match="minimum arc angle"):
        fitter.fit(points)


def test_ransac_arc_fitter_rejects_excessive_radial_error() -> None:
    center = (0.0, 0.0)
    radius = 15.0
    angles = [math.radians(15.0 + (index * 15.0)) for index in range(10)]
    points = tuple(
        (
            center[0] + (radius + (0.11 if index % 2 == 0 else -0.12)) * math.cos(angle),
            center[1] + (radius + (0.11 if index % 2 == 0 else -0.12)) * math.sin(angle),
        )
        for index, angle in enumerate(angles)
    )
    fitter = RansacArcFitter(
        RansacArcConfig(
            iterations=96,
            inlier_threshold=0.25,
            min_inlier_ratio=1.0,
            random_seed=2,
            min_arc_angle=0.4,
            max_radial_error=0.05,
        )
    )

    with pytest.raises(ValueError, match="maximum radial error"):
        fitter.fit(points)


def test_ransac_arc_fitter_requires_three_points() -> None:
    fitter = RansacArcFitter()

    with pytest.raises(ValueError, match="at least three Vector Space points"):
        fitter.fit(((0.0, 0.0), (1.0, 1.0)))


def test_ransac_arc_fitter_has_no_forbidden_dependencies() -> None:
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
