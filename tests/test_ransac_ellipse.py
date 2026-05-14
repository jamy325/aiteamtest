from __future__ import annotations

import ast
import math
import random
import warnings
from pathlib import Path

import pytest

from services.ellipse_fitter import PreciseEllipseFitter, RansacEllipseConfig, RansacEllipseFitter
from services.refiner import PreciseEllipseFitter as ExportedPreciseEllipseFitter
from services.refiner import RansacEllipseConfig as ExportedRansacEllipseConfig
from services.refiner import RansacEllipseFitter as ExportedRansacEllipseFitter


def generate_ellipse_points(
    *,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    rotation: float,
    count: int,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[tuple[float, float], ...]:
    rng = random.Random(seed)
    points: list[tuple[float, float]] = []
    cos_theta = math.cos(rotation)
    sin_theta = math.sin(rotation)

    for index in range(count):
        angle = (2.0 * math.pi * index) / count
        local_x = rx * math.cos(angle)
        local_y = ry * math.sin(angle)
        x = cx + (local_x * cos_theta) - (local_y * sin_theta)
        y = cy + (local_x * sin_theta) + (local_y * cos_theta)
        if noise > 0.0:
            x += rng.gauss(0.0, noise)
            y += rng.gauss(0.0, noise)
        points.append((x, y))

    return tuple(points)


def angle_close_mod_pi(left: float, right: float, tol: float) -> bool:
    delta = (left - right + (math.pi / 2.0)) % math.pi - (math.pi / 2.0)
    return abs(delta) <= tol


def assert_ellipse_matches(
    result: object,
    *,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    rotation: float,
    center_tol: float,
    axis_tol: float,
    rotation_tol: float,
) -> None:
    assert getattr(result, "cx") == pytest.approx(cx, abs=center_tol)
    assert getattr(result, "cy") == pytest.approx(cy, abs=center_tol)
    assert getattr(result, "rx") == pytest.approx(rx, abs=axis_tol)
    assert getattr(result, "ry") == pytest.approx(ry, abs=axis_tol)
    assert angle_close_mod_pi(getattr(result, "rotation"), rotation, rotation_tol)


def test_precise_ellipse_fitter_handles_noiseless_rotated_ellipse() -> None:
    params = {
        "cx": 12.0,
        "cy": -4.5,
        "rx": 8.0,
        "ry": 3.5,
        "rotation": math.radians(32.0),
    }
    points = generate_ellipse_points(**params, count=48, noise=0.0, seed=7)

    result = PreciseEllipseFitter().fit(points)

    assert_ellipse_matches(
        result,
        **params,
        center_tol=0.02,
        axis_tol=0.02,
        rotation_tol=0.02,
    )
    assert result.fit_error >= 0.0
    assert result.fit_error < 1e-6
    assert result.inlier_count == len(points)
    assert result.outlier_count == 0
    assert result.inlier_ratio == pytest.approx(1.0)


def test_ransac_ellipse_fitter_handles_noisy_rotated_ellipse() -> None:
    params = {
        "cx": -6.0,
        "cy": 18.0,
        "rx": 10.0,
        "ry": 4.0,
        "rotation": math.radians(-28.0),
    }
    points = generate_ellipse_points(**params, count=72, noise=0.08, seed=19)
    fitter = RansacEllipseFitter(
        RansacEllipseConfig(
            max_iterations=320,
            max_error=0.35,
            min_inlier_ratio=0.85,
            random_seed=23,
        )
    )

    result = fitter.fit(points)

    assert_ellipse_matches(
        result,
        **params,
        center_tol=0.2,
        axis_tol=0.2,
        rotation_tol=0.08,
    )
    assert result.fit_error >= 0.0
    assert result.fit_error < 0.2
    assert result.inlier_count == len(points)
    assert result.outlier_count == 0
    assert result.inlier_ratio == pytest.approx(1.0)


def test_ransac_ellipse_fitter_rejects_outliers_and_reports_inlier_ratio() -> None:
    params = {
        "cx": 4.0,
        "cy": 11.0,
        "rx": 9.0,
        "ry": 5.0,
        "rotation": math.radians(41.0),
    }
    inliers = generate_ellipse_points(**params, count=40, noise=0.05, seed=31)
    outliers = (
        (-25.0, 19.0),
        (30.0, -40.0),
        (12.0, 35.0),
        (42.0, 18.0),
        (-18.0, -22.0),
    )
    points = inliers + outliers
    fitter = RansacEllipseFitter(
        RansacEllipseConfig(
            max_iterations=400,
            max_error=0.3,
            min_inlier_ratio=0.75,
            random_seed=13,
        )
    )

    result = fitter.fit(points)

    assert_ellipse_matches(
        result,
        **params,
        center_tol=0.25,
        axis_tol=0.25,
        rotation_tol=0.1,
    )
    assert result.fit_error >= 0.0
    assert result.fit_error < 0.18
    assert result.inlier_count >= 39
    assert result.outlier_count >= 4
    assert result.inlier_ratio == pytest.approx(result.inlier_count / len(points))
    assert result.inlier_ratio < 1.0
    assert len(result.inlier_indexes) == result.inlier_count
    assert len(result.outlier_indexes) == result.outlier_count


def test_ransac_ellipse_fitter_rejects_insufficient_points() -> None:
    fitter = RansacEllipseFitter()

    with pytest.raises(ValueError, match="at least five Vector Space points"):
        fitter.fit(((0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)))


def test_ransac_ellipse_fitter_rejects_approximately_collinear_points() -> None:
    points = tuple((float(index), (2.0 * float(index)) + ((-1.0) ** index) * 1e-4) for index in range(8))
    fitter = RansacEllipseFitter(
        RansacEllipseConfig(
            max_iterations=64,
            max_error=0.2,
            min_inlier_ratio=0.8,
            random_seed=5,
        )
    )

    with pytest.raises(ValueError, match="collinear|robust ellipse"):
        fitter.fit(points)


def test_ransac_ellipse_fitter_rejects_near_circular_unstable_fit() -> None:
    points = generate_ellipse_points(
        cx=3.0,
        cy=-2.0,
        rx=6.0,
        ry=5.94,
        rotation=math.radians(18.0),
        count=36,
        noise=0.01,
        seed=11,
    )
    fitter = RansacEllipseFitter(
        RansacEllipseConfig(
            max_iterations=128,
            max_error=0.2,
            min_inlier_ratio=0.9,
            random_seed=7,
            min_axis_ratio_delta=0.05,
        )
    )

    with pytest.raises(ValueError, match="near-circular|unstable"):
        fitter.fit(points)


def test_precise_ellipse_fitter_rejects_parabola_like_non_ellipse_conic() -> None:
    points = tuple((float(x), float(x * x) / 10.0) for x in range(-8, 9))

    with pytest.raises(ValueError, match="ellipse|conic|axis|center|degenerate|implausible"):
        PreciseEllipseFitter().fit(points)


def test_ransac_ellipse_fitter_rejects_parabola_like_non_ellipse_conic() -> None:
    points = tuple((float(x), float(x * x) / 10.0) for x in range(-8, 9))
    fitter = RansacEllipseFitter(
        RansacEllipseConfig(
            max_iterations=200,
            max_error=0.2,
            min_inlier_ratio=0.8,
            random_seed=4,
        )
    )

    with pytest.raises(ValueError, match="ellipse|conic|axis|center|degenerate|implausible|robust"):
        fitter.fit(points)


def test_ransac_ellipse_fitter_does_not_emit_complex_warning_on_non_ellipse_conic() -> None:
    points = tuple((float(x), float(x * x) / 10.0) for x in range(-8, 9))
    fitter = RansacEllipseFitter(
        RansacEllipseConfig(
            max_iterations=80,
            max_error=0.2,
            min_inlier_ratio=0.8,
            random_seed=4,
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            fitter.fit(points)

    assert not any(type(item.message).__name__ == "ComplexWarning" for item in caught)


def test_refiner_exports_ellipse_fitters() -> None:
    assert ExportedRansacEllipseFitter is RansacEllipseFitter
    assert ExportedRansacEllipseConfig is RansacEllipseConfig
    assert ExportedPreciseEllipseFitter is PreciseEllipseFitter


def test_ellipse_fitter_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/ellipse_fitter.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic", "ui", "scipy"}

    assert imports.isdisjoint(forbidden_imports)
    assert "open(" not in source
