import ast
import math
from pathlib import Path

import pytest

from services.refiner import RansacCircleConfig, RansacCircleFitter


def test_ransac_circle_fitter_handles_noisy_circle_with_outliers() -> None:
    center_x = 10.0
    center_y = -5.0
    radius = 8.0
    angles = [index * (math.pi / 8.0) for index in range(12)]
    noisy_points = tuple(
        (
            center_x + radius * math.cos(angle) + (0.03 if index % 2 == 0 else -0.02),
            center_y + radius * math.sin(angle) + (0.02 if index % 3 == 0 else -0.01),
        )
        for index, angle in enumerate(angles)
    )
    outliers = ((30.0, 30.0), (-20.0, 5.0), (12.0, -25.0))
    points = noisy_points + outliers
    fitter = RansacCircleFitter(
        RansacCircleConfig(
            iterations=128,
            inlier_threshold=0.2,
            min_inlier_ratio=0.6,
            random_seed=11,
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


def test_ransac_circle_fitter_rejects_insufficient_inlier_ratio() -> None:
    points = ((0.0, 0.0), (10.0, 0.0), (4.0, 9.0), (50.0, 50.0), (-40.0, 12.0), (5.0, -30.0))
    fitter = RansacCircleFitter(
        RansacCircleConfig(
            iterations=32,
            inlier_threshold=0.05,
            min_inlier_ratio=0.75,
            random_seed=2,
        )
    )

    with pytest.raises(ValueError, match="insufficient inlier ratio"):
        fitter.fit(points)


def test_ransac_circle_fitter_requires_three_points() -> None:
    fitter = RansacCircleFitter()

    with pytest.raises(ValueError, match="at least three Vector Space points"):
        fitter.fit(((0.0, 0.0), (1.0, 1.0)))


def test_ransac_circle_fitter_rejects_degenerate_collinear_points() -> None:
    points = ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0))
    fitter = RansacCircleFitter(
        RansacCircleConfig(
            iterations=8,
            inlier_threshold=0.1,
            min_inlier_ratio=0.5,
            random_seed=1,
        )
    )

    with pytest.raises(ValueError, match="unable to fit a robust circle"):
        fitter.fit(points)


def test_ransac_circle_fitter_has_no_forbidden_dependencies() -> None:
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
