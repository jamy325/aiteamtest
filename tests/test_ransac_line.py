import ast
from pathlib import Path

import pytest

from services.refiner import RansacLineConfig, RansacLineFitter


def test_ransac_line_fitter_handles_line_with_outliers() -> None:
    points = tuple(
        [(float(index), 2.0 * float(index) + 1.0) for index in range(8)]
        + [(2.0, 40.0), (7.5, -12.0), (20.0, 3.0)]
    )
    fitter = RansacLineFitter(
        RansacLineConfig(
            iterations=80,
            inlier_threshold=0.15,
            min_inlier_ratio=0.5,
            random_seed=7,
        )
    )

    result = fitter.fit(points)

    assert result.inlier_ratio == pytest.approx(8.0 / 11.0)
    assert result.fit_error < 1e-6
    assert len(result.inlier_indexes) == 8
    assert set(result.outlier_indexes) == {8, 9, 10}
    assert result.params["start"] == pytest.approx([0.0, 1.0])
    assert result.params["end"] == pytest.approx([7.0, 15.0])
    assert result.params["direction"] == pytest.approx([0.4472135955, 0.8944271910], rel=1e-5)


def test_ransac_line_fitter_supports_vertical_lines() -> None:
    points = tuple([(3.0, float(index)) for index in range(6)] + [(20.0, 20.0)])
    fitter = RansacLineFitter(
        RansacLineConfig(
            iterations=40,
            inlier_threshold=0.05,
            min_inlier_ratio=0.6,
            random_seed=3,
        )
    )

    result = fitter.fit(points)

    assert result.inlier_ratio == pytest.approx(6.0 / 7.0)
    assert result.fit_error < 1e-6
    assert result.params["start"] == pytest.approx([3.0, 0.0])
    assert result.params["end"] == pytest.approx([3.0, 5.0])


def test_ransac_line_fitter_rejects_insufficient_inlier_ratio() -> None:
    points = ((0.0, 0.0), (1.0, 5.0), (10.0, -4.0), (20.0, 7.0))
    fitter = RansacLineFitter(
        RansacLineConfig(
            iterations=20,
            inlier_threshold=0.01,
            min_inlier_ratio=0.75,
            random_seed=1,
        )
    )

    with pytest.raises(ValueError, match="insufficient inlier ratio"):
        fitter.fit(points)


def test_ransac_line_fitter_requires_at_least_two_points() -> None:
    fitter = RansacLineFitter()

    with pytest.raises(ValueError, match="at least two Vector Space points"):
        fitter.fit(((0.0, 0.0),))


def test_ransac_line_fitter_has_no_forbidden_dependencies() -> None:
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
