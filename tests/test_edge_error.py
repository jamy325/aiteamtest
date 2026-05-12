import ast
import math
from pathlib import Path

import pytest

from services.edge_error import EdgeErrorCalculator


def test_edge_error_calculator_reports_zero_for_identical_line_samples() -> None:
    calculator = EdgeErrorCalculator()
    source_points = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    vector_points = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))

    result = calculator.calculate(source_points, vector_points)

    assert result.missing_edge_error == pytest.approx(0.0)
    assert result.overdraw_error == pytest.approx(0.0)
    assert result.chamfer_error == pytest.approx(0.0)
    assert result.source_point_count == 3
    assert result.vector_point_count == 3


def test_edge_error_calculator_reports_bidirectional_error_for_shifted_line_samples() -> None:
    calculator = EdgeErrorCalculator()
    source_points = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    vector_points = ((0.0, 1.0), (1.0, 1.0), (2.0, 1.0))

    result = calculator.calculate(source_points, vector_points)

    assert result.missing_edge_error == pytest.approx(1.0)
    assert result.overdraw_error == pytest.approx(1.0)
    assert result.chamfer_error == pytest.approx(1.0)


def test_edge_error_calculator_reports_missing_edge_for_closed_contour_gap() -> None:
    calculator = EdgeErrorCalculator()
    source_points = (
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (1.0, 2.0),
        (0.0, 2.0),
        (0.0, 1.0),
    )
    vector_points = (
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (1.0, 2.0),
        (0.0, 2.0),
    )

    result = calculator.calculate(source_points, vector_points)

    assert result.missing_edge_error == pytest.approx(0.125)
    assert result.overdraw_error == pytest.approx(0.0)
    assert result.chamfer_error == pytest.approx(0.0625)


def test_edge_error_calculator_handles_empty_vector_samples_without_mutating_inputs() -> None:
    calculator = EdgeErrorCalculator()
    source_points = [[0.0, 0.0], [1.0, 0.0]]
    vector_points: list[list[float]] = []
    original_source_points = [point[:] for point in source_points]
    original_vector_points = [point[:] for point in vector_points]

    result = calculator.calculate(source_points, vector_points)

    assert math.isinf(result.missing_edge_error)
    assert result.overdraw_error == pytest.approx(0.0)
    assert math.isinf(result.chamfer_error)
    assert source_points == original_source_points
    assert vector_points == original_vector_points


def test_edge_error_service_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/edge_error.py")
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
