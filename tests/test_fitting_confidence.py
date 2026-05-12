import ast
import math
from pathlib import Path

import pytest

from services.fitting_confidence import (
    FittingConfidenceConfig,
    FittingConfidenceInputs,
    FittingConfidenceMetric,
)


def test_fitting_confidence_metric_scores_high_confidence_line_fit() -> None:
    metric = FittingConfidenceMetric()

    result = metric.evaluate(
        FittingConfidenceInputs(
            segment_type="line",
            inlier_ratio=0.94,
            rmse=0.02,
            segment_length=24.0,
            parameter_delta={
                "direction_angle": 0.01,
                "start_distance": 0.08,
                "end_distance": 0.06,
                "line_offset": 0.03,
            },
        )
    )

    assert result.failure_reason is None
    assert result.confidence == pytest.approx(0.95, abs=0.05)
    assert 0.0 <= result.confidence <= 1.0


def test_fitting_confidence_metric_flags_low_inlier_ratio() -> None:
    metric = FittingConfidenceMetric()

    result = metric.evaluate(
        FittingConfidenceInputs(
            segment_type="line",
            inlier_ratio=0.35,
            rmse=0.03,
            segment_length=20.0,
            parameter_delta={
                "direction_angle": 0.02,
                "start_distance": 0.05,
                "end_distance": 0.05,
                "line_offset": 0.01,
            },
        )
    )

    assert result.failure_reason == "low_inlier_ratio"
    assert 0.0 <= result.confidence < 0.45


def test_fitting_confidence_metric_flags_high_error_circle_fit() -> None:
    metric = FittingConfidenceMetric()

    result = metric.evaluate(
        FittingConfidenceInputs(
            segment_type="circle",
            inlier_ratio=0.88,
            rmse=0.42,
            segment_length=28.0,
            radial_error=0.22,
            parameter_delta={
                "center_distance": 0.12,
                "radius_delta": 0.08,
            },
        )
    )

    assert result.failure_reason == "high_radial_error"
    assert 0.0 <= result.confidence < 0.5


def test_fitting_confidence_metric_flags_unstable_arc_parameters() -> None:
    metric = FittingConfidenceMetric(
        FittingConfidenceConfig(
            min_arc_angle_coverage=math.pi / 5.0,
            target_arc_angle_coverage=math.pi / 2.0,
        )
    )

    result = metric.evaluate(
        FittingConfidenceInputs(
            segment_type="arc",
            inlier_ratio=0.9,
            rmse=0.03,
            segment_length=18.0,
            radial_error=0.03,
            arc_angle_coverage=math.pi * 0.7,
            parameter_delta={
                "center_distance": 1.4,
                "radius_delta": 1.2,
                "start_angle_delta": 0.4,
                "end_angle_delta": 0.45,
                "direction_changed": True,
            },
        )
    )

    assert result.failure_reason == "parameter_unstable"
    assert 0.0 <= result.confidence < 0.5


@pytest.mark.parametrize(
    ("segment_type", "parameter_delta"),
    (
        ("line", {}),
        ("circle", {"center_distance": 0.1}),
        ("arc", {"center_distance": 0.1, "radius_delta": 0.1, "start_angle_delta": 0.05}),
    ),
)
def test_fitting_confidence_metric_rejects_missing_parameter_delta_fields(
    segment_type: str,
    parameter_delta: dict[str, object],
) -> None:
    metric = FittingConfidenceMetric()
    inputs = FittingConfidenceInputs(
        segment_type=segment_type,
        inlier_ratio=0.9,
        rmse=0.02,
        segment_length=12.0,
        radial_error=0.02 if segment_type != "line" else None,
        arc_angle_coverage=math.pi if segment_type == "arc" else None,
        parameter_delta=parameter_delta,
    )

    result = metric.evaluate(inputs)

    assert result.confidence == 0.0
    assert result.failure_reason == "missing_parameter_delta"


@pytest.mark.parametrize("invalid_value", (math.nan, math.inf, -math.inf))
def test_fitting_confidence_metric_rejects_invalid_numeric_inputs(invalid_value: float) -> None:
    metric = FittingConfidenceMetric()

    result = metric.evaluate(
        FittingConfidenceInputs(
            segment_type="line",
            inlier_ratio=invalid_value,
            rmse=0.02,
            segment_length=12.0,
            parameter_delta={
                "direction_angle": 0.01,
                "start_distance": 0.02,
                "end_distance": 0.02,
                "line_offset": 0.01,
            },
        )
    )

    assert result.confidence == 0.0
    assert result.failure_reason == "invalid_numeric_input"


def test_fitting_confidence_metric_rejects_invalid_parameter_delta_numeric_inputs() -> None:
    metric = FittingConfidenceMetric()

    result = metric.evaluate(
        FittingConfidenceInputs(
            segment_type="arc",
            inlier_ratio=0.9,
            rmse=0.02,
            segment_length=12.0,
            radial_error=0.02,
            arc_angle_coverage=math.pi,
            parameter_delta={
                "center_distance": math.nan,
                "radius_delta": 0.02,
                "start_angle_delta": 0.01,
                "end_angle_delta": 0.01,
                "direction_changed": False,
            },
        )
    )

    assert result.confidence == 0.0
    assert result.failure_reason == "invalid_numeric_input"


def test_fitting_confidence_metric_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/fitting_confidence.py")
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
