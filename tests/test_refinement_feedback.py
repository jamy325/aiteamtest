import ast
import json
import math
from pathlib import Path

import pytest

from services.fitting_confidence import FittingConfidenceResult
from services.refinement_feedback import (
    RefinementFeedback,
    RefinementFeedbackInputs,
)


def test_refinement_feedback_accepts_stable_fit_and_is_json_serializable() -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="line",
            inlier_ratio=0.91,
            fit_error=0.03,
            confidence_result=FittingConfidenceResult(confidence=0.92, failure_reason=None),
        )
    )

    assert result.success is True
    assert result.reason is None
    assert result.confidence == 0.92
    assert result.suggestion == "fit accepted"
    assert result.retry_policy == "accept"
    assert json.loads(json.dumps(result.to_dict())) == {
        "success": True,
        "reason": None,
        "inlier_ratio": 0.91,
        "fit_error": 0.03,
        "confidence": 0.92,
        "suggestion": "fit accepted",
        "retry_policy": "accept",
    }


def test_refinement_feedback_flags_low_inlier_ratio_before_other_failures() -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="line",
            inlier_ratio=0.4,
            fit_error=0.04,
            confidence_result=FittingConfidenceResult(confidence=0.25, failure_reason="parameter_unstable"),
        )
    )

    assert result.success is False
    assert result.reason == "low_inlier_ratio"
    assert result.suggestion == "retry with new split points or region classification"
    assert result.retry_policy == "retry_with_split_adjustment"


def test_refinement_feedback_flags_high_fit_error() -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="arc",
            inlier_ratio=0.82,
            fit_error=0.31,
            confidence_result=FittingConfidenceResult(confidence=0.8, failure_reason=None),
        )
    )

    assert result.success is False
    assert result.reason == "high_fit_error"
    assert result.suggestion == "retry with a different primitive or review contour noise"
    assert result.retry_policy == "retry_with_new_shape"


def test_refinement_feedback_normalizes_unstable_parameter_failures() -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="bezier",
            inlier_ratio=0.88,
            fit_error=0.05,
            confidence_result=FittingConfidenceResult(confidence=0.4, failure_reason="parameter_unstable"),
        )
    )

    assert result.success is False
    assert result.reason == "unstable_params"
    assert result.suggestion == "retry with a more stable segment type or wider support region"
    assert result.retry_policy == "retry_with_stability_guard"


def test_refinement_feedback_normalizes_high_rmse_from_confidence_result() -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="circle",
            inlier_ratio=0.87,
            fit_error=0.09,
            confidence_result=FittingConfidenceResult(confidence=0.38, failure_reason="high_rmse"),
        )
    )

    assert result.success is False
    assert result.reason == "high_fit_error"
    assert result.retry_policy == "retry_with_new_shape"


def test_refinement_feedback_flags_low_confidence_without_explicit_failure_reason() -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="line",
            inlier_ratio=0.84,
            fit_error=0.05,
            confidence_result=FittingConfidenceResult(confidence=0.3, failure_reason=None),
        )
    )

    assert result.success is False
    assert result.reason == "low_confidence"
    assert result.suggestion == "review line fitting proposal before retry"
    assert result.retry_policy == "manual_review"


@pytest.mark.parametrize("invalid_value", (math.nan, math.inf, -math.inf))
def test_refinement_feedback_rejects_invalid_inlier_ratio(invalid_value: float) -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="line",
            inlier_ratio=invalid_value,
            fit_error=0.01,
            confidence_result=FittingConfidenceResult(confidence=0.9, failure_reason=None),
        )
    )

    assert result.success is False
    assert result.reason == "invalid_numeric_input"
    assert result.inlier_ratio == 0.0
    assert result.fit_error == 0.01
    assert result.confidence == 0.9
    assert result.suggestion == "reject invalid fit metrics and recompute deterministically"
    assert result.retry_policy == "abort_and_recompute"
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("invalid_value", (-0.1, 1.1, 2.0))
def test_refinement_feedback_rejects_out_of_range_inlier_ratio(invalid_value: float) -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="line",
            inlier_ratio=invalid_value,
            fit_error=0.01,
            confidence_result=FittingConfidenceResult(confidence=0.9, failure_reason=None),
        )
    )

    assert result.success is False
    assert result.reason == "invalid_numeric_input"
    assert result.inlier_ratio == 0.0
    assert result.fit_error == 0.01
    assert result.confidence == 0.9
    assert result.retry_policy == "abort_and_recompute"
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("invalid_value", (math.nan, math.inf, -math.inf))
def test_refinement_feedback_rejects_invalid_fit_error(invalid_value: float) -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="arc",
            inlier_ratio=0.82,
            fit_error=invalid_value,
            confidence_result=FittingConfidenceResult(confidence=0.8, failure_reason=None),
        )
    )

    assert result.success is False
    assert result.reason == "invalid_numeric_input"
    assert result.inlier_ratio == 0.82
    assert result.fit_error == 0.0
    assert result.confidence == 0.8
    assert result.suggestion == "reject invalid fit metrics and recompute deterministically"
    assert result.retry_policy == "abort_and_recompute"
    json.dumps(result.to_dict(), allow_nan=False)


def test_refinement_feedback_rejects_negative_fit_error() -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="line",
            inlier_ratio=0.9,
            fit_error=-1.0,
            confidence_result=FittingConfidenceResult(confidence=0.9, failure_reason=None),
        )
    )

    assert result.success is False
    assert result.reason == "invalid_numeric_input"
    assert result.inlier_ratio == 0.9
    assert result.fit_error == 0.0
    assert result.confidence == 0.9
    assert result.retry_policy == "abort_and_recompute"
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("invalid_value", (math.nan, math.inf, -math.inf))
def test_refinement_feedback_rejects_invalid_confidence(invalid_value: float) -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="circle",
            inlier_ratio=0.87,
            fit_error=0.09,
            confidence_result=FittingConfidenceResult(confidence=invalid_value, failure_reason=None),
        )
    )

    assert result.success is False
    assert result.reason == "invalid_numeric_input"
    assert result.inlier_ratio == 0.87
    assert result.fit_error == 0.09
    assert result.confidence == 0.0
    assert result.suggestion == "reject invalid fit metrics and recompute deterministically"
    assert result.retry_policy == "abort_and_recompute"
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("invalid_value", (-0.1, 1.1, 2.0))
def test_refinement_feedback_rejects_out_of_range_confidence(invalid_value: float) -> None:
    feedback = RefinementFeedback()

    result = feedback.evaluate(
        RefinementFeedbackInputs(
            segment_type="circle",
            inlier_ratio=0.87,
            fit_error=0.09,
            confidence_result=FittingConfidenceResult(confidence=invalid_value, failure_reason=None),
        )
    )

    assert result.success is False
    assert result.reason == "invalid_numeric_input"
    assert result.inlier_ratio == 0.87
    assert result.fit_error == 0.09
    assert result.confidence == 0.0
    assert result.retry_policy == "abort_and_recompute"
    json.dumps(result.to_dict(), allow_nan=False)


def test_refinement_feedback_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/refinement_feedback.py")
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
