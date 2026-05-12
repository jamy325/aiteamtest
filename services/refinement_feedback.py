from __future__ import annotations

from dataclasses import asdict, dataclass

from services.fitting_confidence import FittingConfidenceResult


@dataclass(frozen=True, slots=True)
class RefinementFeedbackConfig:
    min_inlier_ratio: float = 0.6
    max_fit_error: float = 0.25
    min_confidence: float = 0.65


@dataclass(frozen=True, slots=True)
class RefinementFeedbackInputs:
    segment_type: str
    inlier_ratio: float
    fit_error: float
    confidence_result: FittingConfidenceResult


@dataclass(frozen=True, slots=True)
class RefinementFeedbackResult:
    success: bool
    reason: str | None
    inlier_ratio: float
    fit_error: float
    confidence: float
    suggestion: str
    retry_policy: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RefinementFeedback:
    def __init__(self, config: RefinementFeedbackConfig | None = None) -> None:
        self.config = config or RefinementFeedbackConfig()

    def evaluate(self, inputs: RefinementFeedbackInputs) -> RefinementFeedbackResult:
        reason = self._failure_reason(inputs)
        success = reason is None
        suggestion, retry_policy = self._suggestion(reason, inputs.segment_type)
        return RefinementFeedbackResult(
            success=success,
            reason=reason,
            inlier_ratio=inputs.inlier_ratio,
            fit_error=inputs.fit_error,
            confidence=inputs.confidence_result.confidence,
            suggestion=suggestion,
            retry_policy=retry_policy,
        )

    def _failure_reason(self, inputs: RefinementFeedbackInputs) -> str | None:
        if inputs.inlier_ratio < self.config.min_inlier_ratio:
            return "low_inlier_ratio"
        if inputs.fit_error > self.config.max_fit_error:
            return "high_fit_error"
        if inputs.confidence_result.failure_reason is not None:
            return self._normalize_confidence_reason(inputs.confidence_result.failure_reason)
        if inputs.confidence_result.confidence < self.config.min_confidence:
            return "low_confidence"
        return None

    def _normalize_confidence_reason(self, reason: str) -> str:
        allowed_reasons = {
            "low_inlier_ratio",
            "high_fit_error",
            "unstable_params",
            "low_confidence",
            "high_radial_error",
            "low_arc_angle_coverage",
            "missing_parameter_delta",
            "invalid_numeric_input",
            "segment_too_short",
        }
        if reason == "parameter_unstable":
            return "unstable_params"
        if reason == "high_rmse":
            return "high_fit_error"
        return reason if reason in allowed_reasons else "low_confidence"

    def _suggestion(self, reason: str | None, segment_type: str) -> tuple[str, str]:
        if reason is None:
            return ("fit accepted", "accept")
        if reason == "low_inlier_ratio":
            return ("retry with new split points or region classification", "retry_with_split_adjustment")
        if reason == "high_fit_error":
            return ("retry with a different primitive or review contour noise", "retry_with_new_shape")
        if reason == "unstable_params":
            return ("retry with a more stable segment type or wider support region", "retry_with_stability_guard")
        if reason == "high_radial_error":
            return ("review radial residuals and consider non-circular geometry", "retry_with_new_shape")
        if reason == "low_arc_angle_coverage":
            return ("extend the arc support range or reconsider the arc proposal", "retry_with_more_coverage")
        if reason == "missing_parameter_delta":
            return ("re-run precise fitting before feedback evaluation", "retry_after_precise_fit")
        if reason == "invalid_numeric_input":
            return ("reject invalid fit metrics and recompute deterministically", "abort_and_recompute")
        if reason == "segment_too_short":
            return ("collect a longer support span before fitting", "retry_with_longer_segment")
        return (f"review {segment_type} fitting proposal before retry", "manual_review")


__all__ = [
    "RefinementFeedback",
    "RefinementFeedbackConfig",
    "RefinementFeedbackInputs",
    "RefinementFeedbackResult",
]
