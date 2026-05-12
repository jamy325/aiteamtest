from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FittingConfidenceConfig:
    min_inlier_ratio: float = 0.6
    target_inlier_ratio: float = 0.9
    target_rmse: float = 0.05
    max_rmse: float = 0.5
    min_segment_length: float = 2.0
    target_segment_length: float = 10.0
    target_radial_error: float = 0.03
    max_radial_error: float = 0.25
    min_arc_angle_coverage: float = math.pi / 6.0
    target_arc_angle_coverage: float = math.pi / 2.0
    max_line_direction_delta: float = 0.25
    max_line_endpoint_delta: float = 2.0
    max_line_offset_delta: float = 1.0
    max_center_delta: float = 1.0
    max_radius_delta: float = 1.0
    max_arc_angle_delta: float = 0.35
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "inlier_ratio": 0.3,
            "rmse": 0.25,
            "segment_length": 0.1,
            "parameter_stability": 0.2,
            "radial_error": 0.1,
            "arc_angle_coverage": 0.05,
        }
    )


@dataclass(frozen=True, slots=True)
class FittingConfidenceInputs:
    segment_type: str
    inlier_ratio: float
    rmse: float
    segment_length: float
    parameter_delta: dict[str, object]
    radial_error: float | None = None
    arc_angle_coverage: float | None = None


@dataclass(frozen=True, slots=True)
class FittingConfidenceResult:
    confidence: float
    failure_reason: str | None


class FittingConfidenceMetric:
    def __init__(self, config: FittingConfidenceConfig | None = None) -> None:
        self.config = config or FittingConfidenceConfig()

    def evaluate(self, inputs: FittingConfidenceInputs) -> FittingConfidenceResult:
        segment_type = inputs.segment_type.lower()
        missing_parameter_delta = self._missing_parameter_delta_fields(segment_type, inputs.parameter_delta)
        if missing_parameter_delta:
            return FittingConfidenceResult(confidence=0.0, failure_reason="missing_parameter_delta")
        if not self._has_valid_numeric_inputs(segment_type, inputs):
            return FittingConfidenceResult(confidence=0.0, failure_reason="invalid_numeric_input")

        scores = {
            "inlier_ratio": self._higher_is_better(
                inputs.inlier_ratio,
                minimum=self.config.min_inlier_ratio,
                target=self.config.target_inlier_ratio,
            ),
            "rmse": self._lower_is_better(
                inputs.rmse,
                target=self.config.target_rmse,
                maximum=self.config.max_rmse,
            ),
            "segment_length": self._higher_is_better(
                inputs.segment_length,
                minimum=self.config.min_segment_length,
                target=self.config.target_segment_length,
            ),
        }
        failure_reason = self._base_failure_reason(inputs)

        if segment_type == "line":
            scores["parameter_stability"] = self._line_stability_score(inputs.parameter_delta)
            if failure_reason is None and scores["parameter_stability"] < 0.5:
                failure_reason = "parameter_unstable"
            weight_keys = ("inlier_ratio", "rmse", "segment_length", "parameter_stability")
        elif segment_type == "circle":
            scores["parameter_stability"] = self._circle_stability_score(inputs.parameter_delta)
            scores["radial_error"] = self._radial_error_score(inputs.radial_error)
            if failure_reason is None and scores["radial_error"] < 0.25:
                failure_reason = "high_radial_error"
            if failure_reason is None and scores["parameter_stability"] < 0.5:
                failure_reason = "parameter_unstable"
            weight_keys = ("inlier_ratio", "rmse", "segment_length", "radial_error", "parameter_stability")
        elif segment_type == "arc":
            scores["parameter_stability"] = self._arc_stability_score(inputs.parameter_delta)
            scores["radial_error"] = self._radial_error_score(inputs.radial_error)
            scores["arc_angle_coverage"] = self._arc_coverage_score(inputs.arc_angle_coverage)
            if failure_reason is None and scores["radial_error"] < 0.25:
                failure_reason = "high_radial_error"
            if failure_reason is None and scores["arc_angle_coverage"] < 0.5:
                failure_reason = "low_arc_angle_coverage"
            if failure_reason is None and scores["parameter_stability"] < 0.5:
                failure_reason = "parameter_unstable"
            weight_keys = (
                "inlier_ratio",
                "rmse",
                "segment_length",
                "radial_error",
                "arc_angle_coverage",
                "parameter_stability",
            )
        else:
            raise ValueError(f"unsupported segment type: {inputs.segment_type}")

        confidence = self._weighted_average(scores, weight_keys) * min(scores[key] for key in weight_keys)
        confidence = self._clamp01(confidence)
        return FittingConfidenceResult(confidence=confidence, failure_reason=failure_reason)

    def _missing_parameter_delta_fields(self, segment_type: str, parameter_delta: dict[str, object]) -> tuple[str, ...]:
        required_fields_by_type = {
            "line": ("direction_angle", "start_distance", "end_distance", "line_offset"),
            "circle": ("center_distance", "radius_delta"),
            "arc": ("center_distance", "radius_delta", "start_angle_delta", "end_angle_delta", "direction_changed"),
        }
        if segment_type not in required_fields_by_type:
            raise ValueError(f"unsupported segment type: {segment_type}")
        return tuple(field for field in required_fields_by_type[segment_type] if field not in parameter_delta)

    def _has_valid_numeric_inputs(self, segment_type: str, inputs: FittingConfidenceInputs) -> bool:
        numeric_values = [inputs.inlier_ratio, inputs.rmse, inputs.segment_length]
        if inputs.radial_error is not None:
            numeric_values.append(inputs.radial_error)
        if inputs.arc_angle_coverage is not None:
            numeric_values.append(inputs.arc_angle_coverage)
        if not all(math.isfinite(value) for value in numeric_values):
            return False

        numeric_parameter_delta_fields_by_type = {
            "line": ("direction_angle", "start_distance", "end_distance", "line_offset"),
            "circle": ("center_distance", "radius_delta"),
            "arc": ("center_distance", "radius_delta", "start_angle_delta", "end_angle_delta"),
        }
        for field in numeric_parameter_delta_fields_by_type[segment_type]:
            value = inputs.parameter_delta[field]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return False

        if segment_type == "arc":
            direction_changed = inputs.parameter_delta["direction_changed"]
            if not isinstance(direction_changed, bool):
                return False
        return True

    def _base_failure_reason(self, inputs: FittingConfidenceInputs) -> str | None:
        if inputs.inlier_ratio < self.config.min_inlier_ratio:
            return "low_inlier_ratio"
        if inputs.rmse >= self.config.max_rmse:
            return "high_rmse"
        if inputs.segment_length < self.config.min_segment_length:
            return "segment_too_short"
        return None

    def _line_stability_score(self, parameter_delta: dict[str, object]) -> float:
        return self._average(
            (
                self._delta_score(float(parameter_delta.get("direction_angle", 0.0)), self.config.max_line_direction_delta),
                self._delta_score(float(parameter_delta.get("start_distance", 0.0)), self.config.max_line_endpoint_delta),
                self._delta_score(float(parameter_delta.get("end_distance", 0.0)), self.config.max_line_endpoint_delta),
                self._delta_score(float(parameter_delta.get("line_offset", 0.0)), self.config.max_line_offset_delta),
            )
        )

    def _circle_stability_score(self, parameter_delta: dict[str, object]) -> float:
        return self._average(
            (
                self._delta_score(float(parameter_delta.get("center_distance", 0.0)), self.config.max_center_delta),
                self._delta_score(float(parameter_delta.get("radius_delta", 0.0)), self.config.max_radius_delta),
            )
        )

    def _arc_stability_score(self, parameter_delta: dict[str, object]) -> float:
        direction_score = 0.0 if bool(parameter_delta.get("direction_changed", False)) else 1.0
        return self._average(
            (
                self._delta_score(float(parameter_delta.get("center_distance", 0.0)), self.config.max_center_delta),
                self._delta_score(float(parameter_delta.get("radius_delta", 0.0)), self.config.max_radius_delta),
                self._delta_score(float(parameter_delta.get("start_angle_delta", 0.0)), self.config.max_arc_angle_delta),
                self._delta_score(float(parameter_delta.get("end_angle_delta", 0.0)), self.config.max_arc_angle_delta),
                direction_score,
            )
        )

    def _radial_error_score(self, radial_error: float | None) -> float:
        if radial_error is None:
            return 0.0
        return self._lower_is_better(
            radial_error,
            target=self.config.target_radial_error,
            maximum=self.config.max_radial_error,
        )

    def _arc_coverage_score(self, arc_angle_coverage: float | None) -> float:
        if arc_angle_coverage is None:
            return 0.0
        return self._higher_is_better(
            arc_angle_coverage,
            minimum=self.config.min_arc_angle_coverage,
            target=self.config.target_arc_angle_coverage,
        )

    def _delta_score(self, delta: float, maximum: float) -> float:
        if maximum <= 0.0:
            return 0.0
        return self._clamp01(1.0 - (max(0.0, delta) / maximum))

    def _higher_is_better(self, value: float, *, minimum: float, target: float) -> float:
        if target <= minimum:
            return 1.0 if value >= target else 0.0
        return self._clamp01((value - minimum) / (target - minimum))

    def _lower_is_better(self, value: float, *, target: float, maximum: float) -> float:
        if value <= target:
            return 1.0
        if value >= maximum:
            return 0.0
        return self._clamp01((maximum - value) / (maximum - target))

    def _weighted_average(self, scores: dict[str, float], keys: tuple[str, ...]) -> float:
        total_weight = sum(self.config.weights[key] for key in keys)
        weighted_sum = sum(self.config.weights[key] * scores[key] for key in keys)
        if total_weight <= 0.0:
            return 0.0
        return self._clamp01(weighted_sum / total_weight)

    def _average(self, values: tuple[float, ...]) -> float:
        if not values:
            return 0.0
        return self._clamp01(sum(values) / len(values))

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))


__all__ = [
    "FittingConfidenceConfig",
    "FittingConfidenceInputs",
    "FittingConfidenceMetric",
    "FittingConfidenceResult",
]
