from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_PROFILE_NAMES: tuple[str, ...] = (
    "circle",
    "ellipse",
    "line_arc",
    "bezier_fallback",
    "logo",
    "mechanical",
    "transparent",
    "noisy_scan",
)

PROFILE_CASE_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("concentric", "circle"),
    ("circle", "circle"),
    ("ellipse", "ellipse"),
    ("line_arc", "line_arc"),
    ("logo", "logo"),
    ("mechanical", "mechanical"),
    ("transparent", "transparent"),
    ("noisy_scan", "noisy_scan"),
    ("bezier", "bezier_fallback"),
)

PROFILE_GEOMETRY_RULES: tuple[tuple[str, str], ...] = (
    ("circle", "circle"),
    ("ellipse", "ellipse"),
    ("line", "line_arc"),
    ("arc", "line_arc"),
    ("bezier", "bezier_fallback"),
)

HIGHER_IS_WORSE_METRICS: tuple[str, ...] = (
    "total_score",
    "edge_error",
    "complexity_score",
    "requires_external_decision_count",
)

LOWER_IS_WORSE_METRICS: tuple[str, ...] = (
    "algorithm_confidence",
    "inlier_ratio",
)

DEFAULT_PROFILE_THRESHOLDS: dict[str, float | int] = {
    "max_total_score": 0.0,
    "max_edge_error": 0.0,
    "max_complexity_score": 0.0,
    "min_algorithm_confidence": 0.55,
    "min_inlier_ratio": 0.6,
    "max_requires_external_decision_count": 0,
}


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    case_id: str
    suite_name: str
    case_type: str
    geometry_type: str
    success: bool
    metrics: dict[str, float]
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class QualityMetricCalibrationReport:
    profiles: dict[str, dict[str, Any]]
    summary: dict[str, Any]
    warnings: tuple[str, ...]
    markdown: str

    def write(
        self,
        *,
        output_json: str | Path | None = None,
        report_path: str | Path | None = None,
        dry_run: bool = False,
    ) -> None:
        if output_json is not None and not dry_run:
            path = Path(output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.profiles, indent=2, sort_keys=True), encoding="utf-8")
        if report_path is not None:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.markdown, encoding="utf-8")


class QualityMetricCalibrator:
    def calibrate(
        self,
        input_dirs: str | Path | Sequence[str | Path],
        *,
        output_json: str | Path | None = None,
        report_path: str | Path | None = None,
        dry_run: bool = False,
    ) -> QualityMetricCalibrationReport:
        warnings: list[str] = []
        samples = self._collect_samples(input_dirs, warnings=warnings)
        profiles = self._build_profiles(samples)
        summary = self._build_summary(samples, profiles, warnings=warnings, dry_run=dry_run)
        markdown = self._build_markdown(samples, profiles, warnings=warnings, dry_run=dry_run)
        report = QualityMetricCalibrationReport(
            profiles=profiles,
            summary=summary,
            warnings=tuple(warnings),
            markdown=markdown,
        )
        report.write(output_json=output_json, report_path=report_path, dry_run=dry_run)
        return report

    def _collect_samples(
        self,
        input_dirs: str | Path | Sequence[str | Path],
        *,
        warnings: list[str],
    ) -> tuple[CalibrationSample, ...]:
        roots = self._normalize_input_dirs(input_dirs)
        samples: list[CalibrationSample] = []
        for root in roots:
            if not root.exists():
                warnings.append(f"missing_input_dir: {root}")
                continue
            suite_name, suite_report_path = self._suite_identity(root)
            suite_report = self._read_json(suite_report_path, warnings=warnings)
            raw_cases = suite_report.get("cases") if isinstance(suite_report, Mapping) else None
            if not isinstance(raw_cases, list):
                warnings.append(f"missing_suite_cases: {root}")
                continue
            for item in raw_cases:
                if not isinstance(item, Mapping):
                    warnings.append(f"invalid_case_entry: {root}")
                    continue
                case_id = item.get("case_id")
                if not isinstance(case_id, str):
                    warnings.append(f"missing_case_id: {root}")
                    continue
                metrics = self._coerce_metrics(item.get("metrics"), case_id=case_id, warnings=warnings)
                case_type = self._infer_case_type(case_id)
                geometry_type = self._infer_geometry_type(item, case_type=case_type)
                samples.append(
                    CalibrationSample(
                        case_id=case_id,
                        suite_name=suite_name,
                        case_type=case_type,
                        geometry_type=geometry_type,
                        success=bool(item.get("success", True)),
                        metrics=metrics,
                        failure_reason=None if item.get("failure_reason") is None else str(item.get("failure_reason")),
                    )
                )
        return tuple(samples)

    def _build_profiles(self, samples: Sequence[CalibrationSample]) -> dict[str, dict[str, Any]]:
        grouped: defaultdict[str, list[CalibrationSample]] = defaultdict(list)
        for sample in samples:
            grouped[sample.case_type].append(sample)
            if sample.geometry_type != sample.case_type:
                grouped[sample.geometry_type].append(sample)

        profiles: dict[str, dict[str, Any]] = {}
        for profile_name in sorted(set(grouped) | set(REQUIRED_PROFILE_NAMES)):
            profiles[profile_name] = self._profile_for_group(profile_name, grouped.get(profile_name, ()), samples)
        return profiles

    def _profile_for_group(
        self,
        profile_name: str,
        grouped_samples: Sequence[CalibrationSample],
        all_samples: Sequence[CalibrationSample],
    ) -> dict[str, Any]:
        success_samples = [sample for sample in grouped_samples if sample.success]
        fallback_samples = [sample for sample in all_samples if sample.success]
        source_samples = success_samples or fallback_samples

        total_scores = self._trim_outliers([sample.metrics.get("total_score") for sample in source_samples], direction="higher")
        edge_errors = self._trim_outliers([sample.metrics.get("edge_error") for sample in source_samples], direction="higher")
        complexity_scores = self._trim_outliers([sample.metrics.get("complexity_score") for sample in source_samples], direction="higher")
        requires_external = self._trim_outliers(
            [sample.metrics.get("requires_external_decision_count") for sample in source_samples],
            direction="higher",
        )
        algorithm_confidences = self._trim_outliers(
            [self._algorithm_confidence(sample) for sample in source_samples],
            direction="lower",
        )
        inlier_ratios = self._trim_outliers(
            [self._inlier_ratio(sample) for sample in source_samples],
            direction="lower",
        )

        failure_count = sum(1 for sample in grouped_samples if not sample.success)
        sample_count = len(grouped_samples)
        failure_rate = (failure_count / sample_count) if sample_count else 0.0

        thresholds = {
            "max_total_score": self._higher_threshold(total_scores, default=DEFAULT_PROFILE_THRESHOLDS["max_total_score"]),
            "max_edge_error": self._higher_threshold(edge_errors, default=DEFAULT_PROFILE_THRESHOLDS["max_edge_error"]),
            "max_complexity_score": self._higher_threshold(
                complexity_scores,
                default=DEFAULT_PROFILE_THRESHOLDS["max_complexity_score"],
            ),
            "min_algorithm_confidence": self._lower_threshold(
                algorithm_confidences,
                default=float(DEFAULT_PROFILE_THRESHOLDS["min_algorithm_confidence"]),
            ),
            "min_inlier_ratio": self._lower_threshold(
                inlier_ratios,
                default=float(DEFAULT_PROFILE_THRESHOLDS["min_inlier_ratio"]),
            ),
            "max_requires_external_decision_count": int(
                round(self._higher_threshold(requires_external, default=DEFAULT_PROFILE_THRESHOLDS["max_requires_external_decision_count"]))
            ),
        }

        return {
            "profile_name": profile_name,
            "sample_count": sample_count,
            "success_sample_count": len(success_samples),
            "failure_count": failure_count,
            "failure_rate": failure_rate,
            "case_types": sorted({sample.case_type for sample in grouped_samples}) or [profile_name],
            "geometry_types": sorted({sample.geometry_type for sample in grouped_samples}) or [profile_name],
            "recommended_thresholds": thresholds,
            "metric_stats": {
                "total_score": self._metric_stats(total_scores),
                "edge_error": self._metric_stats(edge_errors),
                "complexity_score": self._metric_stats(complexity_scores),
                "requires_external_decision_count": self._metric_stats(requires_external),
                "algorithm_confidence": self._metric_stats(algorithm_confidences),
                "inlier_ratio": self._metric_stats(inlier_ratios),
            },
        }

    def _build_summary(
        self,
        samples: Sequence[CalibrationSample],
        profiles: Mapping[str, Mapping[str, Any]],
        *,
        warnings: Sequence[str],
        dry_run: bool,
    ) -> dict[str, Any]:
        return {
            "dry_run": dry_run,
            "total_sample_count": len(samples),
            "success_sample_count": sum(1 for sample in samples if sample.success),
            "failure_sample_count": sum(1 for sample in samples if not sample.success),
            "profile_count": len(profiles),
            "profile_names": sorted(profiles.keys()),
            "warning_count": len(warnings),
            "warnings": list(warnings),
        }

    def _build_markdown(
        self,
        samples: Sequence[CalibrationSample],
        profiles: Mapping[str, Mapping[str, Any]],
        *,
        warnings: Sequence[str],
        dry_run: bool,
    ) -> str:
        lines = [
            "# Quality Metric Calibration",
            "",
            "## Summary",
            "",
            f"- Dry run: `{dry_run}`",
            f"- Total samples: `{len(samples)}`",
            f"- Profiles: `{len(profiles)}`",
            f"- Warnings: `{len(warnings)}`",
            "",
            "## Recommended Profiles",
            "",
        ]
        for profile_name, payload in profiles.items():
            thresholds = payload["recommended_thresholds"]
            lines.extend(
                [
                    f"### {profile_name}",
                    "",
                    f"- Samples: `{payload['sample_count']}`",
                    f"- Failure rate: `{payload['failure_rate']:.3f}`",
                    f"- `max_total_score`: `{thresholds['max_total_score']}`",
                    f"- `max_edge_error`: `{thresholds['max_edge_error']}`",
                    f"- `max_complexity_score`: `{thresholds['max_complexity_score']}`",
                    f"- `min_algorithm_confidence`: `{thresholds['min_algorithm_confidence']}`",
                    f"- `min_inlier_ratio`: `{thresholds['min_inlier_ratio']}`",
                    f"- `max_requires_external_decision_count`: `{thresholds['max_requires_external_decision_count']}`",
                    "",
                ]
            )
        lines.extend(["## Warnings", ""])
        if warnings:
            for warning in warnings:
                lines.append(f"- {warning}")
        else:
            lines.append("- None")
        lines.append("")
        return "\n".join(lines)

    def _normalize_input_dirs(self, input_dirs: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
        if isinstance(input_dirs, (str, Path)):
            return (Path(input_dirs),)
        return tuple(Path(item) for item in input_dirs)

    def _suite_identity(self, root: Path) -> tuple[str, Path | None]:
        acceptance = root / "acceptance_report.json"
        if acceptance.exists():
            return ("acceptance", acceptance)
        regression = root / "real_world_regression_report.json"
        if regression.exists():
            return ("real_world_regression", regression)
        return (root.name, None)

    def _coerce_metrics(self, payload: object, *, case_id: str, warnings: list[str]) -> dict[str, float]:
        if not isinstance(payload, Mapping):
            warnings.append(f"{case_id}: missing_metrics")
            return {}
        metrics: dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metrics[str(key)] = float(value)
        return metrics

    def _infer_case_type(self, case_id: str) -> str:
        lowered = case_id.lower()
        for token, profile_name in PROFILE_CASE_TYPE_RULES:
            if token in lowered:
                return profile_name
        return "other"

    def _infer_geometry_type(self, case_payload: Mapping[str, Any], *, case_type: str) -> str:
        actual_geometry = case_payload.get("actual_geometry")
        if isinstance(actual_geometry, Mapping):
            non_zero = {
                str(key): int(value)
                for key, value in actual_geometry.items()
                if isinstance(value, (int, float)) and int(value) > 0
            }
            if non_zero:
                dominant = max(non_zero.items(), key=lambda item: item[1])[0]
                for token, profile_name in PROFILE_GEOMETRY_RULES:
                    if token in dominant:
                        return profile_name
        return case_type

    def _algorithm_confidence(self, sample: CalibrationSample) -> float | None:
        return sample.metrics.get("algorithm_fitting_confidence")

    def _inlier_ratio(self, sample: CalibrationSample) -> float | None:
        return sample.metrics.get("inlier_ratio")

    def _trim_outliers(self, values: Sequence[float | None], *, direction: str) -> list[float]:
        cleaned = sorted(float(value) for value in values if value is not None)
        if len(cleaned) < 4:
            return cleaned
        q1 = self._percentile(cleaned, 0.25)
        q3 = self._percentile(cleaned, 0.75)
        iqr = q3 - q1
        if iqr <= 0.0:
            return cleaned
        if direction == "higher":
            upper_bound = q3 + (1.5 * iqr)
            return [value for value in cleaned if value <= upper_bound]
        lower_bound = q1 - (1.5 * iqr)
        return [value for value in cleaned if value >= lower_bound]

    def _higher_threshold(self, values: Sequence[float], *, default: float | int) -> float:
        if not values:
            return float(default)
        if len(values) == 1:
            return float(values[0])
        return float(self._percentile(sorted(values), 0.9))

    def _lower_threshold(self, values: Sequence[float], *, default: float) -> float:
        if not values:
            return float(default)
        if len(values) == 1:
            return float(values[0])
        return float(self._percentile(sorted(values), 0.1))

    def _metric_stats(self, values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {"count": 0.0, "mean": 0.0, "max": 0.0, "p10": 0.0, "p90": 0.0}
        ordered = sorted(values)
        return {
            "count": float(len(ordered)),
            "mean": float(sum(ordered) / len(ordered)),
            "max": float(max(ordered)),
            "p10": float(self._percentile(ordered, 0.1)),
            "p90": float(self._percentile(ordered, 0.9)),
        }

    def _percentile(self, values: Sequence[float], quantile: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        index = (len(values) - 1) * quantile
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return float(values[lower])
        weight = index - lower
        return float(values[lower] + ((values[upper] - values[lower]) * weight))

    def _read_json(self, path: Path | None, *, warnings: list[str]) -> Any:
        if path is None or not path.exists():
            warnings.append(f"missing_file: {path}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"invalid_json: {path}: {exc}")
            return None


__all__ = [
    "CalibrationSample",
    "QualityMetricCalibrationReport",
    "QualityMetricCalibrator",
]
