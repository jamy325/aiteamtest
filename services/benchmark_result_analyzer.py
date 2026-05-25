from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WORST_CASE_METRICS: tuple[str, ...] = (
    "total_score",
    "edge_error",
    "topology_error_count",
    "self_intersection_count",
    "requires_external_decision_count",
    "runtime_ms",
    "auto_reject_count",
    "svg_node_count",
)

WORST_CASE_METRIC_POLICIES: dict[str, dict[str, str]] = {
    metric_name: {"direction": "higher_is_worse"}
    for metric_name in WORST_CASE_METRICS
}

ARTIFACT_FILENAMES: tuple[str, ...] = (
    "document.json",
    "output.json",
    "output.svg",
    "output.dxf",
    "overlay.png",
    "diff.png",
    "decision_report.json",
    "metrics.json",
)

RETRY_BUDGET_REASON_CODES = {
    "retry_budget_exceeded",
    "path_retry_budget_exceeded",
}

CASE_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("concentric", "circle"),
    ("circle", "circle"),
    ("ellipse", "ellipse"),
    ("line_arc", "line_arc"),
    ("logo", "logo"),
    ("mechanical", "mechanical"),
    ("transparent", "transparent"),
    ("noisy_scan", "noisy_scan"),
    ("hole", "hole"),
    ("self_intersection", "self_intersection"),
    ("broken_contour", "broken_contour"),
    ("dxf_unit", "dxf_unit"),
    ("bezier", "bezier"),
    ("rectangle", "rectangle"),
    ("square", "rectangle"),
)


@dataclass(frozen=True, slots=True)
class AnalyzedBenchmarkCase:
    case_id: str
    suite_name: str
    case_type: str
    case_dir: str
    success: bool
    failure_reason: str | None
    metrics: dict[str, float | int]
    decision_counts: dict[str, int]
    policy_reason_codes: tuple[str, ...]
    repeated_failure_signatures: tuple[str, ...]
    retry_budget_exhausted: bool
    artifact_presence: dict[str, bool]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "suite_name": self.suite_name,
            "case_type": self.case_type,
            "case_dir": self.case_dir,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "metrics": dict(self.metrics),
            "decision_counts": dict(self.decision_counts),
            "policy_reason_codes": list(self.policy_reason_codes),
            "repeated_failure_signatures": list(self.repeated_failure_signatures),
            "retry_budget_exhausted": self.retry_budget_exhausted,
            "artifact_presence": dict(self.artifact_presence),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkAnalysisReport:
    summary: dict[str, Any]
    worst_cases: dict[str, list[dict[str, Any]]]
    case_type_aggregates: dict[str, dict[str, Any]]
    cases: tuple[AnalyzedBenchmarkCase, ...]
    warnings: tuple[str, ...]
    markdown: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": json.loads(json.dumps(self.summary)),
            "worst_cases": json.loads(json.dumps(self.worst_cases)),
            "case_type_aggregates": json.loads(json.dumps(self.case_type_aggregates)),
            "cases": [case.to_dict() for case in self.cases],
            "warnings": list(self.warnings),
        }

    def write(self, output_dir: str | Path) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "summary.json").write_text(json.dumps(self.to_dict()["summary"], indent=2, sort_keys=True), encoding="utf-8")
        (root / "worst_cases.json").write_text(json.dumps(self.worst_cases, indent=2, sort_keys=True), encoding="utf-8")
        (root / "benchmark_analysis.md").write_text(self.markdown, encoding="utf-8")


class BenchmarkResultAnalyzer:
    def analyze(
        self,
        input_dirs: str | Path | Sequence[str | Path],
        *,
        output_dir: str | Path | None = None,
        top_k: int = 10,
    ) -> BenchmarkAnalysisReport:
        roots = self._normalize_input_dirs(input_dirs)
        warnings: list[str] = []
        cases: list[AnalyzedBenchmarkCase] = []
        for root in roots:
            cases.extend(self._analyze_root(root, warnings=warnings))

        report = BenchmarkAnalysisReport(
            summary=self._build_summary(cases, warnings=warnings),
            worst_cases=self._build_worst_cases(cases, top_k=top_k),
            case_type_aggregates=self._build_case_type_aggregates(cases),
            cases=tuple(cases),
            warnings=tuple(warnings),
            markdown=self._build_markdown(cases, warnings=warnings, top_k=top_k),
        )
        if output_dir is not None:
            report.write(output_dir)
        return report

    def _normalize_input_dirs(self, input_dirs: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
        if isinstance(input_dirs, (str, Path)):
            return (Path(input_dirs),)
        return tuple(Path(item) for item in input_dirs)

    def _analyze_root(self, root: Path, *, warnings: list[str]) -> list[AnalyzedBenchmarkCase]:
        if not root.exists():
            warnings.append(f"missing_input_dir: {root}")
            return []

        suite_name, suite_report_path = self._suite_identity(root)
        suite_report = self._read_json(suite_report_path, warnings=warnings, required=False)
        report_cases: dict[str, Mapping[str, Any]] = {}
        if isinstance(suite_report, Mapping):
            raw_cases = suite_report.get("cases")
            if isinstance(raw_cases, list):
                for item in raw_cases:
                    if isinstance(item, Mapping) and item.get("case_id") is not None:
                        report_cases[str(item["case_id"])] = item

        case_ids = {path.name for path in root.iterdir() if path.is_dir()}
        case_ids.update(report_cases.keys())

        analyzed: list[AnalyzedBenchmarkCase] = []
        for case_id in sorted(case_ids):
            analyzed.append(
                self._analyze_case(
                    root / case_id,
                    suite_name=suite_name,
                    case_id=case_id,
                    report_case=report_cases.get(case_id),
                    warnings=warnings,
                )
            )
        return analyzed

    def _suite_identity(self, root: Path) -> tuple[str, Path | None]:
        acceptance = root / "acceptance_report.json"
        if acceptance.exists():
            return ("acceptance", acceptance)
        regression = root / "real_world_regression_report.json"
        if regression.exists():
            return ("real_world_regression", regression)
        return (root.name, None)

    def _analyze_case(
        self,
        case_dir: Path,
        *,
        suite_name: str,
        case_id: str,
        report_case: Mapping[str, Any] | None,
        warnings: list[str],
    ) -> AnalyzedBenchmarkCase:
        case_warnings: list[str] = []
        metrics = self._load_metrics(case_dir / "metrics.json", report_case, case_warnings)
        decision_report = self._load_decision_report(case_dir / "decision_report.json", report_case, case_warnings, case_id=case_id)
        for metric_name in WORST_CASE_METRICS:
            if metric_name not in metrics:
                case_warnings.append(f"missing_metric: {metric_name}")

        failure_reason = self._optional_string(decision_report.get("failure_reason"))
        if failure_reason is None and report_case is not None:
            failure_reason = self._optional_string(report_case.get("failure_reason"))

        success = bool(decision_report.get("success", report_case.get("success") if report_case is not None else failure_reason is None))
        decision_counts = self._decision_counts(decision_report, report_case)
        preview_decisions = decision_report.get("preview_decisions")
        repeated_failure_signatures = self._repeated_failure_signatures(preview_decisions)
        policy_reason_codes = self._policy_reason_codes(preview_decisions)
        retry_budget_exhausted = self._retry_budget_exhausted(preview_decisions, failure_reason)
        artifact_presence = {name: (case_dir / name).exists() for name in ARTIFACT_FILENAMES}

        for message in case_warnings:
            warnings.append(f"{case_id}: {message}")

        return AnalyzedBenchmarkCase(
            case_id=case_id,
            suite_name=suite_name,
            case_type=self._infer_case_type(case_id),
            case_dir=str(case_dir),
            success=success,
            failure_reason=failure_reason,
            metrics=metrics,
            decision_counts=decision_counts,
            policy_reason_codes=policy_reason_codes,
            repeated_failure_signatures=repeated_failure_signatures,
            retry_budget_exhausted=retry_budget_exhausted,
            artifact_presence=artifact_presence,
            warnings=tuple(case_warnings),
        )

    def _load_metrics(
        self,
        metrics_path: Path,
        report_case: Mapping[str, Any] | None,
        case_warnings: list[str],
    ) -> dict[str, float | int]:
        payload = self._read_json(metrics_path, warnings=case_warnings, required=True)
        if not isinstance(payload, Mapping) and report_case is not None:
            payload = report_case.get("metrics")
        if not isinstance(payload, Mapping):
            return {}
        return {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    def _load_decision_report(
        self,
        decision_report_path: Path,
        report_case: Mapping[str, Any] | None,
        case_warnings: list[str],
        *,
        case_id: str,
    ) -> dict[str, Any]:
        payload = self._read_json(decision_report_path, warnings=case_warnings, required=True)
        if isinstance(payload, Mapping):
            return dict(payload)
        fallback: dict[str, Any] = {"case_id": case_id}
        if report_case is not None:
            fallback["success"] = bool(report_case.get("success", True))
            fallback["failure_reason"] = report_case.get("failure_reason")
            fallback["preview_decisions"] = list(report_case.get("preview_decisions", ()))
            fallback["command_results"] = list(report_case.get("command_results", ()))
            fallback["decision_counts"] = {
                "auto_apply": int(report_case.get("accepted_count", 0)),
                "auto_reject": int(report_case.get("rejected_count", 0)),
                "requires_external_decision": int(report_case.get("user_confirm_count", 0)),
            }
        return fallback

    def _decision_counts(
        self,
        decision_report: Mapping[str, Any],
        report_case: Mapping[str, Any] | None,
    ) -> dict[str, int]:
        payload = decision_report.get("decision_counts")
        if isinstance(payload, Mapping):
            return {
                "auto_apply": int(payload.get("auto_apply", 0)),
                "auto_reject": int(payload.get("auto_reject", 0)),
                "requires_external_decision": int(payload.get("requires_external_decision", 0)),
            }
        if report_case is None:
            return {"auto_apply": 0, "auto_reject": 0, "requires_external_decision": 0}
        return {
            "auto_apply": int(report_case.get("accepted_count", 0)),
            "auto_reject": int(report_case.get("rejected_count", 0)),
            "requires_external_decision": int(report_case.get("user_confirm_count", 0)),
        }

    def _repeated_failure_signatures(self, preview_decisions: object) -> tuple[str, ...]:
        if not isinstance(preview_decisions, list):
            return ()
        counts: Counter[str] = Counter()
        for item in preview_decisions:
            if not isinstance(item, Mapping):
                continue
            if item.get("decision_kind") == "auto_apply":
                continue
            command = item.get("command")
            if not isinstance(command, Mapping):
                continue
            counts[self._command_signature(command)] += 1
        return tuple(sorted(signature for signature, count in counts.items() if count > 1))

    def _retry_budget_exhausted(self, preview_decisions: object, failure_reason: str | None) -> bool:
        if failure_reason is not None and "retry_budget" in failure_reason:
            return True
        if not isinstance(preview_decisions, list):
            return False
        for item in preview_decisions:
            if not isinstance(item, Mapping):
                continue
            feedback = item.get("policy_feedback")
            if not isinstance(feedback, Mapping):
                continue
            reason_code = feedback.get("reason_code")
            if isinstance(reason_code, str) and reason_code in RETRY_BUDGET_REASON_CODES:
                return True
        return False

    def _build_summary(self, cases: Sequence[AnalyzedBenchmarkCase], *, warnings: Sequence[str]) -> dict[str, Any]:
        failure_reason_counts: Counter[str] = Counter()
        policy_reason_counts: Counter[str] = Counter()
        suite_counts: Counter[str] = Counter()
        case_type_counts: Counter[str] = Counter()
        repeated_case_ids: list[str] = []
        retry_budget_case_ids: list[str] = []

        for case in cases:
            suite_counts[case.suite_name] += 1
            case_type_counts[case.case_type] += 1
            if case.failure_reason is not None:
                failure_reason_counts[self._normalize_reason(case.failure_reason)] += 1
            if case.repeated_failure_signatures:
                repeated_case_ids.append(case.case_id)
            if case.retry_budget_exhausted:
                retry_budget_case_ids.append(case.case_id)
            for reason_code in case.policy_reason_codes:
                policy_reason_counts[reason_code] += 1

        return {
            "overall_pass": all(case.success for case in cases) if cases else True,
            "total_case_count": len(cases),
            "failed_case_count": sum(1 for case in cases if not case.success),
            "suite_counts": dict(sorted(suite_counts.items())),
            "case_type_counts": dict(sorted(case_type_counts.items())),
            "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
            "policy_reason_counts": dict(sorted(policy_reason_counts.items())),
            "repeated_ai_failure_case_count": len(repeated_case_ids),
            "repeated_ai_failure_cases": sorted(repeated_case_ids),
            "retry_budget_exhausted_case_count": len(retry_budget_case_ids),
            "retry_budget_exhausted_cases": sorted(retry_budget_case_ids),
            "warning_count": len(warnings),
            "warnings": list(warnings),
        }

    def _build_case_type_aggregates(self, cases: Sequence[AnalyzedBenchmarkCase]) -> dict[str, dict[str, Any]]:
        groups: defaultdict[str, list[AnalyzedBenchmarkCase]] = defaultdict(list)
        for case in cases:
            groups[case.case_type].append(case)

        aggregates: dict[str, dict[str, Any]] = {}
        for case_type, grouped_cases in sorted(groups.items()):
            total_score = sum(float(case.metrics.get("total_score", 0.0)) for case in grouped_cases)
            auto_reject = sum(int(case.decision_counts.get("auto_reject", 0)) for case in grouped_cases)
            requires_external = sum(int(case.decision_counts.get("requires_external_decision", 0)) for case in grouped_cases)
            failure_reason_counts: Counter[str] = Counter()
            for case in grouped_cases:
                if case.failure_reason is not None:
                    failure_reason_counts[self._normalize_reason(case.failure_reason)] += 1
            aggregates[case_type] = {
                "case_count": len(grouped_cases),
                "failed_case_count": sum(1 for case in grouped_cases if not case.success),
                "average_total_score": (total_score / len(grouped_cases)) if grouped_cases else 0.0,
                "auto_reject_count": auto_reject,
                "requires_external_decision_count": requires_external,
                "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
            }
        return aggregates

    def _build_worst_cases(self, cases: Sequence[AnalyzedBenchmarkCase], *, top_k: int) -> dict[str, list[dict[str, Any]]]:
        rankings: dict[str, list[dict[str, Any]]] = {}
        for metric_name in WORST_CASE_METRICS:
            if metric_name not in WORST_CASE_METRIC_POLICIES:
                raise ValueError(f"missing worst-case metric policy: {metric_name}")
            ordered = sorted(
                cases,
                key=lambda case: self._worst_case_sort_key(case, metric_name),
            )
            rankings[metric_name] = [
                {
                    "case_id": case.case_id,
                    "suite_name": case.suite_name,
                    "case_type": case.case_type,
                    "metric_value": self._metric_value(case, metric_name),
                    "metric_missing": metric_name not in case.metrics,
                    "rank_reason": self._worst_case_rank_reason(case, metric_name),
                    "success": case.success,
                    "failure_reason": case.failure_reason,
                }
                for case in ordered[:top_k]
            ]
        return rankings

    def _build_markdown(self, cases: Sequence[AnalyzedBenchmarkCase], *, warnings: Sequence[str], top_k: int) -> str:
        summary = self._build_summary(cases, warnings=warnings)
        case_type_aggregates = self._build_case_type_aggregates(cases)
        worst_cases = self._build_worst_cases(cases, top_k=min(top_k, 5))

        lines = [
            "# Benchmark Analysis",
            "",
            "## Summary",
            "",
            f"- Overall pass: `{summary['overall_pass']}`",
            f"- Total cases: `{summary['total_case_count']}`",
            f"- Failed cases: `{summary['failed_case_count']}`",
            f"- Warnings: `{summary['warning_count']}`",
            "",
            "## Failure Reasons",
            "",
        ]
        if summary["failure_reason_counts"]:
            for reason, count in summary["failure_reason_counts"].items():
                lines.append(f"- `{reason}`: `{count}`")
        else:
            lines.append("- None")

        lines.extend(["", "## Worst Cases", ""])
        for metric_name, entries in worst_cases.items():
            lines.append(f"### {metric_name}")
            if not entries:
                lines.append("")
                lines.append("- None")
                lines.append("")
                continue
            lines.append("")
            for item in entries:
                lines.append(
                    f"- `{item['case_id']}` ({item['case_type']}, {item['suite_name']}): "
                    f"`{item['metric_value']}`"
                    f" [{item['rank_reason']}]"
                )
            lines.append("")

        lines.extend(["## Case Type Aggregates", ""])
        for case_type, aggregate in case_type_aggregates.items():
            lines.append(
                f"- `{case_type}`: cases=`{aggregate['case_count']}`, failed=`{aggregate['failed_case_count']}`, "
                f"avg_total_score=`{aggregate['average_total_score']:.3f}`, "
                f"auto_reject=`{aggregate['auto_reject_count']}`, "
                f"requires_external_decision=`{aggregate['requires_external_decision_count']}`"
            )

        lines.extend(["", "## Repeated AI Failures", ""])
        repeated_cases = [case for case in cases if case.repeated_failure_signatures or case.retry_budget_exhausted]
        if repeated_cases:
            for case in repeated_cases:
                extra = []
                if case.repeated_failure_signatures:
                    extra.append(f"repeated={list(case.repeated_failure_signatures)}")
                if case.retry_budget_exhausted:
                    extra.append("retry_budget_exhausted=true")
                lines.append(f"- `{case.case_id}`: {', '.join(extra)}")
        else:
            lines.append("- None")

        lines.extend(["", "## Warnings", ""])
        if warnings:
            for message in warnings:
                lines.append(f"- {message}")
        else:
            lines.append("- None")
        lines.append("")
        return "\n".join(lines)

    def _policy_reason_codes(self, preview_decisions: object) -> tuple[str, ...]:
        if not isinstance(preview_decisions, list):
            return ()
        reason_codes: list[str] = []
        for item in preview_decisions:
            if not isinstance(item, Mapping):
                continue
            feedback = item.get("policy_feedback")
            if not isinstance(feedback, Mapping):
                continue
            reason_code = feedback.get("reason_code")
            if isinstance(reason_code, str):
                reason_codes.append(reason_code)
        return tuple(reason_codes)

    def _infer_case_type(self, case_id: str) -> str:
        lowered = case_id.lower()
        for token, case_type in CASE_TYPE_RULES:
            if token in lowered:
                return case_type
        return "other"

    def _command_signature(self, command: Mapping[str, Any]) -> str:
        tool = str(command.get("tool", "unknown_tool"))
        path_id = command.get("path_id")
        segment_range = command.get("segment_range")
        candidate_id = command.get("candidate_id")
        if path_id is not None and isinstance(segment_range, list):
            target = f"{path_id}:{segment_range}"
        elif path_id is not None:
            target = str(path_id)
        elif candidate_id is not None:
            target = str(candidate_id)
        else:
            target = "unknown_target"
        return f"{tool}:{target}"

    def _normalize_reason(self, reason: str) -> str:
        return reason.split(":", 1)[0].strip()

    def _metric_value(self, case: AnalyzedBenchmarkCase, metric_name: str) -> float | int | None:
        return case.metrics.get(metric_name)

    def _worst_case_rank_reason(self, case: AnalyzedBenchmarkCase, metric_name: str) -> str:
        if metric_name in case.metrics:
            return "metric_value"
        if not case.success:
            return "missing_metric_failed_case"
        return "missing_metric_success_case"

    def _worst_case_sort_key(self, case: AnalyzedBenchmarkCase, metric_name: str) -> tuple[int, float]:
        direction = WORST_CASE_METRIC_POLICIES[metric_name]["direction"]
        metric_value = self._metric_value(case, metric_name)
        if metric_value is None:
            if not case.success:
                return (0, 0.0)
            return (2, 0.0)
        if direction != "higher_is_worse":
            raise ValueError(f"unsupported worst-case direction: {direction}")
        return (1, -float(metric_value))

    def _optional_string(self, value: object) -> str | None:
        if value is None:
            return None
        return str(value)

    def _read_json(self, path: Path | None, *, warnings: list[str], required: bool) -> Any:
        if path is None:
            return None
        if not path.exists():
            if required:
                warnings.append(f"missing_file: {path}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"invalid_json: {path}: {exc}")
            return None


__all__ = [
    "AnalyzedBenchmarkCase",
    "BenchmarkAnalysisReport",
    "BenchmarkResultAnalyzer",
]
