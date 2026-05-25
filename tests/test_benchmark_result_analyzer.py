from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from services.benchmark_result_analyzer import BenchmarkResultAnalyzer


def _write_case(
    suite_dir: Path,
    *,
    case_id: str,
    metrics: dict[str, float | int] | None = None,
    success: bool = True,
    failure_reason: str | None = None,
    preview_decisions: list[dict[str, object]] | None = None,
    decision_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    case_dir = suite_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    if metrics is not None:
        (case_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    decision_report = {
        "case_id": case_id,
        "success": success,
        "failure_reason": failure_reason,
        "decision_counts": decision_counts or {
            "auto_apply": 0,
            "auto_reject": 0,
            "requires_external_decision": 0,
        },
        "preview_decisions": preview_decisions or [],
        "command_results": [],
    }
    (case_dir / "decision_report.json").write_text(json.dumps(decision_report, indent=2, sort_keys=True), encoding="utf-8")
    for name in ("document.json", "output.json", "output.svg", "output.dxf", "overlay.png", "diff.png"):
        payload = "{}" if name.endswith(".json") else ""
        (case_dir / name).write_text(payload, encoding="utf-8")
    return {
        "case_id": case_id,
        "success": success,
        "failure_reason": failure_reason,
        "metrics": metrics or {},
        "accepted_count": int((decision_counts or {}).get("auto_apply", 0)),
        "rejected_count": int((decision_counts or {}).get("auto_reject", 0)),
        "user_confirm_count": int((decision_counts or {}).get("requires_external_decision", 0)),
        "preview_decisions": preview_decisions or [],
        "command_results": [],
    }


def _write_suite_report(suite_dir: Path, *, suite_name: str, cases: list[dict[str, object]]) -> None:
    report_name = "acceptance_report.json" if suite_name == "acceptance" else "real_world_regression_report.json"
    payload = {
        "runner_version": 1,
        "cases": cases,
        "summary": {
            "overall_pass": all(bool(case.get("success", True)) for case in cases),
            "failed_case_count": sum(1 for case in cases if not bool(case.get("success", True))),
        },
    }
    (suite_dir / report_name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _metrics(
    *,
    total_score: float,
    edge_error: float,
    topology_error_count: int,
    self_intersection_count: int,
    requires_external_decision_count: int,
    runtime_ms: float,
    auto_reject_count: int = 0,
    svg_node_count: int = 0,
) -> dict[str, float | int]:
    return {
        "total_score": total_score,
        "edge_error": edge_error,
        "complexity_score": 0.0,
        "control_point_count": 0,
        "topology_error_count": topology_error_count,
        "self_intersection_count": self_intersection_count,
        "auto_apply_count": 0,
        "auto_reject_count": auto_reject_count,
        "requires_external_decision_count": requires_external_decision_count,
        "svg_node_count": svg_node_count,
        "runtime_ms": runtime_ms,
    }


def _preview_decision(
    *,
    tool: str,
    path_id: str,
    reason_code: str,
    decision_kind: str = "auto_reject",
) -> dict[str, object]:
    return {
        "command": {
            "tool": tool,
            "path_id": path_id,
        },
        "decision_kind": decision_kind,
        "policy_feedback": {
            "reason_code": reason_code,
        },
    }


def test_benchmark_result_analyzer_aggregates_metrics_and_worst_case_rankings(tmp_path: Path) -> None:
    acceptance_dir = tmp_path / "acceptance"
    regression_dir = tmp_path / "regression"
    acceptance_dir.mkdir()
    regression_dir.mkdir()

    acceptance_cases = [
        _write_case(
            acceptance_dir,
            case_id="circle_auto_apply",
            metrics=_metrics(
                total_score=12.0,
                edge_error=1.2,
                topology_error_count=0,
                self_intersection_count=0,
                requires_external_decision_count=0,
                runtime_ms=25.0,
                svg_node_count=8,
            ),
            success=True,
        ),
        _write_case(
            acceptance_dir,
            case_id="noisy_scan_proxy",
            metrics=_metrics(
                total_score=98.0,
                edge_error=8.0,
                topology_error_count=1,
                self_intersection_count=2,
                requires_external_decision_count=3,
                runtime_ms=120.0,
                auto_reject_count=4,
                svg_node_count=48,
            ),
            success=False,
            failure_reason="max_total_score exceeded: 98.0 > 80.0",
            preview_decisions=[
                _preview_decision(
                    tool="propose_replace_path_with_circle",
                    path_id="path_1",
                    reason_code="low_inlier_ratio",
                ),
                _preview_decision(
                    tool="propose_replace_path_with_circle",
                    path_id="path_1",
                    reason_code="low_inlier_ratio",
                ),
            ],
            decision_counts={
                "auto_apply": 0,
                "auto_reject": 4,
                "requires_external_decision": 3,
            },
        ),
    ]
    regression_cases = [
        _write_case(
            regression_dir,
            case_id="mechanical_part",
            metrics=_metrics(
                total_score=45.0,
                edge_error=7.5,
                topology_error_count=0,
                self_intersection_count=0,
                requires_external_decision_count=1,
                runtime_ms=140.0,
                auto_reject_count=2,
                svg_node_count=30,
            ),
            success=False,
            failure_reason="regression_detected: total_score 45 > 30 + tolerance 2",
            preview_decisions=[
                _preview_decision(
                    tool="propose_replace_segment_with_arc",
                    path_id="path_9",
                    reason_code="path_retry_budget_exceeded",
                ),
            ],
            decision_counts={
                "auto_apply": 0,
                "auto_reject": 2,
                "requires_external_decision": 1,
            },
        ),
    ]

    _write_suite_report(acceptance_dir, suite_name="acceptance", cases=acceptance_cases)
    _write_suite_report(regression_dir, suite_name="real_world_regression", cases=regression_cases)

    output_dir = tmp_path / "analysis"
    report = BenchmarkResultAnalyzer().analyze((acceptance_dir, regression_dir), output_dir=output_dir, top_k=2)

    assert report.summary["total_case_count"] == 3
    assert report.summary["failed_case_count"] == 2
    assert report.summary["failure_reason_counts"]["max_total_score exceeded"] == 1
    assert report.summary["failure_reason_counts"]["regression_detected"] == 1
    assert report.summary["repeated_ai_failure_case_count"] == 1
    assert report.summary["retry_budget_exhausted_case_count"] == 1
    assert report.summary["case_type_counts"]["circle"] == 1
    assert report.summary["case_type_counts"]["noisy_scan"] == 1
    assert report.summary["case_type_counts"]["mechanical"] == 1

    assert report.worst_cases["total_score"][0]["case_id"] == "noisy_scan_proxy"
    assert report.worst_cases["total_score"][0]["metric_missing"] is False
    assert report.worst_cases["total_score"][0]["rank_reason"] == "metric_value"
    assert report.worst_cases["runtime_ms"][0]["case_id"] == "mechanical_part"
    assert report.case_type_aggregates["noisy_scan"]["failed_case_count"] == 1
    assert report.case_type_aggregates["noisy_scan"]["auto_reject_count"] == 4

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "worst_cases.json").exists()
    assert (output_dir / "benchmark_analysis.md").exists()
    markdown = (output_dir / "benchmark_analysis.md").read_text(encoding="utf-8")
    assert "Worst Cases" in markdown
    assert "noisy_scan_proxy" in markdown
    assert "mechanical_part" in markdown


def test_benchmark_result_analyzer_tolerates_missing_files_with_warnings_and_report_fallback(tmp_path: Path) -> None:
    suite_dir = tmp_path / "acceptance"
    suite_dir.mkdir()
    (suite_dir / "fallback_case").mkdir()

    suite_cases = [
        {
            "case_id": "fallback_case",
            "success": False,
            "failure_reason": "case_timeout_exceeded: fallback_case",
            "metrics": _metrics(
                total_score=0.0,
                edge_error=0.0,
                topology_error_count=0,
                self_intersection_count=0,
                requires_external_decision_count=0,
                runtime_ms=5000.0,
            ),
            "accepted_count": 0,
            "rejected_count": 0,
            "user_confirm_count": 0,
            "preview_decisions": [],
            "command_results": [],
        }
    ]
    _write_suite_report(suite_dir, suite_name="acceptance", cases=suite_cases)

    report = BenchmarkResultAnalyzer().analyze(suite_dir)

    assert report.summary["warning_count"] >= 1
    assert any("fallback_case" in warning for warning in report.warnings)
    assert report.cases[0].case_id == "fallback_case"
    assert report.cases[0].success is False
    assert report.cases[0].metrics["runtime_ms"] == 5000.0


def test_benchmark_result_analyzer_ranks_failed_case_with_missing_metric_at_front(tmp_path: Path) -> None:
    suite_dir = tmp_path / "acceptance"
    suite_dir.mkdir()

    suite_cases = [
        _write_case(
            suite_dir,
            case_id="case_good",
            metrics=_metrics(
                total_score=5.5,
                edge_error=1.0,
                topology_error_count=0,
                self_intersection_count=0,
                requires_external_decision_count=0,
                runtime_ms=10.0,
            ),
            success=True,
        ),
        _write_case(
            suite_dir,
            case_id="case_bad_high_score",
            metrics=_metrics(
                total_score=99.9,
                edge_error=9.0,
                topology_error_count=0,
                self_intersection_count=0,
                requires_external_decision_count=0,
                runtime_ms=15.0,
            ),
            success=True,
        ),
        _write_case(
            suite_dir,
            case_id="case_crashed_missing_metrics",
            metrics=None,
            success=False,
            failure_reason="pipeline_crashed",
        ),
    ]
    _write_suite_report(suite_dir, suite_name="acceptance", cases=suite_cases)

    report = BenchmarkResultAnalyzer().analyze(suite_dir, top_k=2)

    entries = report.worst_cases["total_score"]
    assert entries[0]["case_id"] == "case_crashed_missing_metrics"
    assert entries[0]["metric_missing"] is True
    assert entries[0]["metric_value"] is None
    assert entries[0]["rank_reason"] == "missing_metric_failed_case"
    assert {entry["case_id"] for entry in entries} == {
        "case_crashed_missing_metrics",
        "case_bad_high_score",
    }


def test_benchmark_result_analyzer_marks_success_case_missing_metric_without_coercing_zero(tmp_path: Path) -> None:
    suite_dir = tmp_path / "acceptance"
    suite_dir.mkdir()

    partial_metrics = _metrics(
        total_score=4.0,
        edge_error=3.0,
        topology_error_count=0,
        self_intersection_count=0,
        requires_external_decision_count=0,
        runtime_ms=20.0,
    )
    del partial_metrics["edge_error"]

    suite_cases = [
        _write_case(
            suite_dir,
            case_id="case_success_missing_edge_error",
            metrics=partial_metrics,
            success=True,
        ),
        _write_case(
            suite_dir,
            case_id="case_normal",
            metrics=_metrics(
                total_score=3.0,
                edge_error=7.0,
                topology_error_count=0,
                self_intersection_count=0,
                requires_external_decision_count=0,
                runtime_ms=10.0,
            ),
            success=True,
        ),
    ]
    _write_suite_report(suite_dir, suite_name="acceptance", cases=suite_cases)

    report = BenchmarkResultAnalyzer().analyze(suite_dir, top_k=2)

    entry = next(item for item in report.worst_cases["edge_error"] if item["case_id"] == "case_success_missing_edge_error")
    assert entry["metric_missing"] is True
    assert entry["metric_value"] is None
    assert entry["rank_reason"] == "missing_metric_success_case"
    assert any("case_success_missing_edge_error: missing_metric: edge_error" == warning for warning in report.warnings)


def test_benchmark_result_analyzer_preserves_higher_is_worse_metric_ordering(tmp_path: Path) -> None:
    suite_dir = tmp_path / "regression"
    suite_dir.mkdir()

    suite_cases = [
        _write_case(
            suite_dir,
            case_id="case_low",
            metrics=_metrics(
                total_score=1.0,
                edge_error=1.0,
                topology_error_count=0,
                self_intersection_count=0,
                requires_external_decision_count=0,
                runtime_ms=5.0,
            ),
            success=True,
        ),
        _write_case(
            suite_dir,
            case_id="case_high",
            metrics=_metrics(
                total_score=10.0,
                edge_error=2.0,
                topology_error_count=0,
                self_intersection_count=0,
                requires_external_decision_count=0,
                runtime_ms=7.0,
            ),
            success=True,
        ),
    ]
    _write_suite_report(suite_dir, suite_name="real_world_regression", cases=suite_cases)

    report = BenchmarkResultAnalyzer().analyze(suite_dir, top_k=2)

    entries = report.worst_cases["total_score"]
    assert [entry["case_id"] for entry in entries] == ["case_high", "case_low"]
    assert all(entry["rank_reason"] == "metric_value" for entry in entries)


def test_benchmark_result_analyzer_cli_writes_outputs_and_returns_nonzero_on_failure(tmp_path: Path) -> None:
    suite_dir = tmp_path / "regression"
    suite_dir.mkdir()
    suite_cases = [
        _write_case(
            suite_dir,
            case_id="transparent_icon",
            metrics=_metrics(
                total_score=15.0,
                edge_error=2.0,
                topology_error_count=0,
                self_intersection_count=0,
                requires_external_decision_count=0,
                runtime_ms=30.0,
            ),
            success=False,
            failure_reason="regression_detected: total_score 15 > 10 + tolerance 1",
        ),
    ]
    _write_suite_report(suite_dir, suite_name="real_world_regression", cases=suite_cases)

    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_benchmark_results.py",
            "--input-dir",
            str(suite_dir),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "3",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "worst_cases.json").exists()
    assert (output_dir / "benchmark_analysis.md").exists()
