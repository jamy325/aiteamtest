from __future__ import annotations

import json
from pathlib import Path

from services.benchmark_runner import BenchmarkCase, BenchmarkRunner
from services.preview_auto_accept_policy import PreviewAndAutoAcceptPolicyConfig
from services.quality_metric_calibration import QualityMetricCalibrator


def _suite_report_path(root: Path, *, suite_name: str) -> Path:
    if suite_name == "acceptance":
        return root / "acceptance_report.json"
    return root / "real_world_regression_report.json"


def _case_payload(
    *,
    case_id: str,
    success: bool,
    total_score: float,
    edge_error: float,
    complexity_score: float,
    requires_external_decision_count: int,
    algorithm_fitting_confidence: float,
    inlier_ratio: float,
    actual_geometry: dict[str, int],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "success": success,
        "failure_reason": None if success else "threshold_exceeded",
        "metrics": {
            "total_score": total_score,
            "edge_error": edge_error,
            "complexity_score": complexity_score,
            "requires_external_decision_count": requires_external_decision_count,
            "algorithm_fitting_confidence": algorithm_fitting_confidence,
            "inlier_ratio": inlier_ratio,
        },
        "actual_geometry": actual_geometry,
        "accepted_count": 0,
        "rejected_count": 0 if success else 1,
        "user_confirm_count": requires_external_decision_count,
    }


def _write_suite(root: Path, *, suite_name: str, cases: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = _suite_report_path(root, suite_name=suite_name)
    path.write_text(json.dumps({"cases": cases, "summary": {"total_cases": len(cases)}}, indent=2), encoding="utf-8")
    return path


def test_quality_metric_calibration_generates_profiles_and_report(tmp_path: Path) -> None:
    acceptance_dir = tmp_path / "acceptance"
    regression_dir = tmp_path / "regression"
    _write_suite(
        acceptance_dir,
        suite_name="acceptance",
        cases=[
            _case_payload(
                case_id="circle_auto_apply",
                success=True,
                total_score=10.0,
                edge_error=1.0,
                complexity_score=2.0,
                requires_external_decision_count=0,
                algorithm_fitting_confidence=0.9,
                inlier_ratio=0.88,
                actual_geometry={"circle": 1},
            ),
            _case_payload(
                case_id="ellipse_auto_apply",
                success=True,
                total_score=12.0,
                edge_error=1.2,
                complexity_score=2.5,
                requires_external_decision_count=1,
                algorithm_fitting_confidence=0.86,
                inlier_ratio=0.84,
                actual_geometry={"ellipse": 1},
            ),
            _case_payload(
                case_id="line_arc_outline",
                success=True,
                total_score=7.0,
                edge_error=0.8,
                complexity_score=1.5,
                requires_external_decision_count=0,
                algorithm_fitting_confidence=0.8,
                inlier_ratio=0.9,
                actual_geometry={"line": 2, "arc": 1},
            ),
        ],
    )
    _write_suite(
        regression_dir,
        suite_name="real_world_regression",
        cases=[
            _case_payload(
                case_id="logo_sample",
                success=True,
                total_score=20.0,
                edge_error=2.2,
                complexity_score=5.0,
                requires_external_decision_count=1,
                algorithm_fitting_confidence=0.72,
                inlier_ratio=0.7,
                actual_geometry={"bezier": 4},
            ),
            _case_payload(
                case_id="mechanical_part",
                success=True,
                total_score=16.0,
                edge_error=1.5,
                complexity_score=3.0,
                requires_external_decision_count=0,
                algorithm_fitting_confidence=0.78,
                inlier_ratio=0.82,
                actual_geometry={"line": 6, "arc": 2},
            ),
            _case_payload(
                case_id="transparent_icon",
                success=True,
                total_score=18.0,
                edge_error=1.7,
                complexity_score=4.0,
                requires_external_decision_count=1,
                algorithm_fitting_confidence=0.74,
                inlier_ratio=0.76,
                actual_geometry={"circle": 2},
            ),
            _case_payload(
                case_id="noisy_scan",
                success=False,
                total_score=45.0,
                edge_error=5.0,
                complexity_score=8.0,
                requires_external_decision_count=3,
                algorithm_fitting_confidence=0.5,
                inlier_ratio=0.5,
                actual_geometry={"line": 3},
            ),
        ],
    )

    output_json = tmp_path / "profiles.json"
    report_md = tmp_path / "calibration_report.md"
    report = QualityMetricCalibrator().calibrate(
        (acceptance_dir, regression_dir),
        output_json=output_json,
        report_path=report_md,
    )

    assert output_json.exists()
    assert report_md.exists()
    assert report.summary["total_sample_count"] == 7
    assert report.summary["profile_count"] >= 8
    assert {"circle", "ellipse", "line_arc", "bezier_fallback", "logo", "mechanical", "transparent", "noisy_scan"}.issubset(
        set(report.profiles)
    )
    circle_thresholds = report.profiles["circle"]["recommended_thresholds"]
    assert set(circle_thresholds) == {
        "max_total_score",
        "max_edge_error",
        "max_complexity_score",
        "min_algorithm_confidence",
        "min_inlier_ratio",
        "max_requires_external_decision_count",
    }
    assert "Recommended Profiles" in report.markdown


def test_quality_metric_calibration_recommends_thresholds_with_outlier_trimming(tmp_path: Path) -> None:
    suite_dir = tmp_path / "acceptance"
    cases = [
        _case_payload(
            case_id=f"circle_case_{index}",
            success=True,
            total_score=score,
            edge_error=1.0 + index,
            complexity_score=2.0 + index,
            requires_external_decision_count=0,
            algorithm_fitting_confidence=confidence,
            inlier_ratio=inlier_ratio,
            actual_geometry={"circle": 1},
        )
        for index, (score, confidence, inlier_ratio) in enumerate(
            (
                (10.0, 0.9, 0.9),
                (11.0, 0.88, 0.87),
                (12.0, 0.86, 0.85),
                (200.0, 0.1, 0.2),
            ),
            start=1,
        )
    ]
    _write_suite(suite_dir, suite_name="acceptance", cases=cases)

    report = QualityMetricCalibrator().calibrate(suite_dir, dry_run=True)
    thresholds = report.profiles["circle"]["recommended_thresholds"]

    assert thresholds["max_total_score"] < 200.0
    assert thresholds["min_algorithm_confidence"] > 0.1
    assert thresholds["min_inlier_ratio"] > 0.2


def test_quality_metric_calibration_dry_run_does_not_overwrite_json(tmp_path: Path) -> None:
    suite_dir = tmp_path / "acceptance"
    _write_suite(
        suite_dir,
        suite_name="acceptance",
        cases=[
            _case_payload(
                case_id="circle_auto_apply",
                success=True,
                total_score=10.0,
                edge_error=1.0,
                complexity_score=2.0,
                requires_external_decision_count=0,
                algorithm_fitting_confidence=0.9,
                inlier_ratio=0.88,
                actual_geometry={"circle": 1},
            ),
        ],
    )
    output_json = tmp_path / "profiles.json"
    report_md = tmp_path / "report.md"

    report = QualityMetricCalibrator().calibrate(
        suite_dir,
        output_json=output_json,
        report_path=report_md,
        dry_run=True,
    )

    assert not output_json.exists()
    assert report_md.exists()
    assert report.summary["dry_run"] is True


def test_quality_metric_calibration_profile_injection_helpers(tmp_path: Path) -> None:
    suite_dir = tmp_path / "acceptance"
    _write_suite(
        suite_dir,
        suite_name="acceptance",
        cases=[
            _case_payload(
                case_id="circle_auto_apply",
                success=True,
                total_score=14.0,
                edge_error=1.4,
                complexity_score=2.4,
                requires_external_decision_count=1,
                algorithm_fitting_confidence=0.82,
                inlier_ratio=0.79,
                actual_geometry={"circle": 1},
            ),
        ],
    )
    profiles = QualityMetricCalibrator().calibrate(suite_dir, dry_run=True).profiles

    config = PreviewAndAutoAcceptPolicyConfig.from_quality_profile(profiles["circle"])
    assert config.min_fitting_confidence == profiles["circle"]["recommended_thresholds"]["min_algorithm_confidence"]
    assert config.min_inlier_ratio == profiles["circle"]["recommended_thresholds"]["min_inlier_ratio"]

    runner = BenchmarkRunner()
    case = BenchmarkCase(
        case_id="circle_auto_apply",
        image_path="circle.png",
        quality_profile="circle",
    )
    calibrated_case = runner.apply_quality_profile(case, profiles)
    assert calibrated_case.fail_thresholds["max_total_score"] == profiles["circle"]["recommended_thresholds"]["max_total_score"]
    assert calibrated_case.fail_thresholds["max_edge_error"] == profiles["circle"]["recommended_thresholds"]["max_edge_error"]
    assert calibrated_case.fail_thresholds["max_complexity_score"] == profiles["circle"]["recommended_thresholds"]["max_complexity_score"]
    assert calibrated_case.fail_thresholds["max_requires_external_decision_count"] == profiles["circle"]["recommended_thresholds"]["max_requires_external_decision_count"]


def test_benchmark_runner_quality_profile_preserves_manual_fail_threshold_overrides() -> None:
    profiles = {
        "circle": {
            "recommended_thresholds": {
                "max_total_score": 10.0,
                "max_edge_error": 1.0,
                "max_complexity_score": 2.0,
                "max_requires_external_decision_count": 0,
            }
        }
    }
    case = BenchmarkCase(
        case_id="manual_override_case",
        image_path="dummy.png",
        quality_profile="circle",
        fail_thresholds={
            "max_total_score": 999.0,
            "max_edge_error": 99.0,
        },
    )

    calibrated = BenchmarkRunner().apply_quality_profile(case, profiles)

    assert calibrated.fail_thresholds["max_total_score"] == 999.0
    assert calibrated.fail_thresholds["max_edge_error"] == 99.0
    assert calibrated.fail_thresholds["max_complexity_score"] == 2.0
    assert calibrated.fail_thresholds["max_requires_external_decision_count"] == 0
