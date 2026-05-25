from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from services.benchmark_runner import BenchmarkRunner


def _write_case_image(image_path: Path, *, shape: str) -> None:
    image = np.zeros((120, 140, 4), dtype=np.uint8)
    if shape == "logo":
        cv2.circle(image, (45, 60), 24, (255, 255, 255, 255), thickness=-1)
        cv2.circle(image, (80, 60), 14, (255, 255, 255, 255), thickness=-1)
        cv2.rectangle(image, (88, 52), (112, 68), (255, 255, 255, 255), thickness=-1)
    elif shape == "mechanical":
        cv2.rectangle(image, (20, 20), (110, 90), (255, 255, 255, 255), thickness=2)
        cv2.circle(image, (45, 55), 10, (255, 255, 255, 255), thickness=2)
        cv2.circle(image, (85, 55), 10, (255, 255, 255, 255), thickness=2)
    elif shape == "transparent":
        cv2.circle(image, (70, 60), 22, (255, 255, 255, 255), thickness=3)
        cv2.line(image, (70, 24), (70, 96), (255, 255, 255, 255), thickness=3)
        cv2.line(image, (34, 60), (106, 60), (255, 255, 255, 255), thickness=3)
    elif shape == "noisy":
        cv2.rectangle(image, (25, 25), (100, 95), (255, 255, 255, 255), thickness=2)
        noise_rng = np.random.default_rng(7)
        for x, y in noise_rng.integers(low=0, high=120, size=(140, 2)):
            image[int(y % 120), int(x % 140)] = (255, 255, 255, 255)
    else:
        raise ValueError(f"unsupported shape: {shape}")
    assert cv2.imwrite(str(image_path), image)


def test_real_world_regression_repository_samples_and_baseline_cover_required_classes() -> None:
    manifest_path = Path("benchmarks/real_samples/manifest.json")
    baseline_path = Path("benchmarks/baselines/real_world_regression_baseline.json")

    cases = BenchmarkRunner().discover_cases(manifest_path)
    baseline = BenchmarkRunner().load_regression_baseline(baseline_path)
    case_ids = {case.case_id for case in cases}

    assert {
        "logo_sample",
        "mechanical_part",
        "transparent_icon",
        "noisy_scan",
    }.issubset(case_ids)
    assert case_ids.issubset(set(baseline))
    for case in cases:
        assert Path(case.image_path).exists()
        assert "runtime_ms" in case.regression_tolerances


def test_real_world_regression_loads_baseline_and_passes_within_tolerance(tmp_path: Path) -> None:
    image_path = tmp_path / "logo.png"
    _write_case_image(image_path, shape="logo")
    manifest_path = tmp_path / "manifest.json"
    baseline_path = tmp_path / "baseline.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "logo_case",
                        "image_path": str(image_path),
                        "segment_type": "line",
                        "regression_tolerances": {
                            "total_score": 100.0,
                            "topology_error_count": 0,
                            "self_intersection_count": 0,
                            "requires_external_decision_count": 0,
                            "svg_node_count": 20,
                            "runtime_ms": 5000,
                        },
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    first_report = BenchmarkRunner().run_acceptance_manifest(manifest_path, output_dir=tmp_path / "seed")
    baseline_path.write_text(
        json.dumps(
            {
                "cases": {
                    "logo_case": {
                        "metrics": dict(first_report.cases[0].metrics),
                    }
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = BenchmarkRunner().run_real_world_regression(
        manifest_path,
        baseline_path=baseline_path,
        output_dir=tmp_path / "regression",
    )

    assert report.summary["overall_pass"] is True
    assert report.summary["failed_case_count"] == 0
    assert report.summary["regression_metrics"] == list(BenchmarkRunner.REGRESSION_METRICS)
    assert (tmp_path / "regression" / "real_world_regression_report.json").exists()


def test_real_world_regression_fails_when_metric_regresses_beyond_tolerance(tmp_path: Path) -> None:
    image_path = tmp_path / "mechanical.png"
    _write_case_image(image_path, shape="mechanical")
    manifest_path = tmp_path / "manifest.json"
    baseline_path = tmp_path / "baseline.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "mechanical_case",
                        "image_path": str(image_path),
                        "segment_type": "line",
                        "regression_tolerances": {
                            "total_score": 0.0,
                            "topology_error_count": 0,
                            "self_intersection_count": 0,
                            "requires_external_decision_count": 0,
                            "svg_node_count": 0,
                            "runtime_ms": 0,
                        },
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(
            {
                "cases": {
                    "mechanical_case": {
                        "metrics": {
                            "total_score": 0.0,
                            "topology_error_count": 0,
                            "self_intersection_count": 0,
                            "requires_external_decision_count": 0,
                            "svg_node_count": 0,
                            "runtime_ms": 0,
                        }
                    }
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = BenchmarkRunner().run_real_world_regression(
        manifest_path,
        baseline_path=baseline_path,
        output_dir=tmp_path / "regression",
    )

    assert report.summary["overall_pass"] is False
    assert report.summary["failed_case_count"] == 1
    assert "regression_detected" in report.summary["failure_reasons"]["mechanical_case"]
