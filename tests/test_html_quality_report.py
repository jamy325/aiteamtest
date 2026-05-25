from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from services.html_quality_report import HtmlQualityReportGenerator


_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc`\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_case_bundle(root: Path, *, case_id: str, include_diff: bool = True) -> Path:
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "metrics.json").write_text(
        json.dumps(
            {
                "total_score": 12.5,
                "edge_error": 1.2,
                "complexity_score": 2.3,
                "requires_external_decision_count": 1,
                "runtime_ms": 45.0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (case_dir / "decision_report.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "success": False,
                "failure_reason": "max_total_score exceeded",
                "decision_counts": {
                    "auto_apply": 1,
                    "auto_reject": 1,
                    "requires_external_decision": 1,
                },
                "preview_decisions": [
                    {
                        "decision_kind": "auto_apply",
                        "reason": "good",
                        "command": {"tool": "propose_replace_path_with_circle"},
                        "policy_feedback": {"reason_code": "auto_apply"},
                    },
                    {
                        "decision_kind": "auto_reject",
                        "reason": "bad",
                        "command": {"tool": "propose_replace_segment_with_arc"},
                        "policy_feedback": {"reason_code": "low_inlier_ratio"},
                    },
                    {
                        "decision_kind": "requires_external_decision",
                        "reason": "review",
                        "command": {"tool": "propose_replace_path_with_bezier"},
                        "policy_feedback": {"reason_code": "requires_external_decision"},
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (case_dir / "output.json").write_text(
        json.dumps(
            {
                "document_id": case_id,
                "width": 120,
                "height": 80,
                "paths": [{"path_id": "path_1"}],
                "segments": [{"segment_id": "seg_1"}],
                "constraints": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (case_dir / "output.svg").write_text(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 10 10\"><circle cx=\"5\" cy=\"5\" r=\"3\" fill=\"none\" stroke=\"black\"/></svg>",
        encoding="utf-8",
    )
    (case_dir / "overlay.png").write_bytes(_MINIMAL_PNG)
    if include_diff:
        (case_dir / "diff.png").write_bytes(_MINIMAL_PNG)
    return case_dir


def test_html_quality_report_generates_single_case_report(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    image_path = tmp_path / "input.png"
    image_path.write_bytes(_MINIMAL_PNG)
    case_dir = _write_case_bundle(suite_dir, case_id="circle_case")
    (suite_dir / "acceptance_report.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "circle_case",
                        "image_path": str(image_path),
                        "success": False,
                        "failure_reason": "max_total_score exceeded",
                        "metrics": {"total_score": 12.5},
                    }
                ],
                "summary": {"overall_pass": False, "failed_case_count": 1},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = HtmlQualityReportGenerator().generate_case_report(case_dir)
    html = Path(result.report_path).read_text(encoding="utf-8")

    assert Path(result.report_path).exists()
    assert "circle_case Quality Report" in html
    assert "Input Image" in html
    assert "Overlay" in html
    assert "Diff" in html
    assert "SVG Preview" in html
    assert "Decision Timeline" in html
    assert "auto_apply" in html
    assert "auto_reject" in html
    assert "requires_external_decision" in html
    assert "max_total_score exceeded" in html


def test_html_quality_report_generates_suite_index_and_case_links(tmp_path: Path) -> None:
    suite_dir = tmp_path / "acceptance"
    image_path = tmp_path / "input.png"
    image_path.write_bytes(_MINIMAL_PNG)
    _write_case_bundle(suite_dir, case_id="case_a")
    _write_case_bundle(suite_dir, case_id="case_b")
    (suite_dir / "acceptance_report.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case_a",
                        "image_path": str(image_path),
                        "success": True,
                        "failure_reason": None,
                        "metrics": {"total_score": 3.0},
                    },
                    {
                        "case_id": "case_b",
                        "image_path": str(image_path),
                        "success": False,
                        "failure_reason": "bad_case",
                        "metrics": {"total_score": 9.0},
                    },
                ],
                "summary": {"overall_pass": False, "failed_case_count": 1},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = HtmlQualityReportGenerator().generate_suite_report(suite_dir)
    html = Path(result.index_path).read_text(encoding="utf-8")

    assert Path(result.index_path).exists()
    assert len(result.case_reports) == 2
    assert "Benchmark Quality Report Index" in html
    assert "case_a/report.html" in html
    assert "case_b/report.html" in html
    assert "bad_case" in html


def test_html_quality_report_handles_missing_artifacts_with_warnings(tmp_path: Path) -> None:
    suite_dir = tmp_path / "acceptance"
    case_dir = _write_case_bundle(suite_dir, case_id="missing_diff", include_diff=False)
    result = HtmlQualityReportGenerator().generate_case_report(case_dir)
    html = Path(result.report_path).read_text(encoding="utf-8")

    assert any("missing_suite_report" in warning for warning in result.warnings)
    assert "Diff unavailable" in html
    assert "Warnings" in html


def test_html_quality_report_cli_generates_expected_output_file(tmp_path: Path) -> None:
    suite_dir = tmp_path / "acceptance"
    image_path = tmp_path / "input.png"
    image_path.write_bytes(_MINIMAL_PNG)
    _write_case_bundle(suite_dir, case_id="cli_case")
    (suite_dir / "acceptance_report.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "cli_case",
                        "image_path": str(image_path),
                        "success": True,
                        "failure_reason": None,
                        "metrics": {"total_score": 1.0},
                    }
                ],
                "summary": {"overall_pass": True, "failed_case_count": 0},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "index.html"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_quality_report.py",
            "--input-dir",
            str(suite_dir),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert output_path.exists()
    assert "index.html" in completed.stdout
