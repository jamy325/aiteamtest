from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.auto_refinement_pipeline import AutoRefinementPipelineResult, AutoRefinementReport
from services.benchmark_runner import BenchmarkCase, BenchmarkRunner
from services.command_preview import CommandPreviewResult, ConstraintChangeSummary, ExportImpactSummary
from services.contour_extractor import BinaryContour, ExtractedContours
from services.minimal_pipeline import MinimalPipelineResult
from services.preview_auto_accept_policy import PreviewDecision
from services.engine_protocol import DecisionKind


def _write_case_image(image_path: Path, *, shape: str) -> None:
    image = np.zeros((120, 140), dtype=np.uint8)
    if shape == "rectangle":
        cv2.rectangle(image, (15, 15), (90, 95), 255, thickness=-1)
    elif shape == "circle":
        cv2.circle(image, (60, 60), 28, 255, thickness=2)
    else:
        cv2.line(image, (20, 30), (120, 85), 255, thickness=2)
    assert cv2.imwrite(str(image_path), image)


def _preview_result(command_id: str) -> CommandPreviewResult:
    return CommandPreviewResult(
        success=True,
        command_id=command_id,
        reason=None,
        old_score=10.0,
        predicted_new_score=8.0,
        score_delta=-2.0,
        affected_paths=("path_1",),
        affected_segments=("seg_1",),
        topology_status_before={"path_1": "closed"},
        topology_status_after={"path_1": "closed"},
        self_intersection_count_before={"path_1": 0},
        self_intersection_count_after={"path_1": 0},
        segment_type_summary={"before": {"polyline": 1}, "after": {"circle": 1}, "delta": {"polyline": -1, "circle": 1}},
        constraint_change_summary=ConstraintChangeSummary(
            before={},
            after={},
            delta={},
            added_constraint_ids=(),
            removed_constraint_ids=(),
            changed_constraint_ids=(),
        ),
        export_impact_summary=ExportImpactSummary(before={"json_char_count": 8}, after={"json_char_count": 9}, delta={"json_char_count": 1}),
    )


def _fake_pipeline_result(source_image: np.ndarray):
    document = create_document(
        document_id="doc_before",
        width=float(source_image.shape[1]),
        height=float(source_image.shape[0]),
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_1", closed=True, segments=("seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="seg_1",
            path_id="path_1",
            type="polyline",
            params={"points": [[10.0, 10.0], [40.0, 60.0], [70.0, 10.0], [10.0, 10.0]]},
        ),
    )
    contours = ExtractedContours(
        binary_contours=(
            BinaryContour(
                contour_id="binary_1",
                source="binary_contour",
                points=((10.0, 10.0), (40.0, 60.0), (70.0, 10.0)),
                coordinate_space="vector",
                closed=True,
                area=1.0,
                depth=0,
                parent_contour=None,
                children=(),
            ),
        ),
        skeleton_contours=(),
    )
    return MinimalPipelineResult(
        document=document,
        json_payload="{}",
        extracted_contours=contours,
        source_image=source_image,
    )


class _FakePipeline:
    def __init__(self, source_image: np.ndarray) -> None:
        self._result = _fake_pipeline_result(source_image)

    def run_from_file(self, image_path: str | Path, *, document_id: str = "document_1") -> MinimalPipelineResult:
        return self._result

    def export_overlay(self, document, image, output_path) -> bytes:
        canvas = np.zeros((8, 8, 3), dtype=np.uint8)
        canvas[:, :, 1] = 255
        success, encoded = cv2.imencode(".png", canvas)
        assert success
        Path(output_path).write_bytes(bytes(encoded))
        return bytes(encoded)

    def export_distance_field_diff(self, document, output_path) -> bytes:
        canvas = np.zeros((8, 8, 3), dtype=np.uint8)
        canvas[:, :, 2] = 255
        success, encoded = cv2.imencode(".png", canvas)
        assert success
        Path(output_path).write_bytes(bytes(encoded))
        return bytes(encoded)


class _FakeAutoRefinementPipeline:
    def __init__(self) -> None:
        refined_document = create_document(
            document_id="doc_after",
            width=140.0,
            height=120.0,
            coordinate_system=CoordinateSystem(internal_space="vector"),
        )
        refined_document = add_path(refined_document, VectorPath(path_id="path_1", closed=True, segments=("seg_1",)))
        refined_document = add_segment(
            refined_document,
            Segment(
                segment_id="seg_1",
                path_id="path_1",
                type="circle",
                params={"cx": 50.0, "cy": 50.0, "r": 20.0},
            ),
        )
        self._result = AutoRefinementPipelineResult(
            refined_document=refined_document,
            candidates=(),
            proposed_commands=(
                {
                    "command_id": "cmd_auto_1",
                    "tool": "propose_replace_path_with_circle",
                    "path_id": "path_1",
                    "reason": "intent only",
                    "confidence": 0.9,
                    "requires_user_confirmation": True,
                },
            ),
            preview_decisions=(
                PreviewDecision(
                    command={
                        "command_id": "cmd_auto_1",
                        "tool": "propose_replace_path_with_circle",
                        "path_id": "path_1",
                        "reason": "intent only",
                        "confidence": 0.9,
                        "requires_user_confirmation": True,
                    },
                    preview_result=_preview_result("cmd_auto_1"),
                    decision_kind=DecisionKind.AUTO_APPLY,
                    reason="safe",
                    risk_flags=(),
                ),
            ),
            report=AutoRefinementReport(
                candidate_stats={"total": 1, "by_target_type": {"circle": 1}, "by_source": {"raw_contour_points": 1}},
                command_stats={"total": 1, "evaluated_total": 1, "by_tool": {"propose_replace_path_with_circle": 1}, "batch_count": 0},
                decision_stats={
                    "auto_accept": 1,
                    "user_confirm": 0,
                    "reject": 0,
                    "auto_apply": 1,
                    "requires_external_decision": 0,
                    "auto_reject": 0,
                },
                score_before=12.0,
                score_after=8.0,
                integrity={"success": True, "error_count": 0, "warning_count": 0, "affected_ids": [], "errors": [], "warnings": []},
                dry_run_only=False,
                target_types=("circle",),
            ),
        )

    def run_from_pipeline_result(self, pipeline_result: MinimalPipelineResult, *, target_types=None, dry_run_only=None) -> AutoRefinementPipelineResult:
        return self._result


def test_acceptance_benchmark_repository_manifest_discovers_required_cases() -> None:
    cases = BenchmarkRunner().discover_cases(Path("benchmarks/acceptance_manifest.json"))
    case_ids = {case.case_id for case in cases}

    assert len(cases) >= 15
    assert {
        "circle_auto_apply",
        "ellipse_auto_apply",
        "line_arc_outline",
        "transparent_png_proxy",
        "dxf_unit_guard",
        "bezier_fallback_guard",
    }.issubset(case_ids)


def test_acceptance_benchmark_writes_required_artifacts_and_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "case.png"
    _write_case_image(image_path, shape="circle")
    source_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    assert source_image is not None

    runner = BenchmarkRunner(
        pipeline_factory=lambda case: _FakePipeline(source_image),
        auto_refinement_pipeline=_FakeAutoRefinementPipeline(),
    )
    result = runner.run_acceptance_case(
        BenchmarkCase(
            case_id="acceptance_case",
            image_path=str(image_path),
            auto_refine=True,
            auto_refine_target_types=("circle",),
            expected_geometry={"circle": 1},
        ),
        output_dir=tmp_path / "out",
    )

    case_dir = tmp_path / "out" / "acceptance_case"
    for name in (
        "document.json",
        "output.json",
        "output.svg",
        "output.dxf",
        "overlay.png",
        "diff.png",
        "decision_report.json",
        "metrics.json",
    ):
        assert (case_dir / name).exists()

    metrics = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {
        "total_score",
        "edge_error",
        "complexity_score",
        "control_point_count",
        "topology_error_count",
        "self_intersection_count",
        "auto_apply_count",
        "auto_reject_count",
        "requires_external_decision_count",
        "svg_node_count",
        "runtime_ms",
    }.issubset(metrics)
    assert result.artifacts["metrics_json"].endswith("metrics.json")


def test_acceptance_benchmark_manifest_reports_overall_pass_fail_and_failure_reasons(tmp_path: Path) -> None:
    image_path = tmp_path / "case.png"
    _write_case_image(image_path, shape="rectangle")
    manifest_path = tmp_path / "acceptance_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case_pass",
                        "image_path": str(image_path),
                        "expected_geometry": {"line": 1},
                        "fail_thresholds": {"max_total_score": 500},
                    },
                    {
                        "case_id": "case_fail",
                        "image_path": str(image_path),
                        "expected_geometry": {"line": 1},
                        "fail_thresholds": {"max_total_score": 0.1},
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = BenchmarkRunner().run_acceptance_manifest(manifest_path, output_dir=tmp_path / "suite")

    assert report.summary["total_cases"] == 2
    assert report.summary["overall_pass"] is False
    assert report.summary["failed_case_count"] == 1
    assert "case_fail" in report.summary["failure_reasons"]
    assert (tmp_path / "suite" / "acceptance_report.json").exists()


def test_acceptance_benchmark_cli_writes_suite_report(tmp_path: Path) -> None:
    image_path = tmp_path / "cli_case.png"
    _write_case_image(image_path, shape="rectangle")
    manifest_path = tmp_path / "cli_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "cli_case",
                        "image_path": str(image_path),
                        "expected_geometry": {"line": 1},
                        "fail_thresholds": {"max_total_score": 500},
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "cli_out"
    report_json = tmp_path / "cli_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_acceptance_benchmark.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--report-json",
            str(report_json),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["summary"]["total_cases"] == 1
    assert payload["summary"]["overall_pass"] is True
