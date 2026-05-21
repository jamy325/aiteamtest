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
from services.command_executor import CommandExecutionResult
from services.command_preview import CommandPreviewResult, ConstraintChangeSummary, ExportImpactSummary
from services.contour_extractor import BinaryContour, ExtractedContours
from services.minimal_pipeline import MinimalPipelineResult
from services.preview_auto_accept_policy import PreviewDecision


def _write_case_image(image_path: Path, *, shape: str) -> None:
    image = np.zeros((120, 140), dtype=np.uint8)
    if shape == "rectangle":
        cv2.rectangle(image, (15, 15), (90, 95), 255, thickness=-1)
    elif shape == "circle":
        cv2.circle(image, (60, 60), 28, 255, thickness=2)
    elif shape == "line":
        cv2.line(image, (20, 30), (120, 85), 255, thickness=2)
    else:
        raise ValueError(f"unsupported shape: {shape}")
    assert cv2.imwrite(str(image_path), image)


def _write_manifest(tmp_path: Path, cases: list[dict[str, object]]) -> Path:
    manifest_path = tmp_path / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps({"cases": cases}, indent=2), encoding="utf-8")
    return manifest_path


def test_benchmark_runner_parses_manifest_structure(tmp_path: Path) -> None:
    for name in ("rectangle", "circle", "line"):
        _write_case_image(tmp_path / f"{name}.png", shape=name)

    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "case_id": "case_rect",
                "image_path": "rectangle.png",
                "expected_geometry": {"line": 1},
                "expected_constraints": {"g1_continuity": 0},
                "expected_export": {"svg": {"min_element_count": 1}},
                "proposed_commands": [],
                "segment_type": "line",
            },
            {
                "case_id": "case_circle",
                "image_path": "circle.png",
                "expected_geometry": {"line": 1},
                "expected_constraints": {},
                "expected_export": {"dxf": {"min_entity_count": 1}},
            },
            {
                "case_id": "case_line",
                "image_path": "line.png",
                "expected_geometry": {"line": 1},
                "expected_constraints": {},
                "expected_export": {"json": {"min_char_count": 1}},
            },
        ],
    )

    cases = BenchmarkRunner().load_manifest(manifest_path)

    assert len(cases) == 3
    assert cases[0].case_id == "case_rect"
    assert Path(cases[0].image_path).name == "rectangle.png"
    assert cases[0].expected_geometry == {"line": 1}
    assert cases[0].expected_export["svg"]["min_element_count"] == 1


def test_benchmark_runner_runs_single_case_and_returns_stats(tmp_path: Path) -> None:
    image_path = tmp_path / "single_case.png"
    _write_case_image(image_path, shape="rectangle")
    case = BenchmarkCase(
        case_id="single_case",
        image_path=str(image_path),
        expected_geometry={"line": 1},
        expected_export={"svg": {"min_element_count": 1}},
    )

    result = BenchmarkRunner().run_case(case)

    assert result.success is True
    assert result.case_id == "single_case"
    assert result.document_id == "benchmark_single_case"
    assert result.stats["segment_count"] > 0
    assert result.stats["total_score"] >= 0.0
    assert result.stats["edge_error"] >= 0.0
    assert result.actual_geometry["line"] > 0
    assert result.geometry_hits["line"] == 1
    assert result.export_summary["json"]["char_count"] > 0
    assert result.export_summary["svg"]["element_count"] > 0
    assert result.export_summary["dxf"]["entity_count"] > 0


def test_benchmark_runner_optionally_executes_proposed_commands(tmp_path: Path) -> None:
    document = create_document(
        document_id="doc_fake",
        width=10.0,
        height=10.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_1", segments=("seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="seg_1",
            path_id="path_1",
            type="polyline",
            params={
                "points": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                "start": [0.0, 0.0],
                "end": [2.0, 0.0],
            },
        ),
    )
    contours = ExtractedContours(
        binary_contours=(
            BinaryContour(
                contour_id="binary_1",
                source="binary_contour",
                points=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
                coordinate_space="vector",
                closed=False,
                area=0.0,
                depth=0,
                parent_contour=None,
                children=(),
            ),
        ),
        skeleton_contours=(),
    )

    class _FakePipeline:
        def run_from_file(self, image_path: str | Path, *, document_id: str = "document_1") -> MinimalPipelineResult:
            return MinimalPipelineResult(
                document=document,
                json_payload="{}",
                extracted_contours=contours,
                source_image=np.zeros((8, 8), dtype=np.uint8),
            )

    class _FakeExecutor:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def execute(self, command: object, current_document: object) -> CommandExecutionResult:
            assert isinstance(command, dict)
            self.calls.append(dict(command))
            updated_document = current_document
            if isinstance(current_document, type(document)):
                updated_document = current_document
            return CommandExecutionResult(
                success=True,
                command_id=str(command.get("command_id", "cmd")),
                document=updated_document,
                affected_paths=("path_1",),
                affected_segments=("seg_1",),
                old_score=1.0,
                new_score=0.5,
                topology_status="open",
                self_intersection_count=0,
                requires_rerender=True,
                reason=None,
            )

    image_path = tmp_path / "ignored.png"
    _write_case_image(image_path, shape="line")
    executor = _FakeExecutor()
    runner = BenchmarkRunner(
        pipeline_factory=lambda case: _FakePipeline(),
        command_executor=executor,
    )

    result = runner.run_case(
        BenchmarkCase(
            case_id="with_commands",
            image_path=str(image_path),
            proposed_commands=(
                {
                    "command_id": "cmd_1",
                    "tool": "propose_replace_segment_with_line",
                    "path_id": "path_1",
                    "segment_range": [0, 0],
                    "reason": "straight line",
                    "confidence": 0.8,
                    "requires_user_confirmation": True,
                },
            ),
        ),
        execute_proposed_commands=True,
    )

    assert len(executor.calls) == 1
    assert result.command_results[0]["command_id"] == "cmd_1"
    assert result.command_results[0]["success"] is True


def test_benchmark_runner_aggregates_three_case_summary(tmp_path: Path) -> None:
    cases_payload: list[dict[str, object]] = []
    for shape in ("rectangle", "circle", "line"):
        image_path = tmp_path / f"{shape}.png"
        _write_case_image(image_path, shape=shape)
        cases_payload.append(
            {
                "case_id": f"case_{shape}",
                "image_path": image_path.name,
                "expected_geometry": {"line": 1},
                "expected_constraints": {},
                "expected_export": {"svg": {"min_element_count": 1}},
            }
        )
    manifest_path = _write_manifest(tmp_path, cases_payload)

    report = BenchmarkRunner().run_manifest(manifest_path)

    assert len(report.cases) == 3
    assert report.summary["total_cases"] == 3
    assert report.summary["average_total_score"] >= 0.0
    assert report.summary["total_segments"] > 0
    assert report.summary["geometry_hit_totals"]["line"] >= 3
    assert report.summary["case_ids"] == ["case_rectangle", "case_circle", "case_line"]


def test_benchmark_runner_cli_writes_json_report(tmp_path: Path) -> None:
    cases_payload: list[dict[str, object]] = []
    for shape in ("rectangle", "circle", "line"):
        image_path = tmp_path / f"{shape}.png"
        _write_case_image(image_path, shape=shape)
        cases_payload.append(
            {
                "case_id": f"cli_{shape}",
                "image_path": image_path.name,
                "expected_geometry": {"line": 1},
                "expected_constraints": {},
                "expected_export": {"svg": {"min_element_count": 1}},
            }
        )
    manifest_path = _write_manifest(tmp_path, cases_payload)
    output_path = tmp_path / "benchmark_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.benchmark_runner",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_cases"] == 3
    assert len(payload["cases"]) == 3


def _preview_result(command_id: str) -> CommandPreviewResult:
    return CommandPreviewResult(
        success=True,
        command_id=command_id,
        reason=None,
        old_score=10.0,
        predicted_new_score=8.5,
        score_delta=-1.5,
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
        export_impact_summary=ExportImpactSummary(before={"json_char_count": 10}, after={"json_char_count": 9}, delta={"json_char_count": -1}),
    )


def test_benchmark_runner_auto_refine_case_includes_quality_gate_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "circle_case.png"
    _write_case_image(image_path, shape="circle")

    document = create_document(
        document_id="doc_auto_refine_before",
        width=100.0,
        height=100.0,
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
    refined_document = create_document(
        document_id="doc_auto_refine_after",
        width=100.0,
        height=100.0,
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

    class _FakePipeline:
        def run_from_file(self, image_path: str | Path, *, document_id: str = "document_1") -> MinimalPipelineResult:
            return MinimalPipelineResult(
                document=document,
                json_payload="{}",
                extracted_contours=contours,
                source_image=np.zeros((8, 8), dtype=np.uint8),
            )

    class _FakeAutoRefinementPipeline:
        def run_from_pipeline_result(self, pipeline_result: MinimalPipelineResult, *, target_types=None, dry_run_only=None) -> AutoRefinementPipelineResult:
            return AutoRefinementPipelineResult(
                refined_document=refined_document,
                candidates=(),
                proposed_commands=(
                    {
                        "command_id": "cmd_auto_1",
                        "tool": "propose_replace_path_with_circle",
                        "path_id": "path_1",
                        "reason": "intent only",
                        "confidence": 0.82,
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
                            "confidence": 0.82,
                            "requires_user_confirmation": True,
                        },
                        preview_result=_preview_result("cmd_auto_1"),
                        decision="auto_accept",
                        reason="good improvement",
                        risk_flags=(),
                    ),
                ),
                report=AutoRefinementReport(
                    candidate_stats={"total": 1, "by_target_type": {"circle": 1}, "by_source": {"raw_contour_points": 1}},
                    command_stats={"total": 1, "evaluated_total": 1, "by_tool": {"propose_replace_path_with_circle": 1}, "batch_count": 0},
                    decision_stats={"auto_accept": 1, "user_confirm": 0, "reject": 0},
                    score_before=12.0,
                    score_after=8.5,
                    integrity={"success": True, "error_count": 0, "warning_count": 0, "affected_ids": [], "errors": [], "warnings": []},
                    dry_run_only=False,
                    target_types=("circle",),
                ),
            )

    runner = BenchmarkRunner(
        pipeline_factory=lambda case: _FakePipeline(),
        auto_refinement_pipeline=_FakeAutoRefinementPipeline(),
    )

    result = runner.run_case(
        BenchmarkCase(
            case_id="auto_refine_case",
            image_path=str(image_path),
            auto_refine=True,
            auto_refine_target_types=("circle",),
            expected_geometry={"circle": 1},
        )
    )

    assert result.success is True
    assert result.auto_refine is True
    assert result.geometry_before["circle"] == 0
    assert result.geometry_after["circle"] == 1
    assert result.geometry_hits["circle"] == 1
    assert result.candidate_counts["total"] == 1
    assert result.proposed_command_counts["propose_replace_path_with_circle"] == 1
    assert result.accepted_count == 1
    assert result.rejected_count == 0
    assert result.user_confirm_count == 0
    assert result.preview_decisions[0]["decision"] == "auto_accept"
    assert result.score_before >= 0.0
    assert result.score_after >= 0.0
    assert result.score_delta == result.score_after - result.score_before
    assert result.export_summary["dxf"]["entity_counts"]["CIRCLE"] >= 1
    assert result.export_summary["svg"]["element_count"] > 0
    assert set(result.export_summary["svg"]["path_command_counts"]).issuperset({"M", "L", "C", "A", "Z"})


def test_benchmark_runner_auto_refine_case_reports_threshold_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "threshold_case.png"
    _write_case_image(image_path, shape="line")
    case = BenchmarkCase(
        case_id="threshold_fail_case",
        image_path=str(image_path),
        fail_thresholds={"min_circle_count": 1, "max_total_score": 0.1},
    )

    result = BenchmarkRunner().run_case(case)

    assert result.success is False
    assert result.failure_reason is not None
    assert "min_circle_count" in result.failure_reason or "max_total_score" in result.failure_reason
