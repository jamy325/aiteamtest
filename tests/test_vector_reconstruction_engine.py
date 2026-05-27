from __future__ import annotations

import ast
from pathlib import Path

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment, Style
from services.distance_field_diff import DistanceFieldDiffResult
from services.auto_refinement_pipeline import AutoRefinementPipelineResult, AutoRefinementReport
from services.command_preview import CommandPreviewResult, ConstraintChangeSummary, ExportImpactSummary
from services.engine_protocol import (
    AutonomyLevel,
    DecisionKind,
    EngineStatus,
    ExternalDecisionRequest,
    PolicyFeedback,
    RiskLevel,
)
from services.minimal_pipeline import MinimalPipelineResult
from services.preview_auto_accept_policy import PreviewDecision
from services.vector_reconstruction_engine import VectorReconstructionEngine, VectorReconstructionEngineConfig


def _document(document_id: str = "engine_doc"):
    return create_document(
        document_id=document_id,
        width=120.0,
        height=80.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )


def _stroke_document(document_id: str = "engine_stroke_doc"):
    document = create_document(
        document_id=document_id,
        width=120.0,
        height=80.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
        metadata={
            "pipeline": {
                "source_contours": {
                    "binary_contours": [
                        {
                            "contour_id": "binary_1",
                            "points": [[10.0, 36.0], [110.0, 36.0], [110.0, 44.0], [10.0, 44.0]],
                            "coordinate_space": "vector",
                            "closed": True,
                            "depth": 0,
                        }
                    ],
                    "skeleton_contours": [
                        {
                            "contour_id": "skeleton_1",
                            "points": [[10.0, 40.0], [110.0, 40.0]],
                            "coordinate_space": "vector",
                            "closed": False,
                        }
                    ],
                }
            }
        },
    )
    document = add_path(
        document,
        VectorPath(
            path_id="stroke_path",
            source="skeleton_contour",
            segments=("stroke_seg",),
            style=Style(stroke_width=6.0),
            metadata={
                "stroke_width_confidence": 0.9,
                "endpoint_count": 2,
                "junction_count": 1,
                "branch_count": 3,
            },
        ),
    )
    document = add_segment(
        document,
        Segment(
            segment_id="stroke_seg",
            path_id="stroke_path",
            type="line",
            params={"start": [10.0, 40.0], "end": [110.0, 40.0]},
        ),
    )
    return document


def _preview_result(command_id: str, *, path_id: str = "path_1") -> CommandPreviewResult:
    return CommandPreviewResult(
        success=True,
        command_id=command_id,
        reason=None,
        old_score=10.0,
        predicted_new_score=9.2,
        score_delta=-0.8,
        affected_paths=(path_id,),
        affected_segments=(),
        topology_status_before={path_id: "open"},
        topology_status_after={path_id: "open"},
        self_intersection_count_before={path_id: 0},
        self_intersection_count_after={path_id: 0},
        segment_type_summary={},
        constraint_change_summary=ConstraintChangeSummary(
            before={},
            after={},
            delta={},
            added_constraint_ids=(),
            removed_constraint_ids=(),
            changed_constraint_ids=(),
        ),
        export_impact_summary=ExportImpactSummary(before={}, after={}, delta={}),
    )


def _auto_result(
    *,
    decision_kind: DecisionKind = DecisionKind.AUTO_APPLY,
    legacy_decision: str = "auto_accept",
    report_status: str = EngineStatus.COMPLETED.value,
    dry_run_only: bool = False,
    refined_document=None,
    proposed_commands: tuple[dict[str, object], ...] | None = None,
    ai_review_summary: dict[str, object] | None = None,
):
    document = refined_document or _document("engine_result_doc")
    command = {
        "tool": "propose_replace_path_with_circle",
        "path_id": "path_1",
        "reason": "intent only",
        "confidence": 0.82,
        "requires_user_confirmation": True,
    }
    feedback = PolicyFeedback(
        reason_code="auto_apply" if decision_kind == DecisionKind.AUTO_APPLY else "requires_external_decision",
        message="policy feedback",
        metrics_delta={"score_delta": -0.8},
        policy_hint="review if needed",
        retry_allowed=decision_kind != DecisionKind.AUTO_APPLY,
    )
    external_request = None
    if decision_kind == DecisionKind.REQUIRES_EXTERNAL_DECISION:
        external_request = ExternalDecisionRequest(
            decision_id="decision_1",
            reason="Needs external decision.",
            risk_flags=("manual_only_mode",),
            available_actions=("apply", "reject"),
            command=command,
            candidate_id=None,
            preview_summary={"score_delta": -0.8},
            policy_feedback=feedback,
        )
    preview_decision = PreviewDecision(
        command=command,
        preview_result=_preview_result("cmd_1"),
        decision=legacy_decision,  # type: ignore[arg-type]
        reason="decision reason",
        risk_flags=(),
        decision_kind=decision_kind,
        risk_level=RiskLevel.MEDIUM if decision_kind == DecisionKind.REQUIRES_EXTERNAL_DECISION else RiskLevel.LOW,
        policy_feedback=feedback,
        external_decision_request=external_request,
    )
    report = AutoRefinementReport(
        candidate_stats={"total": 0},
        command_stats={"total": 1},
        decision_stats={
            "auto_accept": 1 if legacy_decision == "auto_accept" else 0,
            "user_confirm": 1 if legacy_decision == "user_confirm" else 0,
            "reject": 1 if legacy_decision == "reject" else 0,
            "auto_apply": 1 if decision_kind == DecisionKind.AUTO_APPLY else 0,
            "requires_external_decision": 1 if decision_kind == DecisionKind.REQUIRES_EXTERNAL_DECISION else 0,
            "auto_reject": 1 if decision_kind == DecisionKind.AUTO_REJECT else 0,
        },
        processing_summary={
            "processing_contour_source": "all",
            "processed_path_count": 0,
            "processed_binary_path_count": 0,
            "processed_skeleton_path_count": 0,
            "available_binary_path_count": 0,
            "available_skeleton_path_count": 0,
            "warnings": [],
        },
        score_before=10.0,
        score_after=9.2,
        integrity={"success": True, "errors": [], "warnings": []},
        dry_run_only=dry_run_only,
        status=report_status,
        iteration_count=1,
        policy_feedback=(feedback.to_dict(),),
        rejection_memory=(),
        forbidden_repeated_commands=(),
        unresolved_targets=(),
        ai_review_summary=ai_review_summary or {},
    )
    return AutoRefinementPipelineResult(
        refined_document=document,
        candidates=(),
        proposed_commands=proposed_commands or (command,),
        preview_decisions=(preview_decision,),
        report=report,
    )


class _FakeAutoRefinementPipeline:
    def __init__(self, result: AutoRefinementPipelineResult) -> None:
        self.result = result
        self.calls: list[tuple[str, tuple[str, ...], bool | None]] = []

    def run(self, document, *, target_types=(), dry_run_only=None):
        self.calls.append(("run", tuple(target_types), dry_run_only))
        return self.result

    def run_from_pipeline_result(self, pipeline_result, *, target_types=(), dry_run_only=None):
        self.calls.append(("run_from_pipeline_result", tuple(target_types), dry_run_only))
        return self.result

    def run_with_ai_review(self, document, *, target_types=(), dry_run_only=None):
        self.calls.append(("run_with_ai_review", tuple(target_types), dry_run_only))
        return self.result

    def run_from_pipeline_result_with_ai_review(self, pipeline_result, *, target_types=(), dry_run_only=None):
        self.calls.append(("run_from_pipeline_result_with_ai_review", tuple(target_types), dry_run_only))
        return self.result


class _FakeMinimalPipeline:
    def __init__(self, result: MinimalPipelineResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []
        self.distance_field_diff_renderer = _FakeDistanceFieldDiffRenderer()

    def run_from_file(self, image_path, *, document_id="document_1", **kwargs):
        self.calls.append((str(image_path), document_id))
        return self.result

    def export_overlay(self, document, source_image):
        return b"overlay"

    def export_distance_field_diff(self, document):
        return b"diff"


class _FakeDistanceFieldDiffRenderer:
    def render_diff(self, document):
        stroke_mask_error = 0.125 if document.document_id == "engine_result_doc" else 0.25
        return DistanceFieldDiffResult(
            image=[],
            missing_edge_error=0.0,
            overdraw_error=0.0,
            chamfer_error=0.0,
            source_point_count=0,
            vector_point_count=0,
            stroke_mask_error=stroke_mask_error,
        )


class _RecordingExporter:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def export_document(self, document, *, export_mode="all_debug"):
        self.calls.append((document.document_id, export_mode))
        return self.payload


def test_vector_reconstruction_engine_runs_from_document() -> None:
    pipeline = _FakeAutoRefinementPipeline(_auto_result())
    engine = VectorReconstructionEngine(
        auto_refinement_pipeline=pipeline,
        config=VectorReconstructionEngineConfig(target_types=("circle",), dry_run_only=True),
    )

    result = engine.run_document(_document("doc_input"))

    assert result.status is EngineStatus.COMPLETED
    assert result.document is not None
    assert result.document.document_id == "engine_result_doc"
    assert pipeline.calls == [("run", ("circle",), True)]
    assert result.metadata["target_types"] == ["circle"]
    assert result.metadata["dry_run_only"] is True


def test_vector_reconstruction_engine_runs_from_pipeline_result() -> None:
    pipeline = _FakeAutoRefinementPipeline(_auto_result())
    pipeline_result = MinimalPipelineResult(
        document=_document("doc_pipeline_result"),
        json_payload="{}",
        extracted_contours=None,  # type: ignore[arg-type]
        source_image=None,
        debug_artifacts=None,
    )
    engine = VectorReconstructionEngine(auto_refinement_pipeline=pipeline)

    result = engine.run_pipeline_result(pipeline_result, target_types=("line",))

    assert result.status is EngineStatus.COMPLETED
    assert pipeline.calls == [("run_from_pipeline_result", ("line",), False)]


def test_vector_reconstruction_engine_runs_from_image_path_via_minimal_pipeline() -> None:
    pipeline = _FakeAutoRefinementPipeline(_auto_result())
    pipeline_result = MinimalPipelineResult(
        document=_document("doc_from_image"),
        json_payload="{}",
        extracted_contours=None,  # type: ignore[arg-type]
        source_image=None,
        debug_artifacts=None,
    )
    minimal_pipeline = _FakeMinimalPipeline(pipeline_result)
    engine = VectorReconstructionEngine(
        minimal_pipeline=minimal_pipeline,
        auto_refinement_pipeline=pipeline,
        config=VectorReconstructionEngineConfig(document_id="image_doc"),
    )

    result = engine.run_image_path("input.png")

    assert result.status is EngineStatus.COMPLETED
    assert minimal_pipeline.calls == [("input.png", "image_doc")]
    assert pipeline.calls == [("run_from_pipeline_result", (), False)]


def test_vector_reconstruction_engine_manual_only_maps_to_requires_external_decision() -> None:
    pipeline = _FakeAutoRefinementPipeline(
        _auto_result(
            decision_kind=DecisionKind.REQUIRES_EXTERNAL_DECISION,
            legacy_decision="user_confirm",
            report_status=EngineStatus.COMPLETED.value,
        )
    )
    engine = VectorReconstructionEngine(
        auto_refinement_pipeline=pipeline,
        config=VectorReconstructionEngineConfig(autonomy_level=AutonomyLevel.MANUAL_ONLY),
    )

    result = engine.run_document(_document("doc_manual"))

    assert result.status is EngineStatus.REQUIRES_EXTERNAL_DECISION
    assert result.external_decisions
    assert result.metadata["autonomy_level"] == AutonomyLevel.MANUAL_ONLY.value


def test_vector_reconstruction_engine_uses_ai_review_path_and_preserves_unresolved_status() -> None:
    pipeline = _FakeAutoRefinementPipeline(
        _auto_result(
            decision_kind=DecisionKind.AUTO_REJECT,
            legacy_decision="reject",
            report_status=EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS.value,
        )
    )
    engine = VectorReconstructionEngine(
        auto_refinement_pipeline=pipeline,
        config=VectorReconstructionEngineConfig(enable_ai_review=True, max_iterations=4),
    )

    result = engine.run_document(_document("doc_ai"))

    assert result.status is EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS
    assert pipeline.calls == [("run_with_ai_review", (), False)]
    assert result.metadata["enable_ai_review"] is True
    assert result.metadata["max_iterations"] == 4


def test_vector_reconstruction_engine_has_no_ui_dependency() -> None:
    source = Path("services/vector_reconstruction_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert all(not item.startswith("ui") for item in imports)


def test_vector_reconstruction_engine_run_artifact_bundle_propagates_export_mode_and_records_it() -> None:
    pipeline_result = MinimalPipelineResult(
        document=_stroke_document("bundle_doc"),
        json_payload="{}",
        extracted_contours=None,  # type: ignore[arg-type]
        source_image="fake-image",
        debug_artifacts=None,
    )
    minimal_pipeline = _FakeMinimalPipeline(pipeline_result)
    pipeline = _FakeAutoRefinementPipeline(_auto_result(refined_document=_stroke_document("engine_result_doc")))
    engine = VectorReconstructionEngine(
        minimal_pipeline=minimal_pipeline,
        auto_refinement_pipeline=pipeline,
        config=VectorReconstructionEngineConfig(export_mode="all_debug"),
    )
    svg_exporter = _RecordingExporter("<svg/>")
    dxf_exporter = _RecordingExporter("0\nEOF\n")
    engine.svg_exporter = svg_exporter
    engine.dxf_exporter = dxf_exporter

    bundle = engine.run_artifact_bundle("input.png", export_mode="centerline")

    assert svg_exporter.calls == [("engine_result_doc", "centerline")]
    assert dxf_exporter.calls == [("engine_result_doc", "centerline")]
    assert bundle.metrics["export_mode"] == "centerline"
    assert bundle.metrics["processing_contour_source"] == "skeleton"
    assert bundle.metrics["stroke_width"] == 6.0
    assert bundle.metrics["stroke_width_confidence"] == 0.9
    assert bundle.metrics["stroke_mask_error"] == 0.125
    assert bundle.metrics["stroke_mask_error_score"] == 0.125
    assert bundle.decision_report["metadata"]["export_mode"] == "centerline"
    assert bundle.decision_report["metadata"]["processing_contour_source"] == "skeleton"
    assert bundle.decision_report["stroke_summary"]["stroke_width"] == 6.0
    assert bundle.decision_report["report"]["stroke_mask_error"] == 0.125


def test_vector_reconstruction_engine_records_ai_review_metadata_and_command_counts() -> None:
    proposed_commands = (
        {
            "tool": "propose_replace_path_with_circle",
            "path_id": "path_algo",
            "reason": "algorithm",
            "confidence": 0.7,
            "requires_user_confirmation": True,
        },
        {
            "tool": "propose_replace_path_with_ellipse",
            "path_id": "path_ai",
            "reason": "ai",
            "confidence": 0.8,
            "requires_user_confirmation": True,
            "proposal_source": "ai_review",
        },
    )
    pipeline = _FakeAutoRefinementPipeline(
        _auto_result(
            decision_kind=DecisionKind.REQUIRES_EXTERNAL_DECISION,
            legacy_decision="user_confirm",
            report_status=EngineStatus.REQUIRES_EXTERNAL_DECISION.value,
            proposed_commands=proposed_commands,
            ai_review_summary={
                "ai_input_mode": "local_visual_context",
                "ai_input_truncated": False,
                "ai_prompt_char_count": 1842,
                "ai_max_prompt_chars": 120000,
                "ai_review_job_count": 2,
                "ai_review_image_count": 3,
                "ai_review_image_file_count": 3,
                "ai_review_panel_count": 6,
                "ai_review_crop_max_size_px": 512,
                "ai_review_candidate_count": 2,
                "ai_review_sampled_point_count": 0,
                "review_jobs": [
                    {
                        "job_id": "review_job_1",
                        "path_id": "path_ai",
                        "window_id": "path_ai:window_1",
                        "crop_bbox": [0, 0, 40, 40],
                        "image_count": 3,
                        "truncated": False,
                    }
                ],
            },
        )
    )
    pipeline_result = MinimalPipelineResult(
        document=_stroke_document("bundle_doc_ai"),
        json_payload="{}",
        extracted_contours=None,  # type: ignore[arg-type]
        source_image="fake-image",
        debug_artifacts=None,
    )
    minimal_pipeline = _FakeMinimalPipeline(pipeline_result)
    engine = VectorReconstructionEngine(
        minimal_pipeline=minimal_pipeline,
        auto_refinement_pipeline=pipeline,
        config=VectorReconstructionEngineConfig(
            enable_ai_review=True,
            ai_provider="openai",
            ai_model="gpt-4.1-mini",
            ai_status="recorded_replay",
        ),
    )

    bundle = engine.run_artifact_bundle("input.png")

    assert bundle.metrics["enable_ai_review"] is True
    assert bundle.metrics["ai_provider"] == "openai"
    assert bundle.metrics["ai_model"] == "gpt-4.1-mini"
    assert bundle.metrics["ai_status"] == "recorded_replay"
    assert bundle.metrics["ai_proposed_count"] == 1
    assert bundle.metrics["algorithm_proposed_count"] == 1
    assert bundle.metrics["ai_input_mode"] == "local_visual_context"
    assert bundle.metrics["ai_input_truncated"] is False
    assert bundle.metrics["ai_prompt_char_count"] == 1842
    assert bundle.metrics["ai_review_job_count"] == 2
    assert bundle.metrics["ai_review_image_count"] == 3
    assert bundle.metrics["ai_review_image_file_count"] == 3
    assert bundle.metrics["ai_review_panel_count"] == 6
    assert bundle.metrics["ai_review_crop_max_size_px"] == 512
    assert bundle.decision_report["ai_review_summary"]["ai_input_mode"] == "local_visual_context"
    assert bundle.decision_report["metadata"]["ai_provider"] == "openai"
    assert bundle.decision_report["metadata"]["ai_model"] == "gpt-4.1-mini"
    assert bundle.decision_report["metadata"]["ai_status"] == "recorded_replay"
