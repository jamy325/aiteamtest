from __future__ import annotations

import ast
from pathlib import Path

from core.document import create_document
from core.types import CoordinateSystem
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
):
    document = _document("engine_result_doc")
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
    )
    return AutoRefinementPipelineResult(
        refined_document=document,
        candidates=(),
        proposed_commands=(command,),
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

    def run_from_file(self, image_path, *, document_id="document_1", **kwargs):
        self.calls.append((str(image_path), document_id))
        return self.result


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
