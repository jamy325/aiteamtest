from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.document import create_document
from core.types import CoordinateSystem
from services.auto_refinement_pipeline import AutoRefinementPipelineResult, AutoRefinementReport
from services.command_preview import CommandPreviewResult, ConstraintChangeSummary, ExportImpactSummary
from services.engine_protocol import (
    DecisionKind,
    ExternalDecisionAction,
    ExternalDecisionRequest,
    ExternalDecisionStatus,
    PolicyFeedback,
    RiskLevel,
)
from services.external_decision_queue import ExternalDecisionQueue
from services.preview_auto_accept_policy import PreviewDecision


def _document(document_id: str = "queued_preview"):
    return create_document(
        document_id=document_id,
        width=120.0,
        height=80.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )


def _feedback(reason_code: str = "requires_external_decision") -> PolicyFeedback:
    return PolicyFeedback(
        reason_code=reason_code,
        message="Needs external consumer.",
        metrics_delta={"score_delta": -0.4},
        policy_hint="review",
        retry_allowed=False,
    )


def _request(decision_id: str = "decision_1") -> ExternalDecisionRequest:
    return ExternalDecisionRequest(
        decision_id=decision_id,
        reason="High-risk change requires review.",
        risk_flags=("high_risk",),
        available_actions=("apply", "reject", "defer"),
        command={"tool": "propose_replace_path_with_circle", "path_id": "path_1"},
        candidate_id="candidate_1",
        preview_summary={"score_before": 8.0, "score_after": 7.6},
        policy_feedback=_feedback(),
    )


def _preview_result(preview_document=None) -> CommandPreviewResult:
    return CommandPreviewResult(
        success=True,
        command_id="cmd_1",
        reason=None,
        old_score=8.0,
        predicted_new_score=7.6,
        score_delta=-0.4,
        affected_paths=("path_1",),
        affected_segments=(),
        topology_status_before={"path_1": "closed"},
        topology_status_after={"path_1": "closed"},
        self_intersection_count_before={"path_1": 0},
        self_intersection_count_after={"path_1": 0},
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
        preview_document=preview_document,
    )


def test_external_decision_queue_enqueue_apply_and_json_round_trip(tmp_path: Path) -> None:
    queue = ExternalDecisionQueue()
    preview_document = _document()
    request = _request("decision_apply")

    enqueued = queue.enqueue(
        request,
        preview_document=preview_document,
        created_at="2026-05-25T00:00:00+00:00",
    )
    applied = queue.apply(
        "decision_apply",
        reason="approved by external consumer",
        timestamp="2026-05-25T00:05:00+00:00",
    )

    assert enqueued.status is ExternalDecisionStatus.PENDING
    assert applied.status is ExternalDecisionStatus.APPLIED
    assert applied.preview_document == preview_document
    assert applied.action_reason == "approved by external consumer"
    assert applied.updated_at == "2026-05-25T00:05:00+00:00"

    path = tmp_path / "queue.json"
    queue.save_json(path)
    restored = ExternalDecisionQueue.load_json(path)
    restored_record = restored.get("decision_apply")

    assert restored_record == applied
    assert json.loads(path.read_text(encoding="utf-8"))["records"][0]["status"] == "applied"


def test_external_decision_queue_reject_and_defer_record_reason_and_timestamp() -> None:
    queue = ExternalDecisionQueue()
    queue.enqueue(_request("decision_reject"))
    queue.enqueue(_request("decision_defer"))

    rejected = queue.reject(
        "decision_reject",
        reason="reviewer rejected for geometry mismatch",
        timestamp="2026-05-25T00:10:00+00:00",
    )
    deferred = queue.defer(
        "decision_defer",
        reason="waiting for human audit",
        timestamp="2026-05-25T00:12:00+00:00",
    )

    assert rejected.status is ExternalDecisionStatus.REJECTED
    assert rejected.action_reason == "reviewer rejected for geometry mismatch"
    assert rejected.updated_at == "2026-05-25T00:10:00+00:00"
    assert deferred.status is ExternalDecisionStatus.DEFERRED
    assert deferred.action_reason == "waiting for human audit"
    assert deferred.updated_at == "2026-05-25T00:12:00+00:00"


def test_external_decision_queue_rejects_duplicate_decision_id() -> None:
    queue = ExternalDecisionQueue()
    queue.enqueue(_request("decision_dup"))

    with pytest.raises(ValueError, match="duplicate decision_id"):
        queue.enqueue(_request("decision_dup"))


def test_external_decision_queue_rejects_illegal_action_and_missing_preview_document_for_apply() -> None:
    queue = ExternalDecisionQueue()
    queue.enqueue(_request("decision_no_preview"))
    queue.enqueue(_request("decision_bad_action"))

    with pytest.raises(ValueError, match="preview_document"):
        queue.apply("decision_no_preview", reason="attempt apply without preview")

    with pytest.raises(ValueError, match="unsupported external decision action"):
        queue.resolve("decision_bad_action", "ship-it", reason="invalid")


def test_external_decision_queue_can_be_built_from_auto_refinement_external_requests() -> None:
    preview_document = _document("preview_external")
    request = _request("decision_from_pipeline")
    preview_decision = PreviewDecision(
        command=request.command,
        preview_result=_preview_result(preview_document=preview_document),
        decision_kind=DecisionKind.REQUIRES_EXTERNAL_DECISION,
        reason="Needs external decision.",
        risk_flags=("high_risk",),
        risk_level=RiskLevel.HIGH,
        policy_feedback=_feedback(),
        external_decision_request=request,
    )
    result = AutoRefinementPipelineResult(
        refined_document=_document("result_doc"),
        candidates=(),
        proposed_commands=(dict(request.command),),
        preview_decisions=(preview_decision,),
        report=AutoRefinementReport(
            candidate_stats={},
            command_stats={},
            decision_stats={"auto_accept": 0, "user_confirm": 1, "reject": 0, "auto_apply": 0, "requires_external_decision": 1, "auto_reject": 0},
            score_before=8.0,
            score_after=7.6,
            integrity={"success": True, "errors": [], "warnings": []},
            dry_run_only=False,
        ),
    )

    records = result.external_decision_records()
    queue = ExternalDecisionQueue(records)

    assert len(records) == 1
    assert records[0].request == request
    assert records[0].preview_document == preview_document
    assert queue.get("decision_from_pipeline").status is ExternalDecisionStatus.PENDING
