from __future__ import annotations

import json

from core.document import create_document
from core.types import CoordinateSystem
from services.engine_protocol import (
    AutonomyLevel,
    DecisionKind,
    DecisionPolicyResult,
    EngineResult,
    EngineStatus,
    ExternalDecisionRequest,
    PolicyFeedback,
    RejectionMemoryItem,
    RiskLevel,
)


def _sample_document():
    return create_document(
        document_id="engine_protocol_doc",
        width=320.0,
        height=240.0,
        coordinate_system=CoordinateSystem(),
        metadata={"source": "unit_test"},
    )


def test_decision_kind_legacy_mapping_round_trips() -> None:
    assert DecisionKind.from_legacy("auto_accept") is DecisionKind.AUTO_APPLY
    assert DecisionKind.from_legacy("user_confirm") is DecisionKind.REQUIRES_EXTERNAL_DECISION
    assert DecisionKind.from_legacy("reject") is DecisionKind.AUTO_REJECT

    assert DecisionKind.AUTO_APPLY.to_legacy() == "auto_accept"
    assert DecisionKind.REQUIRES_EXTERNAL_DECISION.to_legacy() == "user_confirm"
    assert DecisionKind.AUTO_REJECT.to_legacy() == "reject"


def test_policy_feedback_round_trips_to_json_safe_dict() -> None:
    feedback = PolicyFeedback(
        reason_code="topology_regression",
        message="Topology worsened in preview.",
        metrics_delta={"score_after": 13.1, "self_intersection_delta": 1},
        policy_hint="avoid freeform replacement",
        retry_allowed=False,
        retry_constraints={"max_scope": "single_segment"},
        forbidden_repeated_commands=("propose_replace_path_with_bezier:path_001",),
    )

    payload = feedback.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    restored = PolicyFeedback.from_dict(payload)

    assert restored == feedback


def test_rejection_memory_item_round_trips() -> None:
    item = RejectionMemoryItem(
        target="path_001",
        tool="propose_replace_path_with_bezier",
        reason_code="high_risk",
        retry_count=2,
        last_metrics_delta={"score_delta": 0.2},
    )

    restored = RejectionMemoryItem.from_dict(item.to_dict())

    assert restored == item


def test_external_decision_request_includes_policy_feedback_and_preview_summary() -> None:
    feedback = PolicyFeedback(
        reason_code="needs_external_decision",
        message="High-risk freeform replacement requires escalation.",
        metrics_delta={"score_delta": -0.8},
        policy_hint="request human review",
        retry_allowed=True,
        retry_constraints={"forbid_risk_level_above": "medium"},
        forbidden_repeated_commands=("propose_replace_path_with_bezier:path_002",),
    )
    request = ExternalDecisionRequest(
        decision_id="decision_001",
        reason="High risk freeform replacement.",
        risk_flags=("high_risk", "freeform_bezier"),
        available_actions=("apply", "reject", "defer"),
        command={"tool": "propose_replace_path_with_bezier", "path_id": "path_002"},
        candidate_id="candidate_002",
        preview_summary={"score_before": 18.0, "score_after": 17.1},
        policy_feedback=feedback,
    )

    payload = request.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    restored = ExternalDecisionRequest.from_dict(payload)

    assert restored == request


def test_decision_policy_result_requires_external_decision_payload() -> None:
    feedback = PolicyFeedback(
        reason_code="risk_level_too_high",
        message="Escalate to external consumer.",
    )
    request = ExternalDecisionRequest(
        decision_id="decision_002",
        reason="Escalation required.",
        risk_flags=("high",),
        available_actions=("apply", "reject"),
        command={"tool": "propose_replace_segment_with_arc", "path_id": "path_001"},
        candidate_id="candidate_003",
        preview_summary={"self_intersection_delta": 0},
        policy_feedback=feedback,
    )
    result = DecisionPolicyResult(
        decision=DecisionKind.REQUIRES_EXTERNAL_DECISION,
        reason_code="risk_level_too_high",
        risk_level=RiskLevel.HIGH,
        policy_feedback=feedback,
        external_decision_request=request,
    )

    payload = result.to_dict()
    restored = DecisionPolicyResult.from_dict(payload)

    assert restored == result


def test_engine_status_uses_requires_external_decision_value_and_accepts_legacy_alias() -> None:
    assert EngineStatus.REQUIRES_EXTERNAL_DECISION.value == "requires_external_decision"
    assert EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS.value == "completed_with_unresolved_regions"
    assert EngineStatus.from_legacy("needs_external_decision") is EngineStatus.REQUIRES_EXTERNAL_DECISION
    assert EngineStatus.from_legacy("requires_external_decision") is EngineStatus.REQUIRES_EXTERNAL_DECISION
    assert (
        EngineStatus.from_legacy("completed_with_unresolved_regions")
        is EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS
    )


def test_risk_levels_are_sortable() -> None:
    levels = [RiskLevel.HIGH, RiskLevel.LOW, RiskLevel.MEDIUM_HIGH, RiskLevel.MEDIUM]

    assert sorted(levels) == [
        RiskLevel.LOW,
        RiskLevel.MEDIUM,
        RiskLevel.MEDIUM_HIGH,
        RiskLevel.HIGH,
    ]


def test_engine_result_round_trips_with_document_and_feedback_collections() -> None:
    document = _sample_document()
    feedback = PolicyFeedback(
        reason_code="auto_apply",
        message="Safe to apply.",
        metrics_delta={"score_delta": -1.2},
        policy_hint="commit preview",
        retry_allowed=False,
    )
    rejection_memory = RejectionMemoryItem(
        target="path_003",
        tool="propose_replace_path_with_bezier",
        reason_code="repeated_reject",
        retry_count=3,
        last_metrics_delta={"complexity_delta": 4},
    )
    decision = DecisionPolicyResult(
        decision=DecisionKind.AUTO_APPLY,
        reason_code="score_improved",
        risk_level=RiskLevel.LOW,
        policy_feedback=feedback,
    )
    external_decision = ExternalDecisionRequest(
        decision_id="decision_004",
        reason="Deferred for audit.",
        risk_flags=("audit",),
        available_actions=("apply", "reject", "defer"),
        command={"tool": "propose_replace_path_with_circle", "path_id": "path_010"},
        candidate_id="candidate_010",
        preview_summary={"score_before": 8.0, "score_after": 7.4},
        policy_feedback=feedback,
    )
    result = EngineResult(
        status=EngineStatus.REQUIRES_EXTERNAL_DECISION,
        document=document,
        report={"summary": "one decision escalated"},
        decisions=(decision,),
        external_decisions=(external_decision,),
        policy_feedback=(feedback,),
        rejection_memory=(rejection_memory,),
        errors=("none",),
        metadata={"autonomy_level": AutonomyLevel.AUTONOMOUS_SAFE.value},
    )

    payload = result.to_dict()
    assert json.loads(json.dumps(payload)) == payload

    restored = EngineResult.from_dict(payload)

    assert restored == result
