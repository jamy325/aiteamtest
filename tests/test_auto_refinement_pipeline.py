from __future__ import annotations

import json
from pathlib import Path

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.ai_agent import AIReviewInput, AIReviewService
from services.auto_refinement_pipeline import AutoRefinementPipeline, AutoRefinementPipelineConfig
from services.command_preview import CommandPreviewResult, ConstraintChangeSummary, ExportImpactSummary
from services.engine_protocol import DecisionKind, EngineStatus, ExternalDecisionRequest, PolicyFeedback, RiskLevel
from services.minimal_pipeline import MinimalPipeline
from services.preview_auto_accept_policy import PreviewDecision, PreviewPolicyResult


def _rectangle_document():
    document = create_document(
        document_id="doc_rectangle_pipeline",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="rect_path",
            closed=True,
            segments=("rect_0", "rect_1", "rect_2", "rect_3"),
        ),
    )
    corners = ((20.0, 20.0), (120.0, 20.0), (120.0, 80.0), (20.0, 80.0))
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        document = add_segment(
            document,
            Segment(
                segment_id=f"rect_{index}",
                path_id="rect_path",
                type="line",
                params={"start": [start[0], start[1]], "end": [end[0], end[1]]},
            ),
        )
    return document


def _mixed_source_rectangle_document():
    document = create_document(
        document_id="doc_mixed_pipeline",
        width=240.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="binary_rect_path",
            closed=True,
            source="binary_contour",
            segments=("binary_0", "binary_1", "binary_2", "binary_3"),
        ),
    )
    document = add_path(
        document,
        VectorPath(
            path_id="skeleton_rect_path",
            closed=True,
            source="skeleton_contour",
            segments=("skeleton_0", "skeleton_1", "skeleton_2", "skeleton_3"),
        ),
    )
    binary_corners = ((20.0, 20.0), (100.0, 20.0), (100.0, 80.0), (20.0, 80.0))
    skeleton_corners = ((120.0, 20.0), (200.0, 20.0), (200.0, 80.0), (120.0, 80.0))
    for index in range(4):
        binary_start = binary_corners[index]
        binary_end = binary_corners[(index + 1) % 4]
        skeleton_start = skeleton_corners[index]
        skeleton_end = skeleton_corners[(index + 1) % 4]
        document = add_segment(
            document,
            Segment(
                segment_id=f"binary_{index}",
                path_id="binary_rect_path",
                type="line",
                params={"start": [binary_start[0], binary_start[1]], "end": [binary_end[0], binary_end[1]]},
            ),
        )
        document = add_segment(
            document,
            Segment(
                segment_id=f"skeleton_{index}",
                path_id="skeleton_rect_path",
                type="line",
                params={"start": [skeleton_start[0], skeleton_start[1]], "end": [skeleton_end[0], skeleton_end[1]]},
            ),
        )
    return document


def _policy_preview_result(
    *,
    command_id: str,
    path_id: str = "path_1",
    score_delta: float | None = -0.1,
    self_intersection_after: int = 0,
) -> CommandPreviewResult:
    old_score = 10.0 if score_delta is not None else None
    predicted_new_score = (old_score + score_delta) if old_score is not None and score_delta is not None else None
    return CommandPreviewResult(
        success=score_delta is not None,
        command_id=command_id,
        reason=None,
        old_score=old_score,
        predicted_new_score=predicted_new_score,
        score_delta=score_delta,
        affected_paths=(path_id,),
        affected_segments=(),
        topology_status_before={path_id: "open"},
        topology_status_after={path_id: "open"},
        self_intersection_count_before={path_id: 0},
        self_intersection_count_after={path_id: self_intersection_after},
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


class _EmptyDetector:
    def detect_candidates(self, document, *, contour_source="all"):
        return ()


class _EmptyPlanner:
    def plan_commands(self, document, candidates):
        return ()


class _SequentialRejectingPreviewPolicy:
    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.calls = 0

    def evaluate_commands(self, commands, document):
        self.calls += 1
        if not self._decisions:
            return PreviewPolicyResult(
                final_document=document,
                decisions=(),
                accepted_count=0,
                rejected_count=0,
                user_confirm_count=0,
            )
        next_decisions = tuple(self._decisions.pop(0))
        return PreviewPolicyResult(
            final_document=document,
            decisions=next_decisions,
            accepted_count=sum(1 for item in next_decisions if item.decision_kind == DecisionKind.AUTO_APPLY),
            rejected_count=sum(1 for item in next_decisions if item.decision_kind == DecisionKind.AUTO_REJECT),
            user_confirm_count=sum(
                1 for item in next_decisions if item.decision_kind == DecisionKind.REQUIRES_EXTERNAL_DECISION
            ),
        )


def test_auto_refinement_pipeline_circle_fixture_auto_accepts_circle_command() -> None:
    pipeline_result = MinimalPipeline(segment_type="line").run_from_file(
        Path("test_images/circle/test_input_circle.png"),
        document_id="circle_auto_pipeline",
    )

    result = AutoRefinementPipeline().run_from_pipeline_result(
        pipeline_result,
        target_types=("circle",),
    )

    assert any(candidate.target_type == "circle" for candidate in result.candidates)
    assert any(command["tool"] == "propose_replace_path_with_circle" for command in result.proposed_commands)
    assert any(decision.decision == "auto_accept" for decision in result.preview_decisions)
    assert any(decision.decision_kind is DecisionKind.AUTO_APPLY for decision in result.preview_decisions)
    auto_apply_decisions = [
        decision for decision in result.preview_decisions if decision.decision_kind is DecisionKind.AUTO_APPLY
    ]
    assert auto_apply_decisions
    assert auto_apply_decisions[-1].preview_result.preview_document is not None
    assert result.refined_document == auto_apply_decisions[-1].preview_result.preview_document
    assert result.report.candidate_stats["by_target_type"]["circle"] >= 1
    assert result.report.decision_stats["auto_accept"] >= 1
    assert result.report.decision_stats["auto_apply"] >= 1
    assert result.refined_document != pipeline_result.document


def test_auto_refinement_pipeline_ellipse_fixture_produces_reviewable_decision() -> None:
    pipeline_result = MinimalPipeline(segment_type="line").run_from_file(
        Path("test_images/ellipse/test_input_ellipse.png"),
        document_id="ellipse_auto_pipeline",
    )

    result = AutoRefinementPipeline().run_from_pipeline_result(
        pipeline_result,
        target_types=("ellipse",),
    )

    assert any(candidate.target_type == "ellipse" for candidate in result.candidates)
    assert any(command["tool"] == "propose_replace_path_with_ellipse" for command in result.proposed_commands)
    assert result.preview_decisions
    assert {decision.decision for decision in result.preview_decisions} <= {"auto_accept", "user_confirm", "reject"}
    assert {
        decision.decision_kind for decision in result.preview_decisions
    } <= {
        DecisionKind.AUTO_APPLY,
        DecisionKind.REQUIRES_EXTERNAL_DECISION,
        DecisionKind.AUTO_REJECT,
    }
    assert any(decision.decision in {"auto_accept", "user_confirm"} for decision in result.preview_decisions)


def test_auto_refinement_pipeline_rectangle_document_generates_line_or_batch_commands() -> None:
    document = _rectangle_document()

    result = AutoRefinementPipeline().run(
        document,
        target_types=("rectangle", "line"),
    )

    tools = {command["tool"] for command in result.proposed_commands}
    assert "propose_replace_segment_with_line" in tools or "propose_batch_refinement" in tools
    assert result.preview_decisions
    assert result.report.command_stats["total"] >= 1
    assert result.report.decision_stats["auto_accept"] + result.report.decision_stats["user_confirm"] + result.report.decision_stats["reject"] == len(result.preview_decisions)


def test_auto_refinement_pipeline_centerline_mode_only_processes_skeleton_paths() -> None:
    document = _mixed_source_rectangle_document()
    result = AutoRefinementPipeline(
        config=AutoRefinementPipelineConfig(processing_contour_source="skeleton"),
    ).run(
        document,
        target_types=("rectangle", "line"),
    )

    assert result.candidates
    assert {candidate.path_id for candidate in result.candidates} == {"skeleton_rect_path"}
    assert {command["path_id"] for command in result.proposed_commands if "path_id" in command} == {"skeleton_rect_path"}
    assert result.report.processing_summary["processing_contour_source"] == "skeleton"
    assert result.report.processing_summary["processed_binary_path_count"] == 0
    assert result.report.processing_summary["processed_skeleton_path_count"] == 1


def test_auto_refinement_pipeline_outline_mode_only_processes_binary_paths() -> None:
    document = _mixed_source_rectangle_document()
    result = AutoRefinementPipeline(
        config=AutoRefinementPipelineConfig(processing_contour_source="binary"),
    ).run(
        document,
        target_types=("rectangle", "line"),
    )

    assert result.candidates
    assert {candidate.path_id for candidate in result.candidates} == {"binary_rect_path"}
    assert {command["path_id"] for command in result.proposed_commands if "path_id" in command} == {"binary_rect_path"}
    assert result.report.processing_summary["processing_contour_source"] == "binary"
    assert result.report.processing_summary["processed_binary_path_count"] == 1
    assert result.report.processing_summary["processed_skeleton_path_count"] == 0


def test_auto_refinement_pipeline_dry_run_only_keeps_original_document_and_serializes() -> None:
    pipeline_result = MinimalPipeline(segment_type="line").run_from_file(
        Path("test_images/circle/test_input_circle.png"),
        document_id="circle_auto_dry_run",
    )

    result = AutoRefinementPipeline(
        config=AutoRefinementPipelineConfig(dry_run_only=True),
    ).run_from_pipeline_result(
        pipeline_result,
        target_types=("circle",),
    )

    payload = json.loads(result.to_json())

    assert result.refined_document == pipeline_result.document
    assert result.report.dry_run_only is True
    assert payload["report"]["dry_run_only"] is True
    assert payload["report"]["score_before"] >= 0.0
    assert payload["report"]["score_after"] >= 0.0
    assert payload["refined_document"]["document_id"] == "circle_auto_dry_run"
    assert payload["report"]["decision_stats"]["auto_apply"] >= 0
    assert payload["preview_decisions"]
    assert payload["preview_decisions"][0]["decision"] in {"auto_accept", "user_confirm", "reject"}
    assert payload["preview_decisions"][0]["decision_kind"] in {
        "auto_apply",
        "requires_external_decision",
        "auto_reject",
    }


def test_auto_refinement_pipeline_includes_self_intersection_feedback_in_next_ai_review_payload() -> None:
    captured_inputs: list[AIReviewInput] = []

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        captured_inputs.append(review_input)
        if len(captured_inputs) == 1:
            return {
                "summary": "First try should be rejected for self intersection.",
                "issues": [],
                "proposed_commands": [
                    {
                        "tool": "propose_replace_path_with_circle",
                        "path_id": "path_1",
                        "reason": "try circle once",
                        "confidence": 0.8,
                        "requires_user_confirmation": True,
                    }
                ],
            }
        return {
            "summary": "Stop after feedback arrives.",
            "issues": [],
            "proposed_commands": [],
        }

    feedback = PolicyFeedback(
        reason_code="self_intersection_increase",
        message="Preview increases self intersections.",
        metrics_delta={"self_intersection_delta": 1},
        policy_hint="avoid repeating this proposal",
        retry_allowed=True,
        retry_constraints={"max_retry_per_target": 2},
        forbidden_repeated_commands=("propose_replace_path_with_circle:path_1",),
    )
    preview_policy = _SequentialRejectingPreviewPolicy(
        [
            (
                PreviewDecision(
                    command={
                        "tool": "propose_replace_path_with_circle",
                        "path_id": "path_1",
                        "reason": "try circle once",
                        "confidence": 0.8,
                        "requires_user_confirmation": True,
                    },
                    preview_result=_policy_preview_result(
                        command_id="cmd_self_intersection",
                        path_id="path_1",
                        self_intersection_after=1,
                    ),
                    decision="reject",
                    reason="Preview increases self intersections.",
                    risk_flags=("self_intersection_increase",),
                    decision_kind=DecisionKind.AUTO_REJECT,
                    risk_level=RiskLevel.MEDIUM_HIGH,
                    policy_feedback=feedback,
                ),
            ),
        ]
    )

    pipeline = AutoRefinementPipeline(
        shape_candidate_detector=_EmptyDetector(),
        proposed_command_planner=_EmptyPlanner(),
        preview_policy=preview_policy,
        ai_review_service=AIReviewService(responder=responder),
        config=AutoRefinementPipelineConfig(max_iterations=2, max_stalled_rounds=2),
    )

    result = pipeline.run_with_ai_review(_rectangle_document())

    assert preview_policy.calls == 1
    assert len(captured_inputs) == 2
    assert captured_inputs[1].policy_feedback
    assert captured_inputs[1].policy_feedback[0]["reason_code"] == "self_intersection_increase"
    assert "propose_replace_path_with_circle:path_1" in captured_inputs[1].forbidden_repeated_commands
    assert result.report.policy_feedback[0]["reason_code"] == "self_intersection_increase"


def test_auto_refinement_pipeline_stops_repeated_low_inlier_ratio_before_infinite_retry() -> None:
    captured_inputs: list[AIReviewInput] = []

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        captured_inputs.append(review_input)
        return {
            "summary": "Keep proposing the same invalid command.",
            "issues": [],
            "proposed_commands": [
                {
                    "tool": "propose_replace_path_with_circle",
                    "path_id": "path_1",
                    "reason": "same retry",
                    "confidence": 0.78,
                    "requires_user_confirmation": True,
                }
            ],
        }

    feedback = PolicyFeedback(
        reason_code="low_inlier_ratio",
        message="Preview inlier ratio is too low.",
        metrics_delta={"inlier_ratio": 0.42},
        policy_hint="do not retry unchanged proposal",
        retry_allowed=True,
        retry_constraints={"min_inlier_ratio": 0.6},
    )
    rejecting_decision = PreviewDecision(
        command={
            "tool": "propose_replace_path_with_circle",
            "path_id": "path_1",
            "reason": "same retry",
            "confidence": 0.78,
            "requires_user_confirmation": True,
        },
        preview_result=_policy_preview_result(command_id="cmd_low_inlier", path_id="path_1"),
        decision="reject",
        reason="Preview inlier ratio is too low.",
        risk_flags=("low_inlier_ratio",),
        decision_kind=DecisionKind.AUTO_REJECT,
        risk_level=RiskLevel.MEDIUM,
        policy_feedback=feedback,
    )
    preview_policy = _SequentialRejectingPreviewPolicy([(rejecting_decision,), (rejecting_decision,)])

    pipeline = AutoRefinementPipeline(
        shape_candidate_detector=_EmptyDetector(),
        proposed_command_planner=_EmptyPlanner(),
        preview_policy=preview_policy,
        ai_review_service=AIReviewService(responder=responder),
        config=AutoRefinementPipelineConfig(
            max_iterations=3,
            max_retry_per_target=2,
            max_stalled_rounds=4,
        ),
    )

    result = pipeline.run_with_ai_review(_rectangle_document())

    assert preview_policy.calls == 2
    assert len(captured_inputs) == 3
    assert any(decision.policy_feedback and decision.policy_feedback.reason_code == "retry_budget_exceeded" for decision in result.preview_decisions)
    assert result.report.rejection_memory
    low_inlier_items = [item for item in result.report.rejection_memory if item["reason_code"] == "low_inlier_ratio"]
    assert low_inlier_items[0]["retry_count"] == 2
    assert "propose_replace_path_with_circle:path_1" in result.report.forbidden_repeated_commands


def test_auto_refinement_pipeline_marks_path_unresolved_after_path_retry_budget() -> None:
    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        return {
            "summary": "Repeat the same path command.",
            "issues": [],
            "proposed_commands": [
                {
                    "tool": "propose_replace_path_with_circle",
                    "path_id": "path_1",
                    "reason": "repeat path command",
                    "confidence": 0.78,
                    "requires_user_confirmation": True,
                }
            ],
        }

    feedback = PolicyFeedback(
        reason_code="low_inlier_ratio",
        message="Preview inlier ratio is too low.",
        metrics_delta={"inlier_ratio": 0.42},
        policy_hint="do not retry unchanged proposal",
        retry_allowed=True,
        retry_constraints={"min_inlier_ratio": 0.6},
    )
    preview_policy = _SequentialRejectingPreviewPolicy(
        [
            (
                PreviewDecision(
                    command={
                        "tool": "propose_replace_path_with_circle",
                        "path_id": "path_1",
                        "reason": "repeat path command",
                        "confidence": 0.78,
                        "requires_user_confirmation": True,
                    },
                    preview_result=_policy_preview_result(command_id="cmd_path_budget", path_id="path_1"),
                    decision="reject",
                    reason="Preview inlier ratio is too low.",
                    risk_flags=("low_inlier_ratio",),
                    decision_kind=DecisionKind.AUTO_REJECT,
                    risk_level=RiskLevel.MEDIUM,
                    policy_feedback=feedback,
                ),
            ),
        ]
    )

    pipeline = AutoRefinementPipeline(
        shape_candidate_detector=_EmptyDetector(),
        proposed_command_planner=_EmptyPlanner(),
        preview_policy=preview_policy,
        ai_review_service=AIReviewService(responder=responder),
        config=AutoRefinementPipelineConfig(
            max_iterations=2,
            max_retry_per_target=5,
            max_retry_per_path=1,
            max_stalled_rounds=3,
        ),
    )

    result = pipeline.run_with_ai_review(_rectangle_document())

    assert preview_policy.calls == 1
    assert result.report.status == EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS.value
    assert "path_1" in result.report.unresolved_targets
    assert any(
        decision.policy_feedback and decision.policy_feedback.reason_code == "path_retry_budget_exceeded"
        for decision in result.preview_decisions
    )


def test_auto_refinement_pipeline_aggregates_path_retry_budget_across_tools() -> None:
    commands_by_round = [
        {
            "tool": "propose_replace_path_with_circle",
            "path_id": "path_1",
            "reason": "round 1 circle",
            "confidence": 0.78,
            "requires_user_confirmation": True,
        },
        {
            "tool": "propose_replace_segment_with_line",
            "path_id": "path_1",
            "segment_range": [0, 0],
            "reason": "round 2 line",
            "confidence": 0.78,
            "requires_user_confirmation": True,
        },
        {
            "tool": "propose_replace_segment_with_arc",
            "path_id": "path_1",
            "segment_range": [0, 0],
            "reason": "round 3 arc",
            "confidence": 0.78,
            "requires_user_confirmation": True,
        },
    ]
    round_index = {"value": 0}

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        index = round_index["value"]
        round_index["value"] += 1
        command = commands_by_round[min(index, len(commands_by_round) - 1)]
        return {
            "summary": "Keep retrying the same path with different tools.",
            "issues": [],
            "proposed_commands": [command],
        }

    feedback = PolicyFeedback(
        reason_code="low_inlier_ratio",
        message="Preview inlier ratio is too low.",
        metrics_delta={"inlier_ratio": 0.42},
        policy_hint="do not retry unchanged proposal",
        retry_allowed=True,
        retry_constraints={"min_inlier_ratio": 0.6},
    )
    preview_policy = _SequentialRejectingPreviewPolicy(
        [
            (
                PreviewDecision(
                    command=commands_by_round[0],
                    preview_result=_policy_preview_result(command_id="cmd_path_circle", path_id="path_1"),
                    decision="reject",
                    reason="Preview inlier ratio is too low.",
                    risk_flags=("low_inlier_ratio",),
                    decision_kind=DecisionKind.AUTO_REJECT,
                    risk_level=RiskLevel.MEDIUM,
                    policy_feedback=feedback,
                ),
            ),
            (
                PreviewDecision(
                    command=commands_by_round[1],
                    preview_result=_policy_preview_result(command_id="cmd_path_line", path_id="path_1"),
                    decision="reject",
                    reason="Preview inlier ratio is too low.",
                    risk_flags=("low_inlier_ratio",),
                    decision_kind=DecisionKind.AUTO_REJECT,
                    risk_level=RiskLevel.MEDIUM,
                    policy_feedback=feedback,
                ),
            ),
        ]
    )

    pipeline = AutoRefinementPipeline(
        shape_candidate_detector=_EmptyDetector(),
        proposed_command_planner=_EmptyPlanner(),
        preview_policy=preview_policy,
        ai_review_service=AIReviewService(responder=responder),
        config=AutoRefinementPipelineConfig(
            max_iterations=3,
            max_retry_per_path=2,
            max_retry_per_target=5,
            max_stalled_rounds=10,
        ),
    )

    result = pipeline.run_with_ai_review(_rectangle_document())

    assert preview_policy.calls == 2
    assert result.report.status == EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS.value
    assert "path_1" in result.report.unresolved_targets
    assert any(
        decision.policy_feedback and decision.policy_feedback.reason_code == "path_retry_budget_exceeded"
        for decision in result.preview_decisions
    )


def test_auto_refinement_pipeline_marks_unresolved_when_stalled_after_prior_feedback() -> None:
    round_index = {"value": 0}

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        index = round_index["value"]
        round_index["value"] += 1
        if index == 0:
            return {
                "summary": "First round proposes one command.",
                "issues": [],
                "proposed_commands": [
                    {
                        "tool": "propose_replace_path_with_circle",
                        "path_id": "path_1",
                        "reason": "first round",
                        "confidence": 0.78,
                        "requires_user_confirmation": True,
                    }
                ],
            }
        return {
            "summary": "Second round has no new proposals.",
            "issues": [],
            "proposed_commands": [],
        }

    feedback = PolicyFeedback(
        reason_code="low_inlier_ratio",
        message="Preview inlier ratio is too low.",
        metrics_delta={"inlier_ratio": 0.42},
        policy_hint="do not retry unchanged proposal",
        retry_allowed=True,
        retry_constraints={"min_inlier_ratio": 0.6},
    )
    preview_policy = _SequentialRejectingPreviewPolicy(
        [
            (
                PreviewDecision(
                    command={
                        "tool": "propose_replace_path_with_circle",
                        "path_id": "path_1",
                        "reason": "first round",
                        "confidence": 0.78,
                        "requires_user_confirmation": True,
                    },
                    preview_result=_policy_preview_result(command_id="cmd_stalled_feedback", path_id="path_1"),
                    decision="reject",
                    reason="Preview inlier ratio is too low.",
                    risk_flags=("low_inlier_ratio",),
                    decision_kind=DecisionKind.AUTO_REJECT,
                    risk_level=RiskLevel.MEDIUM,
                    policy_feedback=feedback,
                ),
            ),
        ]
    )

    pipeline = AutoRefinementPipeline(
        shape_candidate_detector=_EmptyDetector(),
        proposed_command_planner=_EmptyPlanner(),
        preview_policy=preview_policy,
        ai_review_service=AIReviewService(responder=responder),
        config=AutoRefinementPipelineConfig(
            max_iterations=3,
            max_stalled_rounds=5,
        ),
    )

    result = pipeline.run_with_ai_review(_rectangle_document())

    assert result.report.policy_feedback
    assert result.report.status == EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS.value


def test_auto_refinement_pipeline_emits_ai_command_processing_progress_events() -> None:
    events: list[dict[str, object]] = []

    def responder(prompt: str, review_input: AIReviewInput) -> dict[str, object]:
        return {
            "summary": "One AI proposal.",
            "issues": [{"issue_id": "issue_1", "category": "topology", "severity": "low", "summary": "check"}],
            "proposed_commands": [
                {
                    "tool": "propose_replace_path_with_circle",
                    "path_id": "path_1",
                    "reason": "try circle",
                    "confidence": 0.78,
                    "requires_user_confirmation": True,
                }
            ],
        }

    feedback = PolicyFeedback(
        reason_code="low_inlier_ratio",
        message="Preview inlier ratio is too low.",
        metrics_delta={"inlier_ratio": 0.42},
        policy_hint="do not retry unchanged proposal",
        retry_allowed=True,
        retry_constraints={"min_inlier_ratio": 0.6},
    )
    preview_policy = _SequentialRejectingPreviewPolicy(
        [
            (
                PreviewDecision(
                    command={
                        "tool": "propose_replace_path_with_circle",
                        "path_id": "path_1",
                        "reason": "try circle",
                        "confidence": 0.78,
                        "requires_user_confirmation": True,
                    },
                    preview_result=_policy_preview_result(command_id="cmd_progress", path_id="path_1"),
                    decision="reject",
                    reason="Preview inlier ratio is too low.",
                    risk_flags=("low_inlier_ratio",),
                    decision_kind=DecisionKind.AUTO_REJECT,
                    risk_level=RiskLevel.MEDIUM,
                    policy_feedback=feedback,
                ),
            ),
        ]
    )

    pipeline = AutoRefinementPipeline(
        shape_candidate_detector=_EmptyDetector(),
        proposed_command_planner=_EmptyPlanner(),
        preview_policy=preview_policy,
        ai_review_service=AIReviewService(responder=responder),
        config=AutoRefinementPipelineConfig(max_iterations=1, max_stalled_rounds=1),
    )
    pipeline.set_progress_callback(events.append)

    result = pipeline.run_with_ai_review(_rectangle_document())

    stages = [event["stage"] for event in events]
    for required in (
        "ai_response_normalized",
        "ai_commands_merged",
        "retry_budget_done",
        "preview_policy_start",
        "preview_policy_done",
        "document_update_done",
        "ai_review_round_done",
    ):
        assert required in stages

    normalized_event = next(event for event in events if event["stage"] == "ai_response_normalized")
    assert normalized_event["summary_length"] == len("One AI proposal.")
    assert normalized_event["issue_count"] == 1
    assert normalized_event["ai_proposed_command_count"] == 1

    merged_event = next(event for event in events if event["stage"] == "ai_commands_merged")
    assert merged_event["algorithm_command_count"] == 0
    assert merged_event["ai_command_count"] == 1
    assert merged_event["merged_command_count"] == 1
    assert merged_event["limited_command_count"] == 1

    retry_event = next(event for event in events if event["stage"] == "retry_budget_done")
    assert retry_event["blocked_count"] == 0
    assert retry_event["executable_count"] == 1
    assert retry_event["unresolved_target_count"] == 0

    preview_done_event = next(event for event in events if event["stage"] == "preview_policy_done")
    assert preview_done_event["decision_count"] == 1
    assert preview_done_event["auto_apply_count"] == 0
    assert preview_done_event["auto_reject_count"] == 1
    assert preview_done_event["requires_external_decision_count"] == 0
    assert preview_done_event["accepted_count"] == 0
    assert preview_done_event["rejected_count"] == 1
    assert preview_done_event["user_confirm_count"] == 0

    update_event = next(event for event in events if event["stage"] == "document_update_done")
    assert update_event["applied_command_count"] == 0
    assert update_event["document_changed"] is False
    assert update_event["score_before"] == 4.0
    assert update_event["score_after"] == 4.0
    assert update_event["improvement"] == 0.0
    assert update_event["stalled_rounds"] == 1

    round_done_event = next(event for event in events if event["stage"] == "ai_review_round_done")
    assert round_done_event["iteration"] == 1
    assert round_done_event["final_status"] == EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS.value
    assert round_done_event["unresolved_target_count"] == 0
    assert result.report.status == EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS.value
