from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from services.ai_agent import AIReviewInput, AIReviewService
from services.command_preview import CommandPreviewResult, ConstraintChangeSummary, ExportImpactSummary
from core.types import ShapeCandidateTargetType, VectorDocument
from services.document_integrity import DocumentIntegrityValidator, IntegrityReport
from services.engine_protocol import (
    DecisionKind,
    EngineStatus,
    ExternalDecisionRecord,
    ExternalDecisionRequest,
    PolicyFeedback,
    RejectionMemoryItem,
    RiskLevel,
)
from services.json_exporter import JsonExporter
from services.minimal_pipeline import MinimalPipelineResult
from services.preview_auto_accept_policy import PreviewAndAutoAcceptPolicy, PreviewDecision, PreviewPolicyResult
from services.proposed_command_planner import ProposedCommandPlanner
from services.scorer import Scorer
from services.shape_candidate_detector import ShapeCandidate, ShapeCandidateDetector


@dataclass(frozen=True, slots=True)
class AutoRefinementPipelineConfig:
    target_types: tuple[ShapeCandidateTargetType, ...] = ()
    dry_run_only: bool = False
    evaluate_batch_commands: bool = False
    max_iterations: int = 3
    max_proposals_per_round: int = 8
    improvement_epsilon: float = 0.1
    max_retry_per_target: int = 2
    max_retry_per_path: int = 3
    max_stalled_rounds: int = 1


@dataclass(frozen=True, slots=True)
class AutoRefinementReport:
    candidate_stats: dict[str, Any]
    command_stats: dict[str, Any]
    decision_stats: dict[str, int]
    score_before: float
    score_after: float
    integrity: dict[str, Any]
    dry_run_only: bool
    target_types: tuple[str, ...] = ()
    status: str = EngineStatus.COMPLETED.value
    iteration_count: int = 1
    policy_feedback: tuple[dict[str, Any], ...] = ()
    rejection_memory: tuple[dict[str, Any], ...] = ()
    forbidden_repeated_commands: tuple[str, ...] = ()
    unresolved_targets: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_stats": json.loads(json.dumps(self.candidate_stats)),
            "command_stats": json.loads(json.dumps(self.command_stats)),
            "decision_stats": dict(self.decision_stats),
            "score_before": self.score_before,
            "score_after": self.score_after,
            "integrity": json.loads(json.dumps(self.integrity)),
            "dry_run_only": self.dry_run_only,
            "target_types": list(self.target_types),
            "status": self.status,
            "iteration_count": self.iteration_count,
            "policy_feedback": json.loads(json.dumps(self.policy_feedback)),
            "rejection_memory": json.loads(json.dumps(self.rejection_memory)),
            "forbidden_repeated_commands": list(self.forbidden_repeated_commands),
            "unresolved_targets": list(self.unresolved_targets),
        }


@dataclass(frozen=True, slots=True)
class AutoRefinementPipelineResult:
    refined_document: VectorDocument
    candidates: tuple[ShapeCandidate, ...]
    proposed_commands: tuple[dict[str, Any], ...]
    preview_decisions: tuple[PreviewDecision, ...]
    report: AutoRefinementReport

    def to_dict(self, *, json_exporter: JsonExporter | None = None) -> dict[str, Any]:
        exporter = json_exporter or JsonExporter()
        return {
            "refined_document": exporter.export_to_dict(self.refined_document),
            "candidates": [_candidate_to_dict(candidate) for candidate in self.candidates],
            "proposed_commands": [json.loads(json.dumps(command)) for command in self.proposed_commands],
            "preview_decisions": [_decision_to_dict(decision) for decision in self.preview_decisions],
            "report": self.report.to_dict(),
        }

    def to_json(self, *, json_exporter: JsonExporter | None = None) -> str:
        return json.dumps(self.to_dict(json_exporter=json_exporter), indent=2, sort_keys=True)

    def external_decision_records(self) -> tuple[ExternalDecisionRecord, ...]:
        records: list[ExternalDecisionRecord] = []
        for decision in self.preview_decisions:
            request = decision.external_decision_request
            if request is None:
                continue
            preview_document = getattr(decision.preview_result, "preview_document", None)
            records.append(
                ExternalDecisionRecord(
                    request=request,
                    preview_document=preview_document,
                )
            )
        return tuple(records)


class AutoRefinementPipeline:
    def __init__(
        self,
        *,
        shape_candidate_detector: ShapeCandidateDetector | None = None,
        proposed_command_planner: ProposedCommandPlanner | None = None,
        preview_policy: PreviewAndAutoAcceptPolicy | None = None,
        ai_review_service: AIReviewService | None = None,
        scorer: Scorer | None = None,
        integrity_validator: DocumentIntegrityValidator | None = None,
        json_exporter: JsonExporter | None = None,
        config: AutoRefinementPipelineConfig | None = None,
    ) -> None:
        self.shape_candidate_detector = shape_candidate_detector or ShapeCandidateDetector()
        self.proposed_command_planner = proposed_command_planner or ProposedCommandPlanner()
        self.preview_policy = preview_policy or PreviewAndAutoAcceptPolicy()
        self.ai_review_service = ai_review_service
        self.scorer = scorer or Scorer()
        self.integrity_validator = integrity_validator or DocumentIntegrityValidator()
        self.json_exporter = json_exporter or JsonExporter()
        self.config = config or AutoRefinementPipelineConfig()

    def run_from_pipeline_result(
        self,
        pipeline_result: MinimalPipelineResult,
        *,
        target_types: tuple[ShapeCandidateTargetType, ...] | list[ShapeCandidateTargetType] | None = None,
        dry_run_only: bool | None = None,
    ) -> AutoRefinementPipelineResult:
        return self.run(
            pipeline_result.document,
            target_types=target_types,
            dry_run_only=dry_run_only,
        )

    def run_from_pipeline_result_with_ai_review(
        self,
        pipeline_result: MinimalPipelineResult,
        *,
        target_types: tuple[ShapeCandidateTargetType, ...] | list[ShapeCandidateTargetType] | None = None,
        dry_run_only: bool | None = None,
    ) -> AutoRefinementPipelineResult:
        return self.run_with_ai_review(
            pipeline_result.document,
            target_types=target_types,
            dry_run_only=dry_run_only,
        )

    def run(
        self,
        document: VectorDocument,
        *,
        target_types: tuple[ShapeCandidateTargetType, ...] | list[ShapeCandidateTargetType] | None = None,
        dry_run_only: bool | None = None,
    ) -> AutoRefinementPipelineResult:
        selected_target_types = tuple(target_types) if target_types is not None else self.config.target_types
        effective_dry_run = self.config.dry_run_only if dry_run_only is None else bool(dry_run_only)

        score_before = self.scorer.score_document(document).total_score
        detected_candidates = self.shape_candidate_detector.detect_candidates(document)
        candidates = self._filter_candidates(detected_candidates, selected_target_types)
        proposed_commands = self.proposed_command_planner.plan_commands(document, candidates)
        evaluation_commands = self._evaluation_commands(proposed_commands)
        policy_result = self.preview_policy.evaluate_commands(evaluation_commands, document)

        refined_document = document if effective_dry_run else policy_result.final_document
        score_after = self.scorer.score_document(refined_document).total_score
        integrity_report = self.integrity_validator.validate(refined_document)
        report = self._build_report(
            candidates=candidates,
            proposed_commands=proposed_commands,
            preview_result=policy_result,
            score_before=score_before,
            score_after=score_after,
            integrity_report=integrity_report,
            dry_run_only=effective_dry_run,
            target_types=selected_target_types,
        )
        return AutoRefinementPipelineResult(
            refined_document=refined_document,
            candidates=candidates,
            proposed_commands=proposed_commands,
            preview_decisions=policy_result.decisions,
            report=report,
        )

    def run_with_ai_review(
        self,
        document: VectorDocument,
        *,
        target_types: tuple[ShapeCandidateTargetType, ...] | list[ShapeCandidateTargetType] | None = None,
        dry_run_only: bool | None = None,
    ) -> AutoRefinementPipelineResult:
        if self.ai_review_service is None:
            raise RuntimeError("AI review service is not configured")

        selected_target_types = tuple(target_types) if target_types is not None else self.config.target_types
        effective_dry_run = self.config.dry_run_only if dry_run_only is None else bool(dry_run_only)
        initial_document = document
        current_document = document
        current_score = self.scorer.score_document(document).total_score
        latest_candidates: tuple[ShapeCandidate, ...] = ()
        considered_commands: list[dict[str, Any]] = []
        all_decisions: list[PreviewDecision] = []
        collected_feedback: list[PolicyFeedback] = []
        rejection_memory: dict[tuple[str, str, str], RejectionMemoryItem] = {}
        forbidden_repeated_commands: set[str] = set()
        unresolved_targets: set[str] = set()
        stalled_rounds = 0
        iteration_count = 0
        final_status = EngineStatus.COMPLETED

        for iteration in range(1, self.config.max_iterations + 1):
            iteration_count = iteration
            detected_candidates = self.shape_candidate_detector.detect_candidates(current_document)
            latest_candidates = self._filter_candidates(detected_candidates, selected_target_types)
            algorithm_commands = self.proposed_command_planner.plan_commands(current_document, latest_candidates)
            review_input = self._build_ai_review_input(
                document=current_document,
                candidates=latest_candidates,
                algorithm_commands=algorithm_commands,
                policy_feedback=tuple(collected_feedback),
                rejection_memory=tuple(rejection_memory.values()),
                forbidden_repeated_commands=tuple(sorted(forbidden_repeated_commands)),
            )
            ai_review_output = self.ai_review_service.run_review(review_input)
            candidate_commands = tuple(dict(command) for command in ai_review_output.proposed_commands)
            proposed_commands = self._limit_commands(
                tuple(dict(command) for command in algorithm_commands) + candidate_commands
            )
            considered_commands.extend(dict(command) for command in proposed_commands)

            blocked_decisions, executable_commands, newly_forbidden, newly_unresolved = self._apply_retry_budget(
                proposed_commands,
                rejection_memory=tuple(rejection_memory.values()),
                forbidden_repeated_commands=tuple(sorted(forbidden_repeated_commands)),
            )
            forbidden_repeated_commands.update(newly_forbidden)
            unresolved_targets.update(newly_unresolved)

            policy_result = self.preview_policy.evaluate_commands(
                self._evaluation_commands(executable_commands),
                current_document,
            ) if executable_commands else PreviewPolicyResult(
                final_document=current_document,
                decisions=(),
                accepted_count=0,
                rejected_count=0,
                user_confirm_count=0,
            )
            round_decisions = tuple(blocked_decisions) + policy_result.decisions
            all_decisions.extend(round_decisions)

            round_feedback = tuple(
                decision.policy_feedback
                for decision in round_decisions
                if decision.policy_feedback is not None and decision.decision_kind != DecisionKind.AUTO_APPLY
            )
            collected_feedback.extend(round_feedback)
            for feedback in round_feedback:
                forbidden_repeated_commands.update(feedback.forbidden_repeated_commands)

            rejection_memory = self._update_rejection_memory(rejection_memory, round_decisions)
            current_document = current_document if effective_dry_run else policy_result.final_document
            next_score = self.scorer.score_document(current_document).total_score
            improvement = current_score - next_score

            if improvement > self.config.improvement_epsilon:
                current_score = next_score
                stalled_rounds = 0
            else:
                current_score = next_score
                stalled_rounds += 1

            if unresolved_targets:
                final_status = EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS
            if stalled_rounds >= self.config.max_stalled_rounds:
                if round_feedback or unresolved_targets:
                    final_status = EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS
                break
            if not candidate_commands and not algorithm_commands:
                break

        if collected_feedback and final_status == EngineStatus.COMPLETED:
            final_status = EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS

        integrity_report = self.integrity_validator.validate(current_document)
        report = self._build_report(
            candidates=latest_candidates,
            proposed_commands=tuple(considered_commands),
            preview_result=PreviewPolicyResult(
                final_document=current_document,
                decisions=tuple(all_decisions),
                accepted_count=sum(1 for decision in all_decisions if decision.decision_kind == DecisionKind.AUTO_APPLY),
                rejected_count=sum(1 for decision in all_decisions if decision.decision_kind == DecisionKind.AUTO_REJECT),
                user_confirm_count=sum(
                    1
                    for decision in all_decisions
                    if decision.decision_kind == DecisionKind.REQUIRES_EXTERNAL_DECISION
                ),
            ),
            score_before=self.scorer.score_document(initial_document).total_score,
            score_after=current_score,
            integrity_report=integrity_report,
            dry_run_only=effective_dry_run,
            target_types=selected_target_types,
            status=final_status.value,
            iteration_count=iteration_count,
            policy_feedback=tuple(collected_feedback),
            rejection_memory=tuple(rejection_memory.values()),
            forbidden_repeated_commands=tuple(sorted(forbidden_repeated_commands)),
            unresolved_targets=tuple(sorted(unresolved_targets)),
        )
        return AutoRefinementPipelineResult(
            refined_document=current_document,
            candidates=latest_candidates,
            proposed_commands=tuple(considered_commands),
            preview_decisions=tuple(all_decisions),
            report=report,
        )

    def _filter_candidates(
        self,
        candidates: tuple[ShapeCandidate, ...],
        target_types: tuple[ShapeCandidateTargetType, ...],
    ) -> tuple[ShapeCandidate, ...]:
        if not target_types:
            return candidates
        allowed = set(target_types)
        return tuple(candidate for candidate in candidates if candidate.target_type in allowed)

    def _evaluation_commands(
        self,
        commands: tuple[dict[str, Any], ...],
    ) -> tuple[dict[str, Any], ...]:
        if self.config.evaluate_batch_commands:
            return commands
        return tuple(command for command in commands if command.get("tool") != "propose_batch_refinement")

    def _build_report(
        self,
        *,
        candidates: tuple[ShapeCandidate, ...],
        proposed_commands: tuple[dict[str, Any], ...],
        preview_result: PreviewPolicyResult,
        score_before: float,
        score_after: float,
        integrity_report: IntegrityReport,
        dry_run_only: bool,
        target_types: tuple[ShapeCandidateTargetType, ...],
        status: str = EngineStatus.COMPLETED.value,
        iteration_count: int = 1,
        policy_feedback: tuple[PolicyFeedback, ...] = (),
        rejection_memory: tuple[RejectionMemoryItem, ...] = (),
        forbidden_repeated_commands: tuple[str, ...] = (),
        unresolved_targets: tuple[str, ...] = (),
    ) -> AutoRefinementReport:
        candidate_stats = {
            "total": len(candidates),
            "by_target_type": _count_by(candidates, lambda candidate: candidate.target_type),
            "by_source": _count_by(candidates, lambda candidate: candidate.source),
        }
        command_stats = {
            "total": len(proposed_commands),
            "evaluated_total": len(preview_result.decisions),
            "by_tool": _count_by(proposed_commands, lambda command: str(command["tool"])),
            "batch_count": sum(1 for command in proposed_commands if command.get("tool") == "propose_batch_refinement"),
        }
        decision_stats = {
            "auto_accept": preview_result.accepted_count,
            "user_confirm": preview_result.user_confirm_count,
            "reject": preview_result.rejected_count,
            "auto_apply": sum(1 for decision in preview_result.decisions if decision.decision_kind == DecisionKind.AUTO_APPLY),
            "requires_external_decision": sum(
                1
                for decision in preview_result.decisions
                if decision.decision_kind == DecisionKind.REQUIRES_EXTERNAL_DECISION
            ),
            "auto_reject": sum(1 for decision in preview_result.decisions if decision.decision_kind == DecisionKind.AUTO_REJECT),
        }
        integrity = {
            "success": integrity_report.success,
            "error_count": len(integrity_report.errors),
            "warning_count": len(integrity_report.warnings),
            "affected_ids": list(integrity_report.affected_ids),
            "errors": [_integrity_issue_to_dict(issue) for issue in integrity_report.errors],
            "warnings": [_integrity_issue_to_dict(issue) for issue in integrity_report.warnings],
        }
        return AutoRefinementReport(
            candidate_stats=candidate_stats,
            command_stats=command_stats,
            decision_stats=decision_stats,
            score_before=score_before,
            score_after=score_after,
            integrity=integrity,
            dry_run_only=dry_run_only,
            target_types=tuple(target_types),
            status=status,
            iteration_count=iteration_count,
            policy_feedback=tuple(feedback.to_dict() for feedback in policy_feedback),
            rejection_memory=tuple(item.to_dict() for item in rejection_memory),
            forbidden_repeated_commands=tuple(forbidden_repeated_commands),
            unresolved_targets=tuple(unresolved_targets),
        )

    def _build_ai_review_input(
        self,
        *,
        document: VectorDocument,
        candidates: tuple[ShapeCandidate, ...],
        algorithm_commands: tuple[dict[str, Any], ...],
        policy_feedback: tuple[PolicyFeedback, ...],
        rejection_memory: tuple[RejectionMemoryItem, ...],
        forbidden_repeated_commands: tuple[str, ...],
    ) -> AIReviewInput:
        score = self.scorer.score_document(document).total_score
        return AIReviewInput(
            original_image=None,
            overlay_image=None,
            distance_field_diff_image=None,
            vector_document_json=self.json_exporter.export_to_dict(document),
            candidates=tuple(_candidate_to_dict(candidate) for candidate in candidates),
            proposed_commands_from_algorithm=tuple(json.loads(json.dumps(command)) for command in algorithm_commands),
            preview_summary={
                "score": score,
                "rejected_count": sum(1 for item in policy_feedback if item.reason_code != "auto_apply"),
                "forbidden_repeated_commands": list(forbidden_repeated_commands),
            },
            policy_feedback=tuple(feedback.to_dict() for feedback in policy_feedback),
            rejection_memory=tuple(item.to_dict() for item in rejection_memory),
            forbidden_repeated_commands=tuple(forbidden_repeated_commands),
            retry_budget={
                "max_iterations": self.config.max_iterations,
                "max_proposals_per_round": self.config.max_proposals_per_round,
                "improvement_epsilon": self.config.improvement_epsilon,
                "max_retry_per_target": self.config.max_retry_per_target,
                "max_retry_per_path": self.config.max_retry_per_path,
            },
            fit_error=score,
            complexity_score=score,
            topology_status=_aggregate_topology_status(document),
            self_intersection_count=_aggregate_self_intersection_count(document),
            coordinate_system=self.json_exporter.export_to_dict(document)["coordinate_system"],
        )

    def _limit_commands(
        self,
        commands: tuple[dict[str, Any], ...],
    ) -> tuple[dict[str, Any], ...]:
        return commands[: self.config.max_proposals_per_round]

    def _apply_retry_budget(
        self,
        commands: tuple[dict[str, Any], ...],
        *,
        rejection_memory: tuple[RejectionMemoryItem, ...],
        forbidden_repeated_commands: tuple[str, ...],
    ) -> tuple[tuple[PreviewDecision, ...], tuple[dict[str, Any], ...], tuple[str, ...], tuple[str, ...]]:
        memory_by_target_tool: dict[tuple[str, str], int] = {}
        memory_by_path: dict[str, int] = {}
        for item in rejection_memory:
            memory_by_target_tool[(item.target, item.tool)] = max(
                memory_by_target_tool.get((item.target, item.tool), 0),
                item.retry_count,
            )
            path_id = _command_path_from_target(item.target)
            if path_id:
                memory_by_path[path_id] = memory_by_path.get(path_id, 0) + item.retry_count

        blocked: list[PreviewDecision] = []
        executable: list[dict[str, Any]] = []
        newly_forbidden: set[str] = set(forbidden_repeated_commands)
        unresolved_targets: set[str] = set()
        forbidden_signatures = set(forbidden_repeated_commands)

        for command in commands:
            signature = _command_signature(command)
            target = _command_target(command)
            path_id = _command_path_id(command)
            tool = str(command.get("tool", "unknown_tool"))
            retry_count = memory_by_target_tool.get((target, tool), 0)
            path_retry_count = memory_by_path.get(path_id, 0) if path_id is not None else 0

            if signature in forbidden_signatures or retry_count >= self.config.max_retry_per_target:
                newly_forbidden.add(signature)
                blocked.append(
                    _synthetic_decision(
                        command=command,
                        decision_kind=DecisionKind.AUTO_REJECT,
                        risk_level=RiskLevel.MEDIUM_HIGH,
                        reason_code="retry_budget_exceeded",
                        message="Retry budget exceeded for target + tool.",
                        forbidden_repeated_commands=(signature,),
                    )
                )
                continue
            if path_id is not None and path_retry_count >= self.config.max_retry_per_path:
                unresolved_targets.add(path_id)
                blocked.append(
                    _synthetic_decision(
                        command=command,
                        decision_kind=DecisionKind.REQUIRES_EXTERNAL_DECISION,
                        risk_level=RiskLevel.MEDIUM_HIGH,
                        reason_code="path_retry_budget_exceeded",
                        message="Path retry budget exceeded; mark region unresolved.",
                        forbidden_repeated_commands=(signature,),
                    )
                )
                newly_forbidden.add(signature)
                continue
            executable.append(dict(command))
        return tuple(blocked), tuple(executable), tuple(sorted(newly_forbidden)), tuple(sorted(unresolved_targets))

    def _update_rejection_memory(
        self,
        existing: dict[tuple[str, str, str], RejectionMemoryItem],
        decisions: tuple[PreviewDecision, ...],
    ) -> dict[tuple[str, str, str], RejectionMemoryItem]:
        updated = dict(existing)
        for decision in decisions:
            if decision.decision_kind == DecisionKind.AUTO_APPLY or decision.policy_feedback is None:
                continue
            target = _command_target(decision.command)
            tool = str(decision.command.get("tool", "unknown_tool"))
            reason_code = decision.policy_feedback.reason_code
            key = (target, tool, reason_code)
            current = updated.get(key)
            retry_count = 1 if current is None else current.retry_count + 1
            updated[key] = RejectionMemoryItem(
                target=target,
                tool=tool,
                reason_code=reason_code,
                retry_count=retry_count,
                last_metrics_delta=decision.policy_feedback.metrics_delta,
            )
        return updated


def _count_by(values: tuple[object, ...] | list[object], key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(key_fn(value))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _candidate_to_dict(candidate: ShapeCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "target_type": candidate.target_type,
        "path_id": candidate.path_id,
        "segment_range": list(candidate.segment_range),
        "source": candidate.source,
        "confidence": candidate.confidence,
        "evidence": json.loads(json.dumps(candidate.evidence)),
        "reason": candidate.reason,
    }


def _decision_to_dict(decision: PreviewDecision) -> dict[str, Any]:
    preview_result = decision.preview_result
    return {
        "command": json.loads(json.dumps(decision.command)),
        "decision": decision.decision,
        "decision_kind": None if decision.decision_kind is None else decision.decision_kind.value,
        "reason": decision.reason,
        "risk_flags": list(decision.risk_flags),
        "risk_level": decision.risk_level.value,
        "policy_feedback": None if decision.policy_feedback is None else decision.policy_feedback.to_dict(),
        "external_decision_request": None
        if decision.external_decision_request is None
        else decision.external_decision_request.to_dict(),
        "policy_result": None if decision.policy_result is None else decision.policy_result.to_dict(),
        "preview_result": _preview_result_to_dict(preview_result),
    }


def _preview_result_to_dict(preview_result: object) -> dict[str, Any]:
    if isinstance(preview_result, PreviewPolicyResult):
        raise TypeError("unexpected preview policy result")
    if hasattr(preview_result, "previews") and hasattr(preview_result, "success_count"):
        batch_preview = preview_result
        return {
            "type": "batch",
            "batch_id": getattr(batch_preview, "batch_id"),
            "success_count": getattr(batch_preview, "success_count"),
            "failure_count": getattr(batch_preview, "failure_count"),
            "previews": [_preview_result_to_dict(item) for item in getattr(batch_preview, "previews")],
        }
    single_preview = preview_result
    return {
        "type": "single",
        "success": getattr(single_preview, "success"),
        "command_id": getattr(single_preview, "command_id"),
        "reason": getattr(single_preview, "reason"),
        "old_score": getattr(single_preview, "old_score"),
        "predicted_new_score": getattr(single_preview, "predicted_new_score"),
        "score_delta": getattr(single_preview, "score_delta"),
        "affected_paths": list(getattr(single_preview, "affected_paths")),
        "affected_segments": list(getattr(single_preview, "affected_segments")),
        "topology_status_before": dict(getattr(single_preview, "topology_status_before")),
        "topology_status_after": dict(getattr(single_preview, "topology_status_after")),
        "self_intersection_count_before": dict(getattr(single_preview, "self_intersection_count_before")),
        "self_intersection_count_after": dict(getattr(single_preview, "self_intersection_count_after")),
        "segment_type_summary": json.loads(json.dumps(getattr(single_preview, "segment_type_summary"))),
    }


def _integrity_issue_to_dict(issue: Any) -> dict[str, Any]:
    return {
        "code": issue.code,
        "message": issue.message,
        "affected_ids": list(issue.affected_ids),
    }


def _aggregate_topology_status(document: VectorDocument) -> str:
    statuses = {path.topology_status for path in document.paths if path.topology_status}
    if "topology_error" in statuses:
        return "topology_error"
    if "open" in statuses:
        return "open"
    return "closed"


def _aggregate_self_intersection_count(document: VectorDocument) -> int:
    total = 0
    for path in document.paths:
        total += int(path.self_intersection_count or 0)
    return total


def _command_signature(command: dict[str, Any]) -> str:
    return f"{command.get('tool', 'unknown_tool')}:{_command_target(command)}"


def _command_target(command: dict[str, Any]) -> str:
    path_id = command.get("path_id")
    segment_range = command.get("segment_range")
    if path_id is not None and segment_range is not None:
        return f"{path_id}:{list(segment_range)}"
    if path_id is not None:
        return str(path_id)
    candidate_id = command.get("candidate_id")
    if candidate_id is not None:
        return str(candidate_id)
    return "unknown_target"


def _command_path_id(command: dict[str, Any]) -> str | None:
    path_id = command.get("path_id")
    if path_id is None:
        return None
    return str(path_id)


def _command_path_from_target(target: str) -> str | None:
    if ":" not in target:
        return target or None
    return target.split(":", 1)[0] or None


def _synthetic_decision(
    *,
    command: dict[str, Any],
    decision_kind: DecisionKind,
    risk_level: RiskLevel,
    reason_code: str,
    message: str,
    forbidden_repeated_commands: tuple[str, ...] = (),
) -> PreviewDecision:
    feedback = PolicyFeedback(
        reason_code=reason_code,
        message=message,
        metrics_delta={},
        policy_hint="do not repeat the same rejected proposal",
        retry_allowed=False,
        retry_constraints={},
        forbidden_repeated_commands=forbidden_repeated_commands,
    )
    external_request = None
    if decision_kind == DecisionKind.REQUIRES_EXTERNAL_DECISION:
        external_request = ExternalDecisionRequest(
            decision_id=f"{command.get('command_id', 'retry_budget')}:external_decision",
            reason=message,
            risk_flags=(reason_code,),
            available_actions=("reject", "defer"),
            command=dict(command),
            candidate_id=command.get("candidate_id"),
            preview_summary={},
            policy_feedback=feedback,
        )
    return PreviewDecision(
        command=dict(command),
        preview_result=CommandPreviewResult(
            success=False,
            command_id=str(command.get("command_id", command.get("tool", "unknown_command"))),
            reason=message,
            old_score=None,
            predicted_new_score=None,
            score_delta=None,
            affected_paths=tuple(filter(None, (_command_path_id(command),))),
            affected_segments=(),
            topology_status_before={},
            topology_status_after={},
            self_intersection_count_before={},
            self_intersection_count_after={},
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
        ),
        decision_kind=decision_kind,
        reason=message,
        risk_flags=(reason_code,),
        risk_level=risk_level,
        policy_feedback=feedback,
        external_decision_request=external_request,
    )


__all__ = [
    "AutoRefinementPipeline",
    "AutoRefinementPipelineConfig",
    "AutoRefinementPipelineResult",
    "AutoRefinementReport",
]
