from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from core.types import ShapeCandidateTargetType, VectorDocument
from services.document_integrity import DocumentIntegrityValidator, IntegrityReport
from services.engine_protocol import DecisionKind
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


class AutoRefinementPipeline:
    def __init__(
        self,
        *,
        shape_candidate_detector: ShapeCandidateDetector | None = None,
        proposed_command_planner: ProposedCommandPlanner | None = None,
        preview_policy: PreviewAndAutoAcceptPolicy | None = None,
        scorer: Scorer | None = None,
        integrity_validator: DocumentIntegrityValidator | None = None,
        json_exporter: JsonExporter | None = None,
        config: AutoRefinementPipelineConfig | None = None,
    ) -> None:
        self.shape_candidate_detector = shape_candidate_detector or ShapeCandidateDetector()
        self.proposed_command_planner = proposed_command_planner or ProposedCommandPlanner()
        self.preview_policy = preview_policy or PreviewAndAutoAcceptPolicy()
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
        )


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


__all__ = [
    "AutoRefinementPipeline",
    "AutoRefinementPipelineConfig",
    "AutoRefinementPipelineResult",
    "AutoRefinementReport",
]
