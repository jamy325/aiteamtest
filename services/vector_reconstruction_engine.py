from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from core.types import ShapeCandidateTargetType, VectorDocument
from services.ai_agent import AIReviewService
from services.auto_refinement_pipeline import (
    AutoRefinementPipeline,
    AutoRefinementPipelineConfig,
    AutoRefinementPipelineResult,
)
from services.engine_protocol import (
    AutonomyLevel,
    DecisionKind,
    DecisionPolicyResult,
    EngineResult,
    EngineStatus,
    ExternalDecisionRequest,
    PolicyFeedback,
    RejectionMemoryItem,
)
from services.minimal_pipeline import MinimalPipeline, MinimalPipelineResult
from services.preview_auto_accept_policy import PreviewAndAutoAcceptPolicy


@dataclass(frozen=True, slots=True)
class VectorReconstructionEngineConfig:
    target_types: tuple[ShapeCandidateTargetType, ...] = ()
    autonomy_level: AutonomyLevel = AutonomyLevel.AUTONOMOUS_SAFE
    max_iterations: int = 3
    dry_run_only: bool = False
    enable_ai_review: bool = False
    max_proposals_per_round: int = 8
    improvement_epsilon: float = 0.1
    max_retry_per_target: int = 2
    max_retry_per_path: int = 3
    max_stalled_rounds: int = 1
    evaluate_batch_commands: bool = False
    document_id: str = "document_1"


class VectorReconstructionEngine:
    def __init__(
        self,
        *,
        minimal_pipeline: MinimalPipeline | None = None,
        auto_refinement_pipeline: AutoRefinementPipeline | Any | None = None,
        ai_review_service: AIReviewService | None = None,
        config: VectorReconstructionEngineConfig | None = None,
    ) -> None:
        self.minimal_pipeline = minimal_pipeline or MinimalPipeline()
        self.auto_refinement_pipeline = auto_refinement_pipeline or AutoRefinementPipeline(
            ai_review_service=ai_review_service,
        )
        self.ai_review_service = ai_review_service
        self.config = config or VectorReconstructionEngineConfig()

    def run_document(
        self,
        document: VectorDocument,
        *,
        target_types: tuple[ShapeCandidateTargetType, ...] | list[ShapeCandidateTargetType] | None = None,
        autonomy_level: AutonomyLevel | None = None,
        max_iterations: int | None = None,
        dry_run_only: bool | None = None,
        enable_ai_review: bool | None = None,
    ) -> EngineResult:
        runtime_config = self._runtime_config(
            target_types=target_types,
            autonomy_level=autonomy_level,
            max_iterations=max_iterations,
            dry_run_only=dry_run_only,
            enable_ai_review=enable_ai_review,
        )
        pipeline = self._configured_pipeline(runtime_config)
        result = self._run_pipeline(
            pipeline,
            document=document,
            pipeline_result=None,
            runtime_config=runtime_config,
        )
        return self._to_engine_result(result, runtime_config)

    def run_pipeline_result(
        self,
        pipeline_result: MinimalPipelineResult,
        *,
        target_types: tuple[ShapeCandidateTargetType, ...] | list[ShapeCandidateTargetType] | None = None,
        autonomy_level: AutonomyLevel | None = None,
        max_iterations: int | None = None,
        dry_run_only: bool | None = None,
        enable_ai_review: bool | None = None,
    ) -> EngineResult:
        runtime_config = self._runtime_config(
            target_types=target_types,
            autonomy_level=autonomy_level,
            max_iterations=max_iterations,
            dry_run_only=dry_run_only,
            enable_ai_review=enable_ai_review,
        )
        pipeline = self._configured_pipeline(runtime_config)
        result = self._run_pipeline(
            pipeline,
            document=None,
            pipeline_result=pipeline_result,
            runtime_config=runtime_config,
        )
        return self._to_engine_result(result, runtime_config)

    def run_image_path(
        self,
        image_path: str | Path,
        *,
        document_id: str | None = None,
        target_types: tuple[ShapeCandidateTargetType, ...] | list[ShapeCandidateTargetType] | None = None,
        autonomy_level: AutonomyLevel | None = None,
        max_iterations: int | None = None,
        dry_run_only: bool | None = None,
        enable_ai_review: bool | None = None,
    ) -> EngineResult:
        runtime_config = self._runtime_config(
            target_types=target_types,
            autonomy_level=autonomy_level,
            max_iterations=max_iterations,
            dry_run_only=dry_run_only,
            enable_ai_review=enable_ai_review,
        )
        pipeline_result = self.minimal_pipeline.run_from_file(
            image_path,
            document_id=document_id or runtime_config.document_id,
        )
        return self.run_pipeline_result(
            pipeline_result,
            target_types=runtime_config.target_types,
            autonomy_level=runtime_config.autonomy_level,
            max_iterations=runtime_config.max_iterations,
            dry_run_only=runtime_config.dry_run_only,
            enable_ai_review=runtime_config.enable_ai_review,
        )

    def _runtime_config(
        self,
        *,
        target_types: tuple[ShapeCandidateTargetType, ...] | list[ShapeCandidateTargetType] | None,
        autonomy_level: AutonomyLevel | None,
        max_iterations: int | None,
        dry_run_only: bool | None,
        enable_ai_review: bool | None,
    ) -> VectorReconstructionEngineConfig:
        return replace(
            self.config,
            target_types=self.config.target_types if target_types is None else tuple(target_types),
            autonomy_level=self.config.autonomy_level if autonomy_level is None else autonomy_level,
            max_iterations=self.config.max_iterations if max_iterations is None else int(max_iterations),
            dry_run_only=self.config.dry_run_only if dry_run_only is None else bool(dry_run_only),
            enable_ai_review=self.config.enable_ai_review if enable_ai_review is None else bool(enable_ai_review),
        )

    def _configured_pipeline(
        self,
        runtime_config: VectorReconstructionEngineConfig,
    ) -> AutoRefinementPipeline | Any:
        base_pipeline = self.auto_refinement_pipeline
        if not isinstance(base_pipeline, AutoRefinementPipeline):
            return base_pipeline

        preview_policy = base_pipeline.preview_policy
        preview_policy_config = replace(
            preview_policy.config,
            autonomy_level=runtime_config.autonomy_level,
        )
        configured_preview_policy = PreviewAndAutoAcceptPolicy(
            preview_service=preview_policy.preview_service,
            command_executor=preview_policy.command_executor,
            integrity_validator=preview_policy.integrity_validator,
            config=preview_policy_config,
        )
        configured_pipeline_config = replace(
            base_pipeline.config,
            target_types=runtime_config.target_types,
            dry_run_only=runtime_config.dry_run_only,
            evaluate_batch_commands=runtime_config.evaluate_batch_commands,
            max_iterations=runtime_config.max_iterations,
            max_proposals_per_round=runtime_config.max_proposals_per_round,
            improvement_epsilon=runtime_config.improvement_epsilon,
            max_retry_per_target=runtime_config.max_retry_per_target,
            max_retry_per_path=runtime_config.max_retry_per_path,
            max_stalled_rounds=runtime_config.max_stalled_rounds,
        )
        return AutoRefinementPipeline(
            shape_candidate_detector=base_pipeline.shape_candidate_detector,
            proposed_command_planner=base_pipeline.proposed_command_planner,
            preview_policy=configured_preview_policy,
            ai_review_service=self.ai_review_service or base_pipeline.ai_review_service,
            scorer=base_pipeline.scorer,
            integrity_validator=base_pipeline.integrity_validator,
            json_exporter=base_pipeline.json_exporter,
            config=configured_pipeline_config,
        )

    def _run_pipeline(
        self,
        pipeline: AutoRefinementPipeline | Any,
        *,
        document: VectorDocument | None,
        pipeline_result: MinimalPipelineResult | None,
        runtime_config: VectorReconstructionEngineConfig,
    ) -> AutoRefinementPipelineResult:
        if runtime_config.enable_ai_review:
            if pipeline_result is not None:
                if hasattr(pipeline, "run_from_pipeline_result_with_ai_review"):
                    return pipeline.run_from_pipeline_result_with_ai_review(
                        pipeline_result,
                        target_types=runtime_config.target_types,
                        dry_run_only=runtime_config.dry_run_only,
                    )
                return pipeline.run_with_ai_review(
                    pipeline_result.document,
                    target_types=runtime_config.target_types,
                    dry_run_only=runtime_config.dry_run_only,
                )
            if document is None:
                raise ValueError("document input is required")
            return pipeline.run_with_ai_review(
                document,
                target_types=runtime_config.target_types,
                dry_run_only=runtime_config.dry_run_only,
            )

        if pipeline_result is not None:
            return pipeline.run_from_pipeline_result(
                pipeline_result,
                target_types=runtime_config.target_types,
                dry_run_only=runtime_config.dry_run_only,
            )
        if document is None:
            raise ValueError("document input is required")
        return pipeline.run(
            document,
            target_types=runtime_config.target_types,
            dry_run_only=runtime_config.dry_run_only,
        )

    def _to_engine_result(
        self,
        result: AutoRefinementPipelineResult,
        runtime_config: VectorReconstructionEngineConfig,
    ) -> EngineResult:
        policy_feedback = self._policy_feedback(result)
        rejection_memory = tuple(
            RejectionMemoryItem.from_dict(item)
            for item in result.report.rejection_memory
        )
        decisions = tuple(
            decision.policy_result
            for decision in result.preview_decisions
            if decision.policy_result is not None
        )
        external_decisions = tuple(
            decision.external_decision_request
            for decision in result.preview_decisions
            if decision.external_decision_request is not None
        )
        status = self._status_for_result(result, decisions)
        return EngineResult(
            status=status,
            document=result.refined_document,
            report=result.report.to_dict(),
            decisions=tuple(decision for decision in decisions if isinstance(decision, DecisionPolicyResult)),
            external_decisions=tuple(item for item in external_decisions if isinstance(item, ExternalDecisionRequest)),
            policy_feedback=policy_feedback,
            rejection_memory=rejection_memory,
            metadata={
                "target_types": list(runtime_config.target_types),
                "autonomy_level": runtime_config.autonomy_level.value,
                "max_iterations": runtime_config.max_iterations,
                "dry_run_only": runtime_config.dry_run_only,
                "enable_ai_review": runtime_config.enable_ai_review,
                "iteration_count": result.report.iteration_count,
            },
        )

    def _policy_feedback(
        self,
        result: AutoRefinementPipelineResult,
    ) -> tuple[PolicyFeedback, ...]:
        if result.report.policy_feedback:
            return tuple(PolicyFeedback.from_dict(item) for item in result.report.policy_feedback)
        return tuple(
            decision.policy_feedback
            for decision in result.preview_decisions
            if decision.policy_feedback is not None
        )

    def _status_for_result(
        self,
        result: AutoRefinementPipelineResult,
        decisions: tuple[DecisionPolicyResult | None, ...],
    ) -> EngineStatus:
        report_status = EngineStatus.from_legacy(result.report.status)
        if report_status == EngineStatus.COMPLETED_WITH_UNRESOLVED_REGIONS:
            return report_status
        if any(decision is not None and decision.decision == DecisionKind.REQUIRES_EXTERNAL_DECISION for decision in decisions):
            return EngineStatus.REQUIRES_EXTERNAL_DECISION
        return report_status


__all__ = [
    "VectorReconstructionEngine",
    "VectorReconstructionEngineConfig",
]
