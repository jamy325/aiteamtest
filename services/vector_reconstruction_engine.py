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
from services.distance_field_diff import DistanceFieldDiffResult
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
from services.dxf_exporter import DxfExporter
from services.json_exporter import JsonExporter
from services.minimal_pipeline import MinimalPipeline, MinimalPipelineResult
from services.preview_auto_accept_policy import PreviewAndAutoAcceptPolicy
from services.scorer import Scorer
from services.svg_exporter import ExportMode, SvgExporter


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
    export_mode: ExportMode = "all_debug"
    processing_contour_source: str = "all"


@dataclass(frozen=True, slots=True)
class VectorReconstructionArtifactBundle:
    engine_result: EngineResult
    pipeline_result: MinimalPipelineResult
    document_json: str
    output_svg: str
    output_dxf: str
    overlay_png: bytes
    diff_png: bytes
    decision_report: dict[str, Any]
    metrics: dict[str, Any]


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
        self.json_exporter = JsonExporter()
        self.svg_exporter = SvgExporter()
        self.dxf_exporter = DxfExporter()

    def run_document(
        self,
        document: VectorDocument,
        *,
        target_types: tuple[ShapeCandidateTargetType, ...] | list[ShapeCandidateTargetType] | None = None,
        autonomy_level: AutonomyLevel | None = None,
        max_iterations: int | None = None,
        dry_run_only: bool | None = None,
        enable_ai_review: bool | None = None,
        export_mode: ExportMode | None = None,
    ) -> EngineResult:
        runtime_config = self._runtime_config(
            target_types=target_types,
            autonomy_level=autonomy_level,
            max_iterations=max_iterations,
            dry_run_only=dry_run_only,
            enable_ai_review=enable_ai_review,
            export_mode=export_mode,
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
        export_mode: ExportMode | None = None,
    ) -> EngineResult:
        runtime_config = self._runtime_config(
            target_types=target_types,
            autonomy_level=autonomy_level,
            max_iterations=max_iterations,
            dry_run_only=dry_run_only,
            enable_ai_review=enable_ai_review,
            export_mode=export_mode,
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
        export_mode: ExportMode | None = None,
    ) -> EngineResult:
        runtime_config = self._runtime_config(
            target_types=target_types,
            autonomy_level=autonomy_level,
            max_iterations=max_iterations,
            dry_run_only=dry_run_only,
            enable_ai_review=enable_ai_review,
            export_mode=export_mode,
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
            export_mode=runtime_config.export_mode,
        )

    def run_artifact_bundle(
        self,
        image_path: str | Path,
        *,
        document_id: str | None = None,
        target_types: tuple[ShapeCandidateTargetType, ...] | list[ShapeCandidateTargetType] | None = None,
        autonomy_level: AutonomyLevel | None = None,
        max_iterations: int | None = None,
        dry_run_only: bool | None = None,
        enable_ai_review: bool | None = None,
        export_mode: ExportMode | None = None,
    ) -> VectorReconstructionArtifactBundle:
        runtime_config = self._runtime_config(
            target_types=target_types,
            autonomy_level=autonomy_level,
            max_iterations=max_iterations,
            dry_run_only=dry_run_only,
            enable_ai_review=enable_ai_review,
            export_mode=export_mode,
        )
        pipeline_result = self.minimal_pipeline.run_from_file(
            image_path,
            document_id=document_id or runtime_config.document_id,
        )
        engine_result = self.run_pipeline_result(
            pipeline_result,
            target_types=runtime_config.target_types,
            autonomy_level=runtime_config.autonomy_level,
            max_iterations=runtime_config.max_iterations,
            dry_run_only=runtime_config.dry_run_only,
            enable_ai_review=runtime_config.enable_ai_review,
            export_mode=runtime_config.export_mode,
        )
        final_document = engine_result.document or pipeline_result.document
        source_image = pipeline_result.source_image
        if source_image is None:
            raise ValueError("pipeline result does not include source_image")
        artifact_score_summary = self._artifact_score_summary(
            before_document=pipeline_result.document,
            after_document=final_document,
        )
        stroke_summary = self._stroke_summary(final_document, artifact_score_summary)
        decision_report = self._bundle_decision_report(
            engine_result=engine_result,
            runtime_config=runtime_config,
            artifact_score_summary=artifact_score_summary,
            stroke_summary=stroke_summary,
        )
        return VectorReconstructionArtifactBundle(
            engine_result=engine_result,
            pipeline_result=pipeline_result,
            document_json=self.json_exporter.export_document(final_document),
            output_svg=self.svg_exporter.export_document(final_document, export_mode=runtime_config.export_mode),
            output_dxf=self.dxf_exporter.export_document(final_document, export_mode=runtime_config.export_mode),
            overlay_png=self.minimal_pipeline.export_overlay(final_document, source_image),
            diff_png=self.minimal_pipeline.export_distance_field_diff(final_document),
            decision_report=decision_report,
            metrics=self._bundle_metrics(
                engine_result=engine_result,
                document=final_document,
                runtime_config=runtime_config,
                artifact_score_summary=artifact_score_summary,
                stroke_summary=stroke_summary,
            ),
        )

    def _runtime_config(
        self,
        *,
        target_types: tuple[ShapeCandidateTargetType, ...] | list[ShapeCandidateTargetType] | None,
        autonomy_level: AutonomyLevel | None,
        max_iterations: int | None,
        dry_run_only: bool | None,
        enable_ai_review: bool | None,
        export_mode: ExportMode | None,
    ) -> VectorReconstructionEngineConfig:
        selected_export_mode = self.config.export_mode if export_mode is None else export_mode
        return replace(
            self.config,
            target_types=self.config.target_types if target_types is None else tuple(target_types),
            autonomy_level=self.config.autonomy_level if autonomy_level is None else autonomy_level,
            max_iterations=self.config.max_iterations if max_iterations is None else int(max_iterations),
            dry_run_only=self.config.dry_run_only if dry_run_only is None else bool(dry_run_only),
            enable_ai_review=self.config.enable_ai_review if enable_ai_review is None else bool(enable_ai_review),
            export_mode=selected_export_mode,
            processing_contour_source=_processing_contour_source_for_export_mode(selected_export_mode),
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
            processing_contour_source=runtime_config.processing_contour_source,
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
                "export_mode": runtime_config.export_mode,
                "processing_contour_source": runtime_config.processing_contour_source,
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

    def _bundle_metrics(
        self,
        *,
        engine_result: EngineResult,
        document: VectorDocument,
        runtime_config: VectorReconstructionEngineConfig,
        artifact_score_summary: dict[str, Any],
        stroke_summary: dict[str, Any],
    ) -> dict[str, Any]:
        report = dict(engine_result.report)
        return {
            "status": engine_result.status.value,
            "document_id": document.document_id,
            "path_count": len(document.paths),
            "segment_count": len(document.segments),
            "constraint_count": len(document.constraints),
            "autonomy_level": runtime_config.autonomy_level.value,
            "target_types": list(runtime_config.target_types),
            "dry_run_only": runtime_config.dry_run_only,
            "enable_ai_review": runtime_config.enable_ai_review,
            "max_iterations": runtime_config.max_iterations,
            "export_mode": runtime_config.export_mode,
            "processing_contour_source": runtime_config.processing_contour_source,
            "processed_path_count": int(report.get("processing_summary", {}).get("processed_path_count", 0)),
            "processed_binary_path_count": int(report.get("processing_summary", {}).get("processed_binary_path_count", 0)),
            "processed_skeleton_path_count": int(report.get("processing_summary", {}).get("processed_skeleton_path_count", 0)),
            "processing_warnings": list(report.get("processing_summary", {}).get("warnings", ())),
            "iteration_count": int(report.get("iteration_count", 0)),
            "score_before": artifact_score_summary["score_before"],
            "score_after": artifact_score_summary["score_after"],
            "decision_stats": dict(report.get("decision_stats", {})),
            "integrity": dict(report.get("integrity", {})),
            "unresolved_targets": list(report.get("unresolved_targets", ())),
            "stroke_width": stroke_summary.get("stroke_width"),
            "stroke_width_confidence": stroke_summary.get("stroke_width_confidence"),
            "stroke_endpoint_count": stroke_summary.get("stroke_endpoint_count"),
            "stroke_junction_count": stroke_summary.get("stroke_junction_count"),
            "stroke_branch_count": stroke_summary.get("stroke_branch_count"),
            "stroke_mask_error": artifact_score_summary.get("stroke_mask_error"),
            "stroke_mask_error_score": artifact_score_summary.get("stroke_mask_error_score"),
            "errors": list(engine_result.errors),
        }

    def _bundle_decision_report(
        self,
        *,
        engine_result: EngineResult,
        runtime_config: VectorReconstructionEngineConfig,
        artifact_score_summary: dict[str, Any],
        stroke_summary: dict[str, Any],
    ) -> dict[str, Any]:
        decision_report = engine_result.to_dict()
        metadata = dict(decision_report.get("metadata", {}))
        metadata.update(
            {
                "export_mode": runtime_config.export_mode,
                "processing_contour_source": runtime_config.processing_contour_source,
                **stroke_summary,
            }
        )
        decision_report["metadata"] = metadata
        decision_report["stroke_summary"] = dict(stroke_summary)
        report = dict(decision_report.get("report", {}))
        report.update(
            {
                "score_before": artifact_score_summary["score_before"],
                "score_after": artifact_score_summary["score_after"],
                "stroke_mask_error": artifact_score_summary.get("stroke_mask_error"),
                "stroke_mask_error_score": artifact_score_summary.get("stroke_mask_error_score"),
            }
        )
        decision_report["report"] = report
        return decision_report

    def _artifact_score_summary(
        self,
        *,
        before_document: VectorDocument,
        after_document: VectorDocument,
    ) -> dict[str, Any]:
        scorer = self._scorer()
        before_diff = self._stroke_diff_result(before_document)
        after_diff = self._stroke_diff_result(after_document)
        before_score = scorer.score_document(
            before_document,
            stroke_mask_error=None if before_diff is None else before_diff.stroke_mask_error,
        )
        after_score = scorer.score_document(
            after_document,
            stroke_mask_error=None if after_diff is None else after_diff.stroke_mask_error,
        )
        return {
            "score_before": before_score.total_score,
            "score_after": after_score.total_score,
            "stroke_mask_error": None if after_diff is None else after_diff.stroke_mask_error,
            "stroke_mask_error_score": after_score.breakdown.stroke_mask_error_score,
        }

    def _stroke_diff_result(self, document: VectorDocument) -> DistanceFieldDiffResult | None:
        renderer = getattr(self.minimal_pipeline, "distance_field_diff_renderer", None)
        if renderer is None or not hasattr(renderer, "render_diff"):
            return None
        try:
            result = renderer.render_diff(document)
        except Exception:
            return None
        return result if isinstance(result, DistanceFieldDiffResult) else None

    def _stroke_summary(
        self,
        document: VectorDocument,
        artifact_score_summary: dict[str, Any],
    ) -> dict[str, Any]:
        stroke_paths = [path for path in document.paths if path.source == "skeleton_contour"]
        widths = [
            float(path.style.stroke_width)
            for path in stroke_paths
            if path.style is not None and float(path.style.stroke_width) > 0.0
        ]
        confidences = [
            float(path.metadata["stroke_width_confidence"])
            for path in stroke_paths
            if "stroke_width_confidence" in path.metadata
        ]
        return {
            "stroke_width": round(sum(widths) / len(widths), 4) if widths else None,
            "stroke_width_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "stroke_endpoint_count": sum(int(path.metadata.get("endpoint_count", 0)) for path in stroke_paths),
            "stroke_junction_count": sum(int(path.metadata.get("junction_count", 0)) for path in stroke_paths),
            "stroke_branch_count": sum(int(path.metadata.get("branch_count", 0)) for path in stroke_paths),
            "stroke_mask_error": artifact_score_summary.get("stroke_mask_error"),
        }

    def _scorer(self) -> Scorer:
        candidate = getattr(self.auto_refinement_pipeline, "scorer", None)
        if isinstance(candidate, Scorer):
            return candidate
        return Scorer()


__all__ = [
    "VectorReconstructionArtifactBundle",
    "VectorReconstructionEngine",
    "VectorReconstructionEngineConfig",
]


def _processing_contour_source_for_export_mode(export_mode: ExportMode) -> str:
    if export_mode == "outline":
        return "binary"
    if export_mode == "centerline":
        return "skeleton"
    return "all"
