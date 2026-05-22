from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from core.types import VectorDocument
from services.ai_agent import AIReviewInput, AIReviewOutput, AIReviewService
from services.auto_refinement_pipeline import AutoRefinementPipeline, AutoRefinementPipelineResult
from services.minimal_pipeline import MinimalPipeline, MinimalPipelineResult
from ui.canvas_widget import CanvasWidget
from ui.canvas_widget import AISuggestionOverlay


@dataclass(frozen=True, slots=True)
class AIReviewDisplayState:
    summary: str = ""
    issues: tuple[dict[str, Any], ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()
    proposed_commands: tuple[dict[str, Any], ...] = ()
    preview_decisions: tuple[dict[str, Any], ...] = ()
    diff_summary: dict[str, Any] = field(default_factory=dict)
    suggestion_overlays: tuple[AISuggestionOverlay, ...] = ()


@dataclass(frozen=True, slots=True)
class AutoRefinementCommandState:
    decision_id: str
    command: dict[str, Any]
    candidate_id: str | None
    decision: str
    review_state: str
    reason: str
    risk_flags: tuple[str, ...]
    locked_target_ids: tuple[str, ...] = ()


class MainWindow:
    def __init__(
        self,
        *,
        ai_review_service: AIReviewService | None = None,
        auto_refinement_pipeline: AutoRefinementPipeline | None = None,
        pipeline: MinimalPipeline | None = None,
        canvas_widget: CanvasWidget | None = None,
    ) -> None:
        self.ai_review_service = ai_review_service
        self.auto_refinement_pipeline = auto_refinement_pipeline or AutoRefinementPipeline()
        self.pipeline = pipeline or MinimalPipeline()
        self.canvas_widget = canvas_widget or CanvasWidget()
        self.review_display_state = AIReviewDisplayState()
        self.selected_image_path: str | None = None
        self.pipeline_result: MinimalPipelineResult | None = None
        self.last_error: str | None = None
        self.last_json_export_path: str | None = None
        self.last_overlay_export_path: str | None = None
        self.last_distance_field_diff_export_path: str | None = None
        self.last_review_input: AIReviewInput | None = None
        self.last_review_output: AIReviewOutput | None = None
        self.last_auto_refinement_result: AutoRefinementPipelineResult | None = None
        self.executed_commands: tuple[dict[str, Any], ...] = ()
        self._auto_refinement_base_document: VectorDocument | None = None
        self._auto_refinement_command_states: tuple[AutoRefinementCommandState, ...] = ()

    def select_image(self, image_path: str | Path) -> str:
        try:
            self.pipeline.load_image(image_path)
            self.selected_image_path = str(image_path)
            self.pipeline_result = None
            self.review_display_state = AIReviewDisplayState()
            self.last_auto_refinement_result = None
            self._auto_refinement_base_document = None
            self._auto_refinement_command_states = ()
            self.canvas_widget.set_image_path(self.selected_image_path)
            self.canvas_widget.set_document(None)
            self.canvas_widget.set_review_display(
                summary="",
                issues=(),
                candidates=(),
                proposed_commands=(),
                preview_decisions=(),
                diff_summary={},
            )
            self._clear_error()
            return self.selected_image_path
        except Exception as exc:
            self._set_error(exc)
            raise

    def trigger_auto_fit(self, *, document_id: str = "document_1") -> MinimalPipelineResult:
        if self.selected_image_path is None:
            exc = ValueError("no image selected")
            self._set_error(exc)
            raise exc

        try:
            result = self.pipeline.run_from_file(self.selected_image_path, document_id=document_id)
            self.pipeline_result = result
            self.review_display_state = AIReviewDisplayState()
            self.last_auto_refinement_result = None
            self._auto_refinement_base_document = None
            self._auto_refinement_command_states = ()
            self.canvas_widget.set_document(result.document)
            self.canvas_widget.set_review_display(
                summary="",
                issues=(),
                candidates=(),
                proposed_commands=(),
                preview_decisions=(),
                diff_summary={},
            )
            self._clear_error()
            return result
        except Exception as exc:
            self._set_error(exc)
            raise

    def trigger_ai_review_for_current_document(
        self,
        *,
        available_tools: tuple[str, ...] = (
            "propose_replace_segment_with_line",
            "propose_replace_segment_with_arc",
            "propose_replace_segment_with_circle",
            "propose_replace_segment_with_ellipse",
            "propose_batch_refinement",
        ),
        alpha_notes: str | None = None,
        color_notes: str | None = None,
    ) -> AIReviewDisplayState:
        if self.ai_review_service is None:
            exc = RuntimeError("AI review service is not configured")
            self._set_error(exc)
            raise exc
        if self.pipeline_result is None or self.selected_image_path is None:
            exc = ValueError("auto fit must run before AI review")
            self._set_error(exc)
            raise exc

        try:
            review_artifacts = self._ensure_review_artifacts()
            display_state = self.trigger_ai_review(
                original_image=self.selected_image_path,
                overlay_image=review_artifacts["overlay"],
                distance_field_diff_image=review_artifacts["distance_field_diff"],
                vector_document_json=json.loads(review_artifacts["json_payload"]),
                fit_error=self._aggregate_fit_error(self.pipeline_result.document),
                complexity_score=self._aggregate_complexity(self.pipeline_result.document),
                topology_status=self._aggregate_topology_status(self.pipeline_result.document),
                self_intersection_count=self._aggregate_self_intersections(self.pipeline_result.document),
                coordinate_system=asdict(self.pipeline_result.document.coordinate_system),
                available_tools=available_tools,
                alpha_notes=alpha_notes,
                color_notes=color_notes,
            )
            self._clear_error()
            return display_state
        except Exception as exc:
            self._set_error(exc)
            raise

    def trigger_auto_refinement_review_for_current_document(
        self,
        *,
        target_types: tuple[str, ...] = (),
        dry_run_only: bool | None = None,
    ) -> AIReviewDisplayState:
        if self.pipeline_result is None:
            exc = ValueError("auto fit must run before auto refinement review")
            self._set_error(exc)
            raise exc

        try:
            base_document = self.pipeline_result.document
            auto_result = self.auto_refinement_pipeline.run_from_pipeline_result(
                self.pipeline_result,
                target_types=target_types,
                dry_run_only=dry_run_only,
            )
            self.last_auto_refinement_result = auto_result
            self._auto_refinement_base_document = base_document
            self._auto_refinement_command_states = self._build_auto_refinement_command_states(
                auto_result,
                document=auto_result.refined_document,
            )
            self.pipeline_result = replace(self.pipeline_result, document=auto_result.refined_document)
            self.canvas_widget.set_document(auto_result.refined_document)
            self._refresh_auto_refinement_display()
            self._clear_error()
            return self.review_display_state
        except Exception as exc:
            self._set_error(exc)
            raise

    def apply_user_confirm_command(self, decision_id: str) -> AIReviewDisplayState:
        command_state = self._require_auto_refinement_command_state(decision_id)
        if command_state.review_state != "user_confirm":
            exc = ValueError(f"decision is not pending user confirmation: {decision_id}")
            self._set_error(exc)
            raise exc
        if command_state.locked_target_ids:
            exc = ValueError(f"locked targets cannot be applied: {', '.join(command_state.locked_target_ids)}")
            self._set_error(exc)
            raise exc
        if self.pipeline_result is None:
            exc = ValueError("auto refinement review must run before applying commands")
            self._set_error(exc)
            raise exc

        try:
            execution = self.auto_refinement_pipeline.preview_policy.command_executor.execute(
                command_state.command,
                self.pipeline_result.document,
            )
            if not execution.success:
                raise ValueError(execution.reason or "command execution failed")

            self.pipeline_result = replace(self.pipeline_result, document=execution.document)
            self.canvas_widget.set_document(execution.document)
            self.executed_commands = self.executed_commands + (dict(command_state.command),)
            self._auto_refinement_command_states = tuple(
                replace(state, review_state="applied") if state.decision_id == decision_id else state
                for state in self._auto_refinement_command_states
            )
            self._refresh_auto_refinement_display()
            self._clear_error()
            return self.review_display_state
        except Exception as exc:
            self._set_error(exc)
            raise

    def apply_all_user_confirm_commands(self) -> AIReviewDisplayState:
        decision_ids = tuple(
            state.decision_id
            for state in self._auto_refinement_command_states
            if state.review_state == "user_confirm"
        )
        for decision_id in decision_ids:
            self.apply_user_confirm_command(decision_id)
        return self.review_display_state

    def ignore_command(self, decision_id: str) -> AIReviewDisplayState:
        command_state = self._require_auto_refinement_command_state(decision_id)
        if command_state.review_state != "user_confirm":
            exc = ValueError(f"decision is not pending user confirmation: {decision_id}")
            self._set_error(exc)
            raise exc

        self._auto_refinement_command_states = tuple(
            replace(state, review_state="ignored") if state.decision_id == decision_id else state
            for state in self._auto_refinement_command_states
        )
        self._refresh_auto_refinement_display()
        self._clear_error()
        return self.review_display_state

    def export_json(self, output_path: str | Path) -> str:
        document = self._require_document()
        try:
            payload = self.pipeline.export_json(document, output_path)
            self.last_json_export_path = str(output_path)
            self._clear_error()
            return payload
        except Exception as exc:
            self._set_error(exc)
            raise

    def export_overlay(self, output_path: str | Path) -> str:
        document = self._require_document()
        image = self._require_source_image()
        try:
            self.pipeline.export_overlay(document, image, output_path)
            self.last_overlay_export_path = str(output_path)
            self._clear_error()
            return self.last_overlay_export_path
        except Exception as exc:
            self._set_error(exc)
            raise

    def export_distance_field_diff(self, output_path: str | Path) -> str:
        document = self._require_document()
        try:
            self.pipeline.export_distance_field_diff(document, output_path)
            self.last_distance_field_diff_export_path = str(output_path)
            self._clear_error()
            return self.last_distance_field_diff_export_path
        except Exception as exc:
            self._set_error(exc)
            raise

    def trigger_ai_review(
        self,
        *,
        original_image: str | None,
        overlay_image: str | None,
        distance_field_diff_image: str | None,
        vector_document_json: dict[str, Any],
        fit_error: float,
        complexity_score: float,
        topology_status: str,
        self_intersection_count: int,
        coordinate_system: dict[str, Any],
        available_tools: tuple[str, ...] = (),
        alpha_notes: str | None = None,
        color_notes: str | None = None,
    ) -> AIReviewDisplayState:
        if self.ai_review_service is None:
            raise RuntimeError("AI review service is not configured")

        review_input = AIReviewInput(
            original_image=original_image,
            overlay_image=overlay_image,
            distance_field_diff_image=distance_field_diff_image,
            vector_document_json=vector_document_json,
            fit_error=float(fit_error),
            complexity_score=float(complexity_score),
            topology_status=str(topology_status),
            self_intersection_count=int(self_intersection_count),
            coordinate_system=dict(coordinate_system),
            user_locked_ids=self.canvas_widget.locked_ids,
            available_tools=tuple(str(tool) for tool in available_tools),
            alpha_notes=alpha_notes,
            color_notes=color_notes,
        )
        review_output = self.ai_review_service.run_review(review_input)
        self.last_review_input = review_input
        self.last_review_output = review_output
        suggestion_overlays = self.canvas_widget.set_review_display(
            summary=review_output.summary,
            issues=review_output.issues,
            candidates=(),
            proposed_commands=review_output.proposed_commands,
            preview_decisions=(),
            diff_summary={},
        )
        self.review_display_state = AIReviewDisplayState(
            summary=review_output.summary,
            issues=review_output.issues,
            candidates=(),
            proposed_commands=review_output.proposed_commands,
            preview_decisions=(),
            diff_summary={},
            suggestion_overlays=suggestion_overlays,
        )
        return self.review_display_state

    def _require_document(self) -> VectorDocument:
        if self.pipeline_result is None:
            exc = ValueError("auto fit must run before export")
            self._set_error(exc)
            raise exc
        return self.pipeline_result.document

    def _require_source_image(self) -> np.ndarray:
        if self.pipeline_result is not None and self.pipeline_result.source_image is not None:
            return self.pipeline_result.source_image
        if self.selected_image_path is None:
            exc = ValueError("no image selected")
            self._set_error(exc)
            raise exc
        return self.pipeline.load_image(self.selected_image_path)

    def _ensure_review_artifacts(self) -> dict[str, str]:
        document = self._require_document()
        image = self._require_source_image()
        artifact_dir = Path(tempfile.gettempdir()) / "aiteamtest_ui_review"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        document_id = document.document_id
        json_path = artifact_dir / f"{document_id}.json"
        overlay_path = artifact_dir / f"{document_id}_overlay.png"
        diff_path = artifact_dir / f"{document_id}_diff.png"
        json_payload = self.pipeline.export_json(document, json_path)
        self.pipeline.export_overlay(document, image, overlay_path)
        self.pipeline.export_distance_field_diff(document, diff_path)
        self.last_json_export_path = str(json_path)
        self.last_overlay_export_path = str(overlay_path)
        self.last_distance_field_diff_export_path = str(diff_path)
        return {
            "json": str(json_path),
            "overlay": str(overlay_path),
            "distance_field_diff": str(diff_path),
            "json_payload": json_payload,
        }

    def _aggregate_fit_error(self, document: VectorDocument) -> float:
        errors = [float(segment.fit_error) for segment in document.segments if segment.fit_error is not None]
        if not errors:
            return 0.0
        return sum(errors) / len(errors)

    def _aggregate_complexity(self, document: VectorDocument) -> float:
        scores = [
            float(segment.complexity_score)
            for segment in document.segments
            if segment.complexity_score is not None
        ]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _aggregate_topology_status(self, document: VectorDocument) -> str:
        statuses = {path.topology_status for path in document.paths}
        if "topology_error" in statuses:
            return "topology_error"
        if statuses == {"closed"}:
            return "closed"
        if "open" in statuses:
            return "open"
        return next(iter(statuses), "open")

    def _aggregate_self_intersections(self, document: VectorDocument) -> int:
        return sum(int(path.self_intersection_count) for path in document.paths)

    def _refresh_auto_refinement_display(self) -> None:
        if self.last_auto_refinement_result is None or self._auto_refinement_base_document is None:
            return
        current_document = self.pipeline_result.document if self.pipeline_result is not None else self.last_auto_refinement_result.refined_document
        score_before = self.last_auto_refinement_result.report.score_before
        if current_document == self.last_auto_refinement_result.refined_document:
            score_after = self.last_auto_refinement_result.report.score_after
        else:
            score_after = self.auto_refinement_pipeline.scorer.score_document(current_document).total_score
        diff_summary = self._build_document_diff_summary(
            before_document=self._auto_refinement_base_document,
            after_document=current_document,
            score_before=score_before,
            score_after=score_after,
        )
        candidates = self._candidate_display_items(
            self.last_auto_refinement_result,
            self._auto_refinement_command_states,
        )
        preview_decisions = tuple(
            self._command_state_display_item(state)
            for state in self._auto_refinement_command_states
        )
        summary = self._auto_refinement_summary(
            candidate_count=len(self.last_auto_refinement_result.candidates),
            command_count=len(self.last_auto_refinement_result.proposed_commands),
            states=self._auto_refinement_command_states,
        )
        suggestion_overlays = self.canvas_widget.set_review_display(
            summary=summary,
            issues=(),
            candidates=candidates,
            proposed_commands=self.last_auto_refinement_result.proposed_commands,
            preview_decisions=preview_decisions,
            diff_summary=diff_summary,
        )
        self.review_display_state = AIReviewDisplayState(
            summary=summary,
            issues=(),
            candidates=candidates,
            proposed_commands=self.last_auto_refinement_result.proposed_commands,
            preview_decisions=preview_decisions,
            diff_summary=diff_summary,
            suggestion_overlays=suggestion_overlays,
        )

    def _build_auto_refinement_command_states(
        self,
        auto_result: AutoRefinementPipelineResult,
        *,
        document: VectorDocument,
    ) -> tuple[AutoRefinementCommandState, ...]:
        states: list[AutoRefinementCommandState] = []
        for index, decision in enumerate(auto_result.preview_decisions):
            command = dict(decision.command)
            states.append(
                AutoRefinementCommandState(
                    decision_id=f"decision_{index}",
                    command=command,
                    candidate_id=self._coerce_optional_string(command.get("candidate_id")),
                    decision=decision.decision,
                    review_state=self._review_state_for_decision(decision.decision),
                    reason=decision.reason,
                    risk_flags=tuple(str(flag) for flag in decision.risk_flags),
                    locked_target_ids=self._locked_target_ids_for_command(command, document),
                )
            )
        return tuple(states)

    def _candidate_display_items(
        self,
        auto_result: AutoRefinementPipelineResult,
        command_states: tuple[AutoRefinementCommandState, ...],
    ) -> tuple[dict[str, Any], ...]:
        state_by_candidate: dict[str, list[AutoRefinementCommandState]] = {}
        for command_state in command_states:
            if command_state.candidate_id is None:
                continue
            state_by_candidate.setdefault(command_state.candidate_id, []).append(command_state)

        items: list[dict[str, Any]] = []
        for candidate in auto_result.candidates:
            matching_states = state_by_candidate.get(candidate.candidate_id, [])
            review_state = self._summarize_candidate_review_state(matching_states)
            items.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "target_type": candidate.target_type,
                    "path_id": candidate.path_id,
                    "segment_range": list(candidate.segment_range),
                    "reason": candidate.reason,
                    "confidence": candidate.confidence,
                    "bbox": candidate.evidence.get("bbox"),
                    "decision": review_state,
                    "review_state": review_state,
                }
            )
        return tuple(items)

    def _command_state_display_item(self, command_state: AutoRefinementCommandState) -> dict[str, Any]:
        item = dict(command_state.command)
        item.update(
            {
                "decision_id": command_state.decision_id,
                "candidate_id": command_state.candidate_id,
                "decision": command_state.decision,
                "review_state": command_state.review_state,
                "reason": command_state.reason,
                "risk_flags": list(command_state.risk_flags),
                "locked_anchor_ids": list(command_state.locked_target_ids),
            }
        )
        return item

    def _review_state_for_decision(self, decision: str) -> str:
        if decision == "auto_accept":
            return "auto_accepted"
        if decision == "reject":
            return "rejected"
        return "user_confirm"

    def _summarize_candidate_review_state(
        self,
        command_states: list[AutoRefinementCommandState],
    ) -> str | None:
        if not command_states:
            return None
        states = {state.review_state for state in command_states}
        if len(states) == 1:
            return next(iter(states))
        return "mixed"

    def _locked_target_ids_for_command(
        self,
        command: dict[str, Any],
        document: VectorDocument,
    ) -> tuple[str, ...]:
        locked_target_ids: list[str] = []
        path_id = self._coerce_optional_string(command.get("path_id"))
        if path_id is not None and path_id in self.canvas_widget.locked_ids:
            locked_target_ids.append(path_id)

        for segment_id in self._resolve_segment_ids(command, document):
            if segment_id in self.canvas_widget.locked_ids:
                locked_target_ids.append(segment_id)

        explicit_segment_id = self._coerce_optional_string(command.get("segment_id"))
        if explicit_segment_id is not None and explicit_segment_id in self.canvas_widget.locked_ids:
            locked_target_ids.append(explicit_segment_id)

        for anchor_id in command.get("locked_anchor_ids", ()):
            anchor_id_text = str(anchor_id)
            if anchor_id_text in self.canvas_widget.locked_ids:
                locked_target_ids.append(anchor_id_text)

        return tuple(dict.fromkeys(locked_target_ids))

    def _resolve_segment_ids(
        self,
        command: dict[str, Any],
        document: VectorDocument,
    ) -> tuple[str, ...]:
        explicit_segment_id = self._coerce_optional_string(command.get("segment_id"))
        if explicit_segment_id is not None:
            return (explicit_segment_id,)

        path_id = self._coerce_optional_string(command.get("path_id"))
        segment_range = command.get("segment_range")
        if not isinstance(segment_range, list) or len(segment_range) != 2:
            return ()
        if path_id is None:
            return ()
        if not all(isinstance(index, int) and not isinstance(index, bool) for index in segment_range):
            return ()
        path = next((value for value in document.paths if value.path_id == path_id), None)
        if path is None:
            return ()
        start_index, end_index = segment_range
        if start_index < 0 or end_index < start_index or end_index >= len(path.segments):
            return ()
        return tuple(path.segments[start_index : end_index + 1])

    def _build_document_diff_summary(
        self,
        *,
        before_document: VectorDocument,
        after_document: VectorDocument,
        score_before: float,
        score_after: float,
    ) -> dict[str, Any]:
        segment_before = self._segment_type_counts(before_document)
        segment_after = self._segment_type_counts(after_document)
        topology_before = self._topology_status_counts(before_document)
        topology_after = self._topology_status_counts(after_document)
        self_intersection_before = self._aggregate_self_intersections(before_document)
        self_intersection_after = self._aggregate_self_intersections(after_document)
        return {
            "segment_type_counts": {
                "before": segment_before,
                "after": segment_after,
                "delta": self._delta_counts(segment_before, segment_after),
            },
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_after - score_before,
            "topology_status_counts": {
                "before": topology_before,
                "after": topology_after,
                "delta": self._delta_counts(topology_before, topology_after),
            },
            "self_intersection_total": {
                "before": self_intersection_before,
                "after": self_intersection_after,
                "delta": self_intersection_after - self_intersection_before,
            },
        }

    def _auto_refinement_summary(
        self,
        *,
        candidate_count: int,
        command_count: int,
        states: tuple[AutoRefinementCommandState, ...],
    ) -> str:
        counts = {
            "auto_accepted": sum(1 for state in states if state.review_state == "auto_accepted"),
            "user_confirm": sum(1 for state in states if state.review_state == "user_confirm"),
            "rejected": sum(1 for state in states if state.review_state == "rejected"),
            "applied": sum(1 for state in states if state.review_state == "applied"),
            "ignored": sum(1 for state in states if state.review_state == "ignored"),
        }
        return (
            f"Auto refinement review: {candidate_count} candidates, {command_count} commands, "
            f"{counts['auto_accepted']} auto accepted, {counts['user_confirm']} pending, "
            f"{counts['rejected']} rejected, {counts['applied']} applied, {counts['ignored']} ignored."
        )

    def _segment_type_counts(self, document: VectorDocument) -> dict[str, int]:
        counts: dict[str, int] = {}
        for segment in document.segments:
            counts[segment.type] = counts.get(segment.type, 0) + 1
        return counts

    def _topology_status_counts(self, document: VectorDocument) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in document.paths:
            counts[path.topology_status] = counts.get(path.topology_status, 0) + 1
        return counts

    def _delta_counts(
        self,
        before: dict[str, int],
        after: dict[str, int],
    ) -> dict[str, int]:
        keys = sorted(set(before) | set(after))
        return {
            key: after.get(key, 0) - before.get(key, 0)
            for key in keys
        }

    def _require_auto_refinement_command_state(self, decision_id: str) -> AutoRefinementCommandState:
        for command_state in self._auto_refinement_command_states:
            if command_state.decision_id == decision_id:
                return command_state
        exc = ValueError(f"unknown auto refinement decision: {decision_id}")
        self._set_error(exc)
        raise exc

    @staticmethod
    def _coerce_optional_string(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def _set_error(self, exc: Exception) -> None:
        self.last_error = str(exc)

    def _clear_error(self) -> None:
        self.last_error = None
