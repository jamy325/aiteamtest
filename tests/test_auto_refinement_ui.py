from __future__ import annotations

from services.auto_refinement_pipeline import AutoRefinementPipelineResult, AutoRefinementReport
from services.command_preview import CommandPreviewResult, ConstraintChangeSummary, ExportImpactSummary
from services.contour_extractor import ExtractedContours
from services.minimal_pipeline import MinimalPipelineResult
from services.preview_auto_accept_policy import PreviewAndAutoAcceptPolicy, PreviewDecision
from services.scorer import Scorer
from ui.canvas_widget import CanvasWidget
from ui.main_window import MainWindow
from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment, ShapeCandidate


class FakeAutoRefinementPipeline:
    def __init__(self, result: AutoRefinementPipelineResult) -> None:
        self._result = result
        self.preview_policy = PreviewAndAutoAcceptPolicy()
        self.scorer = Scorer()

    def run_from_pipeline_result(
        self,
        pipeline_result: MinimalPipelineResult,
        *,
        target_types: tuple[str, ...] = (),
        dry_run_only: bool | None = None,
    ) -> AutoRefinementPipelineResult:
        return self._result


def _contours() -> ExtractedContours:
    return ExtractedContours(binary_contours=(), skeleton_contours=(), skeleton_junctions=())


def _make_document(*, segment_types: tuple[str, ...]) -> object:
    document = create_document(
        document_id="doc_auto_ui",
        width=120.0,
        height=80.0,
        coordinate_system=CoordinateSystem(
            unit="px",
            y_axis="down",
            precision=4,
            view_box=(0.0, 0.0, 120.0, 80.0),
        ),
    )
    for index, segment_type in enumerate(segment_types):
        path_id = f"path_{index + 1}"
        segment_id = f"seg_{index + 1}"
        path = VectorPath(path_id=path_id, segments=(segment_id,), topology_status="open")
        document = add_path(document, path)
        if segment_type == "line":
            segment = Segment(
                segment_id,
                path_id,
                "line",
                {"start": [10.0 + (index * 30.0), 10.0], "end": [30.0 + (index * 30.0), 10.0]},
            )
        elif segment_type == "circle":
            segment = Segment(
                segment_id,
                path_id,
                "circle",
                {
                    "cx": 20.0 + (index * 30.0),
                    "cy": 20.0,
                    "r": 10.0,
                    "start": [10.0 + (index * 30.0), 20.0],
                    "end": [10.0 + (index * 30.0), 20.0],
                },
            )
        else:
            segment = Segment(
                segment_id,
                path_id,
                "polyline",
                {
                    "points": [
                        [10.0 + (index * 30.0), 10.0],
                        [20.0 + (index * 30.0), 10.0],
                        [30.0 + (index * 30.0), 10.0],
                    ]
                },
            )
        document = add_segment(document, segment)
    return document


def _pipeline_result(document) -> MinimalPipelineResult:
    return MinimalPipelineResult(
        document=document,
        json_payload="{}",
        extracted_contours=_contours(),
        source_image=None,
        debug_artifacts=None,
    )


def _preview_result(
    *,
    command_id: str,
    path_id: str,
    segment_id: str,
    before_type: str,
    after_type: str,
    score_delta: float,
) -> CommandPreviewResult:
    return CommandPreviewResult(
        success=True,
        command_id=command_id,
        reason=None,
        old_score=10.0,
        predicted_new_score=10.0 + score_delta,
        score_delta=score_delta,
        affected_paths=(path_id,),
        affected_segments=(segment_id,),
        topology_status_before={path_id: "open"},
        topology_status_after={path_id: "open"},
        self_intersection_count_before={path_id: 0},
        self_intersection_count_after={path_id: 0},
        segment_type_summary={
            "before": {before_type: 1},
            "after": {after_type: 1},
            "delta": {before_type: -1, after_type: 1},
        },
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


def _auto_result_for_review() -> AutoRefinementPipelineResult:
    original_document = _make_document(segment_types=("polyline", "polyline"))
    refined_document = _make_document(segment_types=("circle", "polyline"))
    preview_decisions = (
        PreviewDecision(
            command={
                "tool": "propose_replace_path_with_circle",
                "path_id": "path_1",
                "reason": "Reads as a circle.",
                "confidence": 0.94,
                "requires_user_confirmation": True,
                "candidate_id": "cand_circle_1",
            },
            preview_result=_preview_result(
                command_id="circle_path_1",
                path_id="path_1",
                segment_id="seg_1",
                before_type="polyline",
                after_type="circle",
                score_delta=-2.5,
            ),
            decision="auto_accept",
            reason="Safe to auto-accept.",
            risk_flags=(),
        ),
        PreviewDecision(
            command={
                "tool": "propose_replace_segment_with_line",
                "path_id": "path_2",
                "segment_range": [0, 0],
                "reason": "Locked line candidate.",
                "confidence": 0.62,
                "requires_user_confirmation": True,
                "candidate_id": "cand_line_2",
            },
            preview_result=_preview_result(
                command_id="line_path_2",
                path_id="path_2",
                segment_id="seg_2",
                before_type="polyline",
                after_type="line",
                score_delta=-0.4,
            ),
            decision="reject",
            reason="Locked path cannot be auto-applied.",
            risk_flags=("locked_target",),
        ),
    )
    return AutoRefinementPipelineResult(
        refined_document=refined_document,
        candidates=(
            ShapeCandidate(
                candidate_id="cand_circle_1",
                target_type="circle",
                path_id="path_1",
                segment_range=(0, 0),
                source="raw_contour_points",
                confidence=0.94,
                evidence={"bbox": [8.0, 8.0, 24.0, 24.0]},
                reason="Outer contour is circular.",
            ),
            ShapeCandidate(
                candidate_id="cand_line_2",
                target_type="line",
                path_id="path_2",
                segment_range=(0, 0),
                source="segment_samples_fallback",
                confidence=0.62,
                evidence={"bbox": [38.0, 8.0, 24.0, 8.0]},
                reason="Segment reads as a straight line.",
            ),
        ),
        proposed_commands=tuple(decision.command for decision in preview_decisions),
        preview_decisions=preview_decisions,
        report=AutoRefinementReport(
            candidate_stats={"total": 2},
            command_stats={"total": 2},
            decision_stats={"auto_accept": 1, "user_confirm": 0, "reject": 1},
            score_before=10.0,
            score_after=7.5,
            integrity={"success": True, "errors": [], "warnings": []},
            dry_run_only=False,
            target_types=(),
        ),
    )


def _auto_result_for_apply() -> AutoRefinementPipelineResult:
    document = _make_document(segment_types=("polyline", "polyline"))
    preview_decisions = (
        PreviewDecision(
            command={
                "tool": "propose_replace_segment_with_line",
                "path_id": "path_1",
                "segment_range": [0, 0],
                "reason": "Straighten first path.",
                "confidence": 0.71,
                "requires_user_confirmation": True,
                "candidate_id": "cand_line_1",
            },
            preview_result=_preview_result(
                command_id="line_path_1",
                path_id="path_1",
                segment_id="seg_1",
                before_type="polyline",
                after_type="line",
                score_delta=-0.8,
            ),
            decision="user_confirm",
            reason="Needs user confirmation.",
            risk_flags=("needs_review",),
        ),
        PreviewDecision(
            command={
                "tool": "propose_replace_segment_with_line",
                "path_id": "path_2",
                "segment_range": [0, 0],
                "reason": "Straighten second path.",
                "confidence": 0.7,
                "requires_user_confirmation": True,
                "candidate_id": "cand_line_2",
            },
            preview_result=_preview_result(
                command_id="line_path_2",
                path_id="path_2",
                segment_id="seg_2",
                before_type="polyline",
                after_type="line",
                score_delta=-0.9,
            ),
            decision="user_confirm",
            reason="Needs user confirmation.",
            risk_flags=("needs_review",),
        ),
    )
    return AutoRefinementPipelineResult(
        refined_document=document,
        candidates=(
            ShapeCandidate(
                candidate_id="cand_line_1",
                target_type="line",
                path_id="path_1",
                segment_range=(0, 0),
                source="segment_samples_fallback",
                confidence=0.71,
                evidence={"bbox": [8.0, 8.0, 24.0, 8.0]},
                reason="First path is almost straight.",
            ),
            ShapeCandidate(
                candidate_id="cand_line_2",
                target_type="line",
                path_id="path_2",
                segment_range=(0, 0),
                source="segment_samples_fallback",
                confidence=0.7,
                evidence={"bbox": [38.0, 8.0, 24.0, 8.0]},
                reason="Second path is almost straight.",
            ),
        ),
        proposed_commands=tuple(decision.command for decision in preview_decisions),
        preview_decisions=preview_decisions,
        report=AutoRefinementReport(
            candidate_stats={"total": 2},
            command_stats={"total": 2},
            decision_stats={"auto_accept": 0, "user_confirm": 2, "reject": 0},
            score_before=10.0,
            score_after=10.0,
            integrity={"success": True, "errors": [], "warnings": []},
            dry_run_only=False,
            target_types=(),
        ),
    )


def test_main_window_auto_refinement_review_exposes_candidates_decisions_and_diff_summary() -> None:
    original_document = _make_document(segment_types=("polyline", "polyline"))
    window = MainWindow(
        auto_refinement_pipeline=FakeAutoRefinementPipeline(_auto_result_for_review()),
        canvas_widget=CanvasWidget(locked_ids=("path_2",)),
    )
    window.pipeline_result = _pipeline_result(original_document)

    display_state = window.trigger_auto_refinement_review_for_current_document()

    assert len(display_state.candidates) == 2
    assert len(display_state.preview_decisions) == 2
    assert display_state.diff_summary["score_delta"] == -2.5
    assert display_state.diff_summary["segment_type_counts"]["delta"]["circle"] == 1
    candidate_overlay = next(
        overlay for overlay in display_state.suggestion_overlays
        if overlay.source_type == "candidate" and overlay.candidate_id == "cand_circle_1"
    )
    decision_overlay = next(
        overlay for overlay in display_state.suggestion_overlays
        if overlay.source_type == "decision" and overlay.candidate_id == "cand_line_2"
    )
    assert candidate_overlay.review_state == "auto_accepted"
    assert candidate_overlay.decision == "auto_accepted"
    assert decision_overlay.review_state == "rejected"
    assert decision_overlay.locked_target_ids == ("path_2",)
    assert window.pipeline_result is not None
    assert window.pipeline_result.document.segments[0].type == "circle"


def test_main_window_apply_and_ignore_user_confirm_commands() -> None:
    original_document = _make_document(segment_types=("polyline", "polyline"))
    window = MainWindow(
        auto_refinement_pipeline=FakeAutoRefinementPipeline(_auto_result_for_apply()),
        canvas_widget=CanvasWidget(),
    )
    window.pipeline_result = _pipeline_result(original_document)
    window.trigger_auto_refinement_review_for_current_document()

    display_state = window.apply_user_confirm_command("decision_0")

    assert window.executed_commands[0]["path_id"] == "path_1"
    assert window.pipeline_result is not None
    assert window.pipeline_result.document.segments[0].type == "line"
    applied_decision = next(
        item for item in display_state.preview_decisions
        if item["decision_id"] == "decision_0"
    )
    assert applied_decision["review_state"] == "applied"
    assert display_state.candidates[0]["review_state"] == "applied"

    ignored_state = window.ignore_command("decision_1")
    ignored_decision = next(
        item for item in ignored_state.preview_decisions
        if item["decision_id"] == "decision_1"
    )
    assert ignored_decision["review_state"] == "ignored"


def test_main_window_batch_applies_pending_user_confirm_commands() -> None:
    original_document = _make_document(segment_types=("polyline", "polyline"))
    window = MainWindow(
        auto_refinement_pipeline=FakeAutoRefinementPipeline(_auto_result_for_apply()),
        canvas_widget=CanvasWidget(),
    )
    window.pipeline_result = _pipeline_result(original_document)
    window.trigger_auto_refinement_review_for_current_document()

    display_state = window.apply_all_user_confirm_commands()

    assert window.pipeline_result is not None
    assert [segment.type for segment in window.pipeline_result.document.segments] == ["line", "line"]
    assert len(window.executed_commands) == 2
    assert all(item["review_state"] == "applied" for item in display_state.preview_decisions)
    assert display_state.diff_summary["segment_type_counts"]["after"]["line"] == 2
