from __future__ import annotations

from dataclasses import dataclass

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment, VectorDocument, updated
from services.command_executor import CommandExecutionResult
from services.command_preview import (
    BatchCommandPreviewResult,
    CommandPreviewResult,
    CommandPreviewService,
    ConstraintChangeSummary,
    ExportImpactSummary,
)
from services.document_integrity import DocumentIntegrityValidator, IntegrityIssue, IntegrityReport
from services.preview_auto_accept_policy import PreviewAndAutoAcceptPolicy, PreviewAndAutoAcceptPolicyConfig


def _document(document_id: str = "doc_policy") -> VectorDocument:
    document = create_document(
        document_id=document_id,
        width=120.0,
        height=120.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_1", closed=True, segments=("seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="seg_1",
            path_id="path_1",
            type="polyline",
            params={"points": [[0.0, 0.0], [4.0, 1.0], [8.0, 0.0], [0.0, 0.0]]},
        ),
    )
    return document


def _preview_result(
    *,
    command_id: str,
    success: bool = True,
    reason: str | None = None,
    score_delta: float | None = -1.0,
    topology_before: str = "closed",
    topology_after: str = "closed",
    self_intersection_before: int = 0,
    self_intersection_after: int = 0,
) -> CommandPreviewResult:
    old_score = 10.0 if score_delta is not None else None
    predicted_new_score = (old_score + score_delta) if score_delta is not None and old_score is not None else None
    return CommandPreviewResult(
        success=success,
        command_id=command_id,
        reason=reason,
        old_score=old_score,
        predicted_new_score=predicted_new_score,
        score_delta=score_delta,
        affected_paths=("path_1",),
        affected_segments=("seg_1",),
        topology_status_before={"path_1": topology_before},
        topology_status_after={"path_1": topology_after},
        self_intersection_count_before={"path_1": self_intersection_before},
        self_intersection_count_after={"path_1": self_intersection_after},
        segment_type_summary={"before": {"polyline": 1}, "after": {"circle": 1}, "delta": {"polyline": -1, "circle": 1}},
        constraint_change_summary=ConstraintChangeSummary(
            before={},
            after={},
            delta={},
            added_constraint_ids=(),
            removed_constraint_ids=(),
            changed_constraint_ids=(),
        ),
        export_impact_summary=ExportImpactSummary(before={"json_char_count": 10}, after={"json_char_count": 9}, delta={"json_char_count": -1}),
    )


class FakePreviewService(CommandPreviewService):
    def __init__(
        self,
        previews: dict[str, CommandPreviewResult],
        *,
        batch_preview: BatchCommandPreviewResult | None = None,
    ) -> None:
        self._previews = previews
        self._batch_preview = batch_preview

    def preview(self, command: object, document: VectorDocument) -> CommandPreviewResult:
        assert isinstance(command, dict)
        return self._previews[str(command["command_id"])]

    def preview_batch(
        self,
        commands: list[object] | tuple[object, ...],
        document: VectorDocument,
        *,
        continue_on_failure: bool = True,
    ) -> BatchCommandPreviewResult:
        if self._batch_preview is not None:
            return self._batch_preview
        previews = tuple(self.preview(command, document) for command in commands if isinstance(command, dict))
        return BatchCommandPreviewResult(
            batch_id="preview_batch_test",
            success_count=sum(1 for item in previews if item.success),
            failure_count=sum(1 for item in previews if not item.success),
            previews=previews,
        )


class FakeCommandExecutor:
    def __init__(self, handler):
        self._handler = handler

    def execute(self, command: object, document: VectorDocument) -> CommandExecutionResult:
        assert isinstance(command, dict)
        return self._handler(command, document)


class FakeIntegrityValidator(DocumentIntegrityValidator):
    def __init__(self, reports: dict[str, IntegrityReport] | None = None) -> None:
        self._reports = reports or {}

    def validate(self, document: VectorDocument) -> IntegrityReport:
        return self._reports.get(
            document.document_id,
            IntegrityReport(success=True, errors=(), warnings=(), affected_ids=()),
        )


def _execution_result(command_id: str, document: VectorDocument, *, success: bool = True, reason: str | None = None) -> CommandExecutionResult:
    return CommandExecutionResult(
        success=success,
        command_id=command_id,
        document=document,
        affected_paths=("path_1",),
        affected_segments=("seg_1",),
        old_score=10.0,
        new_score=9.0 if success else None,
        topology_status="closed",
        self_intersection_count=0,
        requires_rerender=success,
        fitting_source="raw_contour_points" if success else None,
        reason=reason,
    )


def test_preview_auto_accept_policy_auto_accepts_high_confidence_circle_preview() -> None:
    document = _document()
    command = {
        "command_id": "circle_ok",
        "tool": "propose_replace_path_with_circle",
        "path_id": "path_1",
        "reason": "intent only",
        "confidence": 0.94,
        "requires_user_confirmation": True,
    }
    preview_service = FakePreviewService({"circle_ok": _preview_result(command_id="circle_ok", score_delta=-1.2)})
    executor = FakeCommandExecutor(
        lambda command, doc: _execution_result(command["command_id"], updated(doc, document_id="doc_policy:auto_circle"))
    )

    result = PreviewAndAutoAcceptPolicy(
        preview_service=preview_service,
        command_executor=executor,
        integrity_validator=FakeIntegrityValidator(),
    ).evaluate_commands([command], document)

    assert result.accepted_count == 1
    assert result.rejected_count == 0
    assert result.user_confirm_count == 0
    assert result.decisions[0].decision == "auto_accept"
    assert result.final_document.document_id == "doc_policy:auto_circle"


def test_preview_auto_accept_policy_marks_low_confidence_preview_for_user_confirmation() -> None:
    document = _document()
    command = {
        "command_id": "circle_review",
        "tool": "propose_replace_path_with_circle",
        "path_id": "path_1",
        "reason": "intent only",
        "confidence": 0.62,
        "requires_user_confirmation": True,
    }
    preview_service = FakePreviewService({"circle_review": _preview_result(command_id="circle_review", score_delta=-0.2)})
    executor = FakeCommandExecutor(
        lambda command, doc: _execution_result(command["command_id"], updated(doc, document_id="doc_policy:review"))
    )

    result = PreviewAndAutoAcceptPolicy(
        preview_service=preview_service,
        command_executor=executor,
        integrity_validator=FakeIntegrityValidator(),
    ).evaluate_commands([command], document)

    assert result.accepted_count == 0
    assert result.user_confirm_count == 1
    assert result.decisions[0].decision == "user_confirm"
    assert "medium_confidence" in result.decisions[0].risk_flags
    assert result.final_document == document


def test_preview_auto_accept_policy_rejects_preview_failure() -> None:
    document = _document()
    command = {
        "command_id": "circle_fail",
        "tool": "propose_replace_path_with_circle",
        "path_id": "path_1",
        "reason": "intent only",
        "confidence": 0.9,
        "requires_user_confirmation": True,
    }
    preview_service = FakePreviewService(
        {"circle_fail": _preview_result(command_id="circle_fail", success=False, reason="fit failed", score_delta=None)}
    )

    result = PreviewAndAutoAcceptPolicy(
        preview_service=preview_service,
        command_executor=FakeCommandExecutor(lambda command, doc: _execution_result(command["command_id"], doc, success=False, reason="fit failed")),
        integrity_validator=FakeIntegrityValidator(),
    ).evaluate_commands([command], document)

    assert result.rejected_count == 1
    assert result.decisions[0].decision == "reject"
    assert result.decisions[0].risk_flags == ("preview_failed",)
    assert result.final_document == document


def test_preview_auto_accept_policy_rejects_topology_regression() -> None:
    document = _document()
    command = {
        "command_id": "topology_bad",
        "tool": "propose_replace_segment_with_line",
        "path_id": "path_1",
        "segment_range": [0, 0],
        "reason": "intent only",
        "confidence": 0.93,
        "requires_user_confirmation": True,
    }
    preview_service = FakePreviewService(
        {"topology_bad": _preview_result(command_id="topology_bad", score_delta=-0.8, topology_before="closed", topology_after="topology_error")}
    )
    executor = FakeCommandExecutor(
        lambda command, doc: _execution_result(command["command_id"], updated(doc, document_id="doc_policy:topology_bad"))
    )

    result = PreviewAndAutoAcceptPolicy(
        preview_service=preview_service,
        command_executor=executor,
        integrity_validator=FakeIntegrityValidator(),
    ).evaluate_commands([command], document)

    assert result.rejected_count == 1
    assert result.decisions[0].decision == "reject"
    assert result.decisions[0].risk_flags == ("topology_regression",)
    assert result.final_document == document


def test_preview_auto_accept_policy_does_not_auto_accept_locked_target_failures() -> None:
    document = _document()
    command = {
        "command_id": "locked_target",
        "tool": "propose_replace_segment_with_line",
        "path_id": "path_1",
        "segment_range": [0, 0],
        "reason": "intent only",
        "confidence": 0.95,
        "requires_user_confirmation": True,
    }
    preview_service = FakePreviewService(
        {"locked_target": _preview_result(command_id="locked_target", success=False, reason="locked path cannot be modified", score_delta=None)}
    )

    result = PreviewAndAutoAcceptPolicy(
        preview_service=preview_service,
        command_executor=FakeCommandExecutor(lambda command, doc: _execution_result(command["command_id"], doc, success=False, reason="locked path")),
        integrity_validator=FakeIntegrityValidator(),
        config=PreviewAndAutoAcceptPolicyConfig(locked_target_decision="user_confirm"),
    ).evaluate_commands([command], document)

    assert result.accepted_count == 0
    assert result.user_confirm_count == 1
    assert result.decisions[0].decision == "user_confirm"
    assert "locked_target" in result.decisions[0].risk_flags


def test_preview_auto_accept_policy_rejects_integrity_failures_and_applies_auto_accepts_sequentially() -> None:
    document = _document()
    commands = [
        {
            "command_id": "auto_1",
            "tool": "propose_replace_path_with_circle",
            "path_id": "path_1",
            "reason": "intent only",
            "confidence": 0.92,
            "requires_user_confirmation": True,
        },
        {
            "command_id": "bad_integrity",
            "tool": "propose_replace_segment_with_line",
            "path_id": "path_1",
            "segment_range": [0, 0],
            "reason": "intent only",
            "confidence": 0.9,
            "requires_user_confirmation": True,
        },
        {
            "command_id": "auto_2",
            "tool": "propose_replace_segment_with_arc",
            "path_id": "path_1",
            "segment_range": [0, 0],
            "reason": "intent only",
            "confidence": 0.93,
            "requires_user_confirmation": True,
        },
    ]
    preview_service = FakePreviewService(
        {
            "auto_1": _preview_result(command_id="auto_1", score_delta=-1.0),
            "bad_integrity": _preview_result(command_id="bad_integrity", score_delta=-0.7),
            "auto_2": _preview_result(command_id="auto_2", score_delta=-0.9),
        }
    )

    def handler(command: dict[str, object], doc: VectorDocument) -> CommandExecutionResult:
        next_doc = updated(doc, document_id=f"{doc.document_id}:{command['command_id']}")
        return _execution_result(str(command["command_id"]), next_doc)

    integrity_reports = {
        "doc_policy:auto_1:bad_integrity": IntegrityReport(
            success=False,
            errors=(IntegrityIssue(code="DANGLING_SEGMENT", message="dangling", affected_ids=("seg_1",)),),
            warnings=(),
            affected_ids=("seg_1",),
        )
    }

    result = PreviewAndAutoAcceptPolicy(
        preview_service=preview_service,
        command_executor=FakeCommandExecutor(handler),
        integrity_validator=FakeIntegrityValidator(integrity_reports),
    ).evaluate_commands(commands, document)

    assert [decision.decision for decision in result.decisions] == ["auto_accept", "reject", "auto_accept"]
    assert result.accepted_count == 2
    assert result.rejected_count == 1
    assert result.final_document.document_id == "doc_policy:auto_1:auto_2"
    assert result.decisions[1].risk_flags == ("integrity_failed",)


def test_preview_auto_accept_policy_handles_batch_commands_without_mutating_on_review_needed() -> None:
    document = _document()
    batch_command = {
        "command_id": "batch_1",
        "tool": "propose_batch_refinement",
        "summary": "batch",
        "commands": [
            {
                "command_id": "batch_child_ok",
                "tool": "propose_replace_path_with_circle",
                "path_id": "path_1",
                "reason": "intent only",
                "confidence": 0.91,
                "requires_user_confirmation": True,
            },
            {
                "command_id": "batch_child_review",
                "tool": "propose_replace_segment_with_line",
                "path_id": "path_1",
                "segment_range": [0, 0],
                "reason": "intent only",
                "confidence": 0.6,
                "requires_user_confirmation": True,
            },
        ],
        "confidence": 0.82,
        "requires_user_confirmation": True,
    }
    preview_service = FakePreviewService(
        {
            "batch_child_ok": _preview_result(command_id="batch_child_ok", score_delta=-1.0),
            "batch_child_review": _preview_result(command_id="batch_child_review", score_delta=-0.1),
        },
        batch_preview=BatchCommandPreviewResult(
            batch_id="batch_1",
            success_count=2,
            failure_count=0,
            previews=(
                _preview_result(command_id="batch_child_ok", score_delta=-1.0),
                _preview_result(command_id="batch_child_review", score_delta=-0.1),
            ),
        ),
    )
    executor = FakeCommandExecutor(
        lambda command, doc: _execution_result(str(command["command_id"]), updated(doc, document_id=f"{doc.document_id}:{command['command_id']}"))
    )

    result = PreviewAndAutoAcceptPolicy(
        preview_service=preview_service,
        command_executor=executor,
        integrity_validator=FakeIntegrityValidator(),
    ).evaluate_commands([batch_command], document)

    assert result.decisions[0].decision == "user_confirm"
    assert "batch_requires_confirmation" in result.decisions[0].risk_flags
    assert result.final_document == document
