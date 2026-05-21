from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from core.types import VectorDocument
from services.command_executor import CommandExecutor
from services.command_preview import BatchCommandPreviewResult, CommandPreviewResult, CommandPreviewService
from services.command_schema import BATCH_TOOL
from services.document_integrity import DocumentIntegrityValidator, IntegrityReport

DecisionType = Literal["auto_accept", "user_confirm", "reject"]


@dataclass(frozen=True, slots=True)
class PreviewAndAutoAcceptPolicyConfig:
    auto_accept_min_confidence: float = 0.85
    auto_accept_min_score_improvement: float = 0.5
    reject_score_regression_over: float = 0.0
    medium_confidence_threshold: float = 0.55
    self_intersection_increase_decision: DecisionType = "reject"
    topology_regression_decision: DecisionType = "reject"
    locked_target_decision: DecisionType = "reject"
    batch_child_reject_decision: DecisionType = "user_confirm"


@dataclass(frozen=True, slots=True)
class PreviewDecision:
    command: dict[str, Any]
    preview_result: CommandPreviewResult | BatchCommandPreviewResult
    decision: DecisionType
    reason: str
    risk_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreviewPolicyResult:
    final_document: VectorDocument
    decisions: tuple[PreviewDecision, ...]
    accepted_count: int
    rejected_count: int
    user_confirm_count: int


class PreviewAndAutoAcceptPolicy:
    def __init__(
        self,
        *,
        preview_service: CommandPreviewService | None = None,
        command_executor: CommandExecutor | None = None,
        integrity_validator: DocumentIntegrityValidator | None = None,
        config: PreviewAndAutoAcceptPolicyConfig | None = None,
    ) -> None:
        self.preview_service = preview_service or CommandPreviewService()
        self.command_executor = command_executor or CommandExecutor()
        self.integrity_validator = integrity_validator or DocumentIntegrityValidator()
        self.config = config or PreviewAndAutoAcceptPolicyConfig()

    def evaluate_commands(
        self,
        commands: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        document: VectorDocument,
    ) -> PreviewPolicyResult:
        current_document = document
        decisions: list[PreviewDecision] = []

        for command in commands:
            if command.get("tool") == BATCH_TOOL:
                decision, next_document = self._evaluate_batch_command(command, current_document)
            else:
                decision, next_document = self._evaluate_single_command(command, current_document)
            decisions.append(decision)
            current_document = next_document

        return PreviewPolicyResult(
            final_document=current_document,
            decisions=tuple(decisions),
            accepted_count=sum(1 for item in decisions if item.decision == "auto_accept"),
            rejected_count=sum(1 for item in decisions if item.decision == "reject"),
            user_confirm_count=sum(1 for item in decisions if item.decision == "user_confirm"),
        )

    def _evaluate_batch_command(
        self,
        command: dict[str, Any],
        document: VectorDocument,
    ) -> tuple[PreviewDecision, VectorDocument]:
        nested_commands = [dict(item) for item in command.get("commands", ())]
        batch_preview = self.preview_service.preview_batch(nested_commands, document, continue_on_failure=True)

        working_document = document
        nested_decisions: list[PreviewDecision] = []
        for nested_command in nested_commands:
            nested_decision, next_document = self._evaluate_single_command(nested_command, working_document)
            nested_decisions.append(nested_decision)
            if nested_decision.decision == "auto_accept":
                working_document = next_document

        if not nested_decisions:
            return (
                PreviewDecision(
                    command=dict(command),
                    preview_result=batch_preview,
                    decision="reject",
                    reason="Batch command does not contain executable nested commands.",
                    risk_flags=("empty_batch",),
                ),
                document,
            )

        risk_flags = tuple(
            dict.fromkeys(
                flag
                for decision in nested_decisions
                for flag in decision.risk_flags
            )
        )
        if all(decision.decision == "auto_accept" for decision in nested_decisions):
            return (
                PreviewDecision(
                    command=dict(command),
                    preview_result=batch_preview,
                    decision="auto_accept",
                    reason="All nested preview commands met auto-accept policy.",
                    risk_flags=risk_flags,
                ),
                working_document,
            )
        if any(decision.decision == "reject" for decision in nested_decisions):
            batch_decision = self.config.batch_child_reject_decision
            return (
                PreviewDecision(
                    command=dict(command),
                    preview_result=batch_preview,
                    decision=batch_decision,
                    reason="At least one nested command was rejected by preview policy.",
                    risk_flags=tuple(dict.fromkeys(risk_flags + ("batch_contains_reject",))),
                ),
                document,
            )
        return (
            PreviewDecision(
                command=dict(command),
                preview_result=batch_preview,
                decision="user_confirm",
                reason="Batch contains commands that require user confirmation.",
                risk_flags=tuple(dict.fromkeys(risk_flags + ("batch_requires_confirmation",))),
            ),
            document,
        )

    def _evaluate_single_command(
        self,
        command: dict[str, Any],
        document: VectorDocument,
    ) -> tuple[PreviewDecision, VectorDocument]:
        preview = self.preview_service.preview(command, document)
        if not preview.success:
            return self._preview_failure_decision(command, preview, document)

        execution_result = self.command_executor.execute(command, document)
        if not execution_result.success:
            return (
                PreviewDecision(
                    command=dict(command),
                    preview_result=preview,
                    decision="reject",
                    reason=execution_result.reason or "Command execution failed after successful preview.",
                    risk_flags=("execution_failed",),
                ),
                document,
            )

        integrity_report = self.integrity_validator.validate(execution_result.document)
        blocking_decision = self._blocking_decision(command, preview, integrity_report)
        if blocking_decision is not None:
            return blocking_decision, document

        confidence = float(command.get("confidence", 0.0))
        score_improvement = self._score_improvement(preview)
        risk_flags: list[str] = []

        if confidence < self.config.auto_accept_min_confidence:
            risk_flags.append("medium_confidence")
        if score_improvement is None:
            risk_flags.append("missing_score_delta")
        elif score_improvement < self.config.auto_accept_min_score_improvement:
            risk_flags.append("limited_score_improvement")

        if (
            confidence >= self.config.auto_accept_min_confidence
            and score_improvement is not None
            and score_improvement >= self.config.auto_accept_min_score_improvement
        ):
            return (
                PreviewDecision(
                    command=dict(command),
                    preview_result=preview,
                    decision="auto_accept",
                    reason="Preview improved score enough and confidence is high.",
                    risk_flags=tuple(risk_flags),
                ),
                execution_result.document,
            )

        return (
            PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="user_confirm",
                reason="Preview is safe enough to review, but confidence or score gain is not high enough for auto-accept.",
                risk_flags=tuple(dict.fromkeys(risk_flags or ["needs_review"])),
            ),
            document,
        )

    def _preview_failure_decision(
        self,
        command: dict[str, Any],
        preview: CommandPreviewResult,
        document: VectorDocument,
    ) -> tuple[PreviewDecision, VectorDocument]:
        reason_text = (preview.reason or "").lower()
        if "locked " in reason_text or "locked_" in reason_text:
            decision = self.config.locked_target_decision
            return (
                PreviewDecision(
                    command=dict(command),
                    preview_result=preview,
                    decision=decision,
                    reason=preview.reason or "Preview hit locked targets.",
                    risk_flags=("locked_target", "preview_failed"),
                ),
                document,
            )
        return (
            PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason=preview.reason or "Preview failed.",
                risk_flags=("preview_failed",),
            ),
            document,
        )

    def _blocking_decision(
        self,
        command: dict[str, Any],
        preview: CommandPreviewResult,
        integrity_report: IntegrityReport,
    ) -> PreviewDecision | None:
        if not integrity_report.success:
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason=self._integrity_reason(integrity_report),
                risk_flags=("integrity_failed",),
            )

        if self._self_intersection_increased(preview):
            decision = self.config.self_intersection_increase_decision
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision=decision,
                reason="Preview increases self-intersection risk.",
                risk_flags=("self_intersection_increase",),
            )

        if self._topology_regressed(preview):
            decision = self.config.topology_regression_decision
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision=decision,
                reason="Preview worsens topology status.",
                risk_flags=("topology_regression",),
            )

        if preview.score_delta is not None and preview.score_delta > self.config.reject_score_regression_over:
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Preview regresses the document score.",
                risk_flags=("score_regression",),
            )

        return None

    def _self_intersection_increased(self, preview: CommandPreviewResult) -> bool:
        for path_id, after_count in preview.self_intersection_count_after.items():
            before_count = preview.self_intersection_count_before.get(path_id, 0)
            if after_count > before_count:
                return True
        return False

    def _topology_regressed(self, preview: CommandPreviewResult) -> bool:
        for path_id, after_status in preview.topology_status_after.items():
            before_status = preview.topology_status_before.get(path_id)
            if self._topology_rank(after_status) > self._topology_rank(before_status):
                return True
        return False

    def _topology_rank(self, status: str | None) -> int:
        order = {
            "closed": 0,
            "open": 1,
            "unknown": 1,
            "topology_error": 2,
        }
        return order.get(status or "unknown", 1)

    def _score_improvement(self, preview: CommandPreviewResult) -> float | None:
        if preview.score_delta is None:
            return None
        return -float(preview.score_delta)

    def _integrity_reason(self, report: IntegrityReport) -> str:
        first_error = report.errors[0] if report.errors else None
        if first_error is None:
            return "Document integrity validation failed."
        return f"Document integrity validation failed: {first_error.code}"


__all__ = [
    "DecisionType",
    "PreviewAndAutoAcceptPolicy",
    "PreviewAndAutoAcceptPolicyConfig",
    "PreviewDecision",
    "PreviewPolicyResult",
]
