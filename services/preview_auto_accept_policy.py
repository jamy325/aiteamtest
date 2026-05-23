from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from core.types import VectorDocument
from services.command_executor import CommandExecutor
from services.command_preview import BatchCommandPreviewResult, CommandPreviewResult, CommandPreviewService
from services.command_schema import BATCH_TOOL
from services.document_integrity import DocumentIntegrityValidator, IntegrityReport
from services.engine_protocol import (
    AutonomyLevel,
    DecisionKind,
    DecisionPolicyResult,
    ExternalDecisionRequest,
    PolicyFeedback,
    RiskLevel,
)

DecisionType = Literal["auto_accept", "user_confirm", "reject"]


@dataclass(frozen=True, slots=True)
class PreviewAndAutoAcceptPolicyConfig:
    autonomy_level: AutonomyLevel = AutonomyLevel.AUTONOMOUS_SAFE
    allowed_auto_risk: RiskLevel = RiskLevel.MEDIUM
    auto_accept_min_confidence: float = 0.73
    min_fitting_confidence: float = 0.55
    min_inlier_ratio: float = 0.6
    auto_accept_min_score_improvement: float = 0.5
    reject_score_regression_over: float = 0.0
    max_fit_error_increase: float = 0.0
    max_complexity_increase: float = 0.0
    min_edge_error_gain_for_complexity_increase: float = 0.01
    max_constraint_violation_increase: int = 0
    max_affected_segments: int = 24
    medium_confidence_threshold: float = 0.73
    batch_child_reject_decision: DecisionType = "user_confirm"


@dataclass(frozen=True, slots=True)
class PreviewDecision:
    command: dict[str, Any]
    preview_result: CommandPreviewResult | BatchCommandPreviewResult
    decision: DecisionType
    reason: str
    risk_flags: tuple[str, ...]
    decision_kind: DecisionKind | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    policy_feedback: PolicyFeedback | None = None
    external_decision_request: ExternalDecisionRequest | None = None
    policy_result: DecisionPolicyResult | None = None

    def __post_init__(self) -> None:
        decision_kind = self.decision_kind or _decision_kind_from_legacy(self.decision)
        policy_feedback = self.policy_feedback
        external_request = self.external_decision_request
        policy_result = self.policy_result
        if policy_result is not None:
            decision_kind = policy_result.decision
            policy_feedback = policy_result.policy_feedback
            external_request = policy_result.external_decision_request
            object.__setattr__(self, "risk_level", policy_result.risk_level)
        elif decision_kind == DecisionKind.REQUIRES_EXTERNAL_DECISION and external_request is not None:
            policy_result = DecisionPolicyResult(
                decision=decision_kind,
                reason_code=policy_feedback.reason_code if policy_feedback is not None else "requires_external_decision",
                risk_level=self.risk_level,
                policy_feedback=policy_feedback,
                external_decision_request=external_request,
            )
        elif policy_feedback is not None:
            policy_result = DecisionPolicyResult(
                decision=decision_kind,
                reason_code=policy_feedback.reason_code,
                risk_level=self.risk_level,
                policy_feedback=policy_feedback,
                external_decision_request=external_request,
            )

        object.__setattr__(self, "decision_kind", decision_kind)
        object.__setattr__(self, "policy_feedback", policy_feedback)
        object.__setattr__(self, "external_decision_request", external_request)
        object.__setattr__(self, "policy_result", policy_result)


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
                    risk_level=RiskLevel.MEDIUM_HIGH,
                    policy_feedback=PolicyFeedback(
                        reason_code="empty_batch",
                        message="Batch command does not contain executable nested commands.",
                        metrics_delta={"command_count": 0},
                        policy_hint="submit at least one valid nested command",
                        retry_allowed=False,
                    ),
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
        if all(decision.decision_kind == DecisionKind.AUTO_APPLY for decision in nested_decisions):
            return (
                PreviewDecision(
                    command=dict(command),
                    preview_result=batch_preview,
                    decision="auto_accept",
                    reason="All nested preview commands met auto-accept policy.",
                    risk_flags=risk_flags,
                    decision_kind=DecisionKind.AUTO_APPLY,
                    risk_level=max((decision.risk_level for decision in nested_decisions), default=RiskLevel.LOW),
                    policy_feedback=PolicyFeedback(
                        reason_code="batch_auto_apply",
                        message="All nested commands satisfied auto-apply gates.",
                        metrics_delta={"nested_command_count": len(nested_decisions)},
                        policy_hint="commit batch",
                        retry_allowed=False,
                    ),
                ),
                working_document,
            )
        if any(decision.decision_kind == DecisionKind.AUTO_REJECT for decision in nested_decisions):
            risk_level = max((decision.risk_level for decision in nested_decisions), default=RiskLevel.MEDIUM_HIGH)
            feedback = PolicyFeedback(
                reason_code="batch_contains_reject",
                message="At least one nested command was rejected by preview policy.",
                metrics_delta={"nested_command_count": len(nested_decisions)},
                policy_hint="inspect rejected child command before retrying batch",
                retry_allowed=True,
                retry_constraints={"max_batch_size": len(nested_decisions)},
            )
            return (
                PreviewDecision(
                    command=dict(command),
                    preview_result=batch_preview,
                    decision="user_confirm",
                    reason="At least one nested command was rejected by preview policy.",
                    risk_flags=tuple(dict.fromkeys(risk_flags + ("batch_contains_reject",))),
                    decision_kind=DecisionKind.REQUIRES_EXTERNAL_DECISION,
                    risk_level=risk_level,
                    policy_feedback=feedback,
                    external_decision_request=self._external_decision_request(
                        command=dict(command),
                        reason="Batch contains at least one rejected nested command.",
                        risk_flags=tuple(dict.fromkeys(risk_flags + ("batch_contains_reject",))),
                        risk_level=risk_level,
                        preview_summary={"nested_decisions": len(nested_decisions)},
                        policy_feedback=feedback,
                    ),
                ),
                document,
            )
        risk_level = max((decision.risk_level for decision in nested_decisions), default=RiskLevel.MEDIUM)
        feedback = PolicyFeedback(
            reason_code="batch_requires_external_decision",
            message="Batch contains commands that require external decision.",
            metrics_delta={"nested_command_count": len(nested_decisions)},
            policy_hint="review nested command mix before apply",
            retry_allowed=True,
            retry_constraints={"max_batch_size": len(nested_decisions)},
        )
        return (
            PreviewDecision(
                command=dict(command),
                preview_result=batch_preview,
                decision="user_confirm",
                reason="Batch contains commands that require user confirmation.",
                risk_flags=tuple(dict.fromkeys(risk_flags + ("batch_requires_confirmation",))),
                decision_kind=DecisionKind.REQUIRES_EXTERNAL_DECISION,
                risk_level=risk_level,
                policy_feedback=feedback,
                external_decision_request=self._external_decision_request(
                    command=dict(command),
                    reason="Batch requires external decision.",
                    risk_flags=tuple(dict.fromkeys(risk_flags + ("batch_requires_confirmation",))),
                    risk_level=risk_level,
                    preview_summary={"nested_decisions": len(nested_decisions)},
                    policy_feedback=feedback,
                ),
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
        if preview.preview_document is None:
            return (
                PreviewDecision(
                    command=dict(command),
                    preview_result=preview,
                    decision="reject",
                    reason="Preview succeeded without producing a committable preview document.",
                    risk_flags=("missing_preview_document",),
                    decision_kind=DecisionKind.AUTO_REJECT,
                    risk_level=self._risk_level(command),
                    policy_feedback=PolicyFeedback(
                        reason_code="missing_preview_document",
                        message="Preview succeeded without producing a committable preview document.",
                        metrics_delta={},
                        policy_hint="repair preview generation before retry",
                        retry_allowed=True,
                    ),
                ),
                document,
            )

        integrity_report = self.integrity_validator.validate(preview.preview_document)
        blocking_decision = self._blocking_decision(command, preview, integrity_report, document, preview.preview_document)
        if blocking_decision is not None:
            return blocking_decision, document

        algorithm_confidence = self._algorithm_fitting_confidence(preview)
        score_improvement = self._score_improvement(preview)
        risk_level = self._risk_level(command)
        policy_metrics = self._policy_metrics(command)
        risk_flags: list[str] = []
        metrics_delta = self._metrics_delta(preview, command)

        if algorithm_confidence is not None and algorithm_confidence < self.config.min_fitting_confidence:
            risk_flags.append("low_fitting_confidence")
        if score_improvement is None:
            risk_flags.append("missing_score_delta")
        elif score_improvement < self.config.auto_accept_min_score_improvement:
            risk_flags.append("limited_score_improvement")
        if risk_level > self.config.allowed_auto_risk:
            risk_flags.append("risk_level_requires_escalation")
        if len(preview.affected_segments) > self.config.max_affected_segments:
            risk_flags.append("affected_scope_too_large")
        if not bool(policy_metrics.get("rollback_snapshot_created", True)):
            risk_flags.append("missing_rollback_snapshot")

        if self.config.autonomy_level == AutonomyLevel.MANUAL_ONLY:
            risk_flags.append("manual_only_mode")
        elif self.config.autonomy_level == AutonomyLevel.ASSISTED and risk_level > RiskLevel.LOW:
            risk_flags.append("assisted_mode_requires_escalation")

        if not risk_flags:
            feedback = PolicyFeedback(
                reason_code="auto_apply",
                message="Preview met all hard gates for auto-apply.",
                metrics_delta=metrics_delta,
                policy_hint="commit preview result",
                retry_allowed=False,
            )
            return (
                PreviewDecision(
                    command=dict(command),
                    preview_result=preview,
                    decision="auto_accept",
                    reason="Preview improved score enough and algorithm fitting confidence is high.",
                    risk_flags=(),
                    decision_kind=DecisionKind.AUTO_APPLY,
                    risk_level=risk_level,
                    policy_feedback=feedback,
                ),
                preview.preview_document,
            )

        feedback = PolicyFeedback(
            reason_code="requires_external_decision",
            message="Preview did not hit reject gates, but did not satisfy auto-apply hard gates.",
            metrics_delta=metrics_delta,
            policy_hint="review scope, risk, or fitting metrics before applying",
            retry_allowed=True,
            retry_constraints={
                "allowed_auto_risk": self.config.allowed_auto_risk.value,
                "max_affected_segments": self.config.max_affected_segments,
            },
            forbidden_repeated_commands=tuple(_forbidden_repeated_commands(command, risk_flags)),
        )
        return (
            PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="user_confirm",
                reason="Preview is safe enough to review, but fitting metrics or score gain are not high enough for auto-apply.",
                risk_flags=tuple(dict.fromkeys(risk_flags or ["needs_review"])),
                decision_kind=DecisionKind.REQUIRES_EXTERNAL_DECISION,
                risk_level=risk_level,
                policy_feedback=feedback,
                external_decision_request=self._external_decision_request(
                    command=dict(command),
                    reason="Preview requires external decision.",
                    risk_flags=tuple(dict.fromkeys(risk_flags or ["needs_review"])),
                    risk_level=risk_level,
                    preview_summary=self._preview_summary(preview, command),
                    policy_feedback=feedback,
                ),
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
            feedback = PolicyFeedback(
                reason_code="locked_target_modified",
                message=preview.reason or "Preview hit locked targets.",
                metrics_delta=self._metrics_delta(preview, command),
                policy_hint="do not target locked geometry",
                retry_allowed=False,
            )
            return (
                PreviewDecision(
                    command=dict(command),
                    preview_result=preview,
                    decision="reject",
                    reason=preview.reason or "Preview hit locked targets.",
                    risk_flags=("locked_target", "preview_failed"),
                    decision_kind=DecisionKind.AUTO_REJECT,
                    risk_level=self._risk_level(command),
                    policy_feedback=feedback,
                ),
                document,
            )
        reason_code = "schema_invalid" if "schema" in reason_text else "dry_run_failed"
        return (
            PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason=preview.reason or "Preview failed.",
                risk_flags=("preview_failed",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=self._risk_level(command),
                policy_feedback=PolicyFeedback(
                    reason_code=reason_code,
                    message=preview.reason or "Preview failed.",
                    metrics_delta=self._metrics_delta(preview, command),
                    policy_hint="fix preview failure before retry",
                    retry_allowed=reason_code != "schema_invalid",
                ),
            ),
            document,
        )

    def _blocking_decision(
        self,
        command: dict[str, Any],
        preview: CommandPreviewResult,
        integrity_report: IntegrityReport,
        before_document: VectorDocument,
        after_document: VectorDocument,
    ) -> PreviewDecision | None:
        risk_level = self._risk_level(command)
        policy_metrics = self._policy_metrics(command)
        metrics_delta = self._metrics_delta(preview, command)

        if not integrity_report.success:
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason=self._integrity_reason(integrity_report),
                risk_flags=("integrity_failed",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="integrity_failed",
                    message=self._integrity_reason(integrity_report),
                    metrics_delta=metrics_delta,
                    policy_hint="repair integrity issues before retry",
                    retry_allowed=True,
                ),
            )

        if policy_metrics.get("locked_target_modified", False):
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Preview modifies a locked target.",
                risk_flags=("locked_target",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="locked_target_modified",
                    message="Preview modifies a locked target.",
                    metrics_delta=metrics_delta,
                    policy_hint="remove locked target from scope",
                    retry_allowed=False,
                ),
            )

        if self._self_intersection_increased(preview):
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Preview increases self-intersection risk.",
                risk_flags=("self_intersection_increase",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="self_intersection_increase",
                    message="Preview increases self-intersection risk.",
                    metrics_delta=metrics_delta,
                    policy_hint="reduce scope or choose simpler geometry",
                    retry_allowed=True,
                ),
            )

        if self._topology_regressed(preview):
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Preview worsens topology status.",
                risk_flags=("topology_regression",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="topology_regression",
                    message="Preview worsens topology status.",
                    metrics_delta=metrics_delta,
                    policy_hint="fix topology before retry",
                    retry_allowed=True,
                ),
            )

        if preview.score_delta is not None and preview.score_delta > self.config.reject_score_regression_over:
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Preview regresses the document score.",
                risk_flags=("score_regression",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="score_regression",
                    message="Preview regresses the document score.",
                    metrics_delta=metrics_delta,
                    policy_hint="improve score delta before retry",
                    retry_allowed=True,
                ),
            )

        if float(policy_metrics.get("fit_error_delta", 0.0)) > self.config.max_fit_error_increase:
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Preview increases fit error beyond the allowed threshold.",
                risk_flags=("fit_error_increased",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="fit_error_increased",
                    message="Preview increases fit error beyond the allowed threshold.",
                    metrics_delta=metrics_delta,
                    policy_hint="reduce fit error before retry",
                    retry_allowed=True,
                    retry_constraints={"max_fit_error_increase": self.config.max_fit_error_increase},
                ),
            )

        complexity_delta = float(policy_metrics.get("complexity_delta", 0.0))
        edge_error_gain = self._edge_error_gain(policy_metrics)
        if (
            complexity_delta > self.config.max_complexity_increase
            and edge_error_gain < self.config.min_edge_error_gain_for_complexity_increase
        ):
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Preview increases complexity without enough edge-error gain.",
                risk_flags=("complexity_increase_without_edge_gain",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="complexity_increase_without_edge_gain",
                    message="Preview increases complexity without enough edge-error gain.",
                    metrics_delta=metrics_delta,
                    policy_hint="only accept extra complexity when edge error drops meaningfully",
                    retry_allowed=True,
                    retry_constraints={
                        "max_complexity_increase": self.config.max_complexity_increase,
                        "min_edge_error_gain_for_complexity_increase": self.config.min_edge_error_gain_for_complexity_increase,
                    },
                ),
            )

        algorithm_confidence = self._algorithm_fitting_confidence(preview)
        if algorithm_confidence is None:
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Preview is missing algorithm fitting confidence.",
                risk_flags=("missing_algorithm_fitting_confidence",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="missing_algorithm_fitting_confidence",
                    message="Preview is missing algorithm fitting confidence.",
                    metrics_delta=metrics_delta,
                    policy_hint="recompute deterministic fitting metrics before retry",
                    retry_allowed=True,
                ),
            )

        if algorithm_confidence < self.config.min_fitting_confidence:
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Algorithm fitting confidence is below the minimum fitting threshold.",
                risk_flags=("low_fitting_confidence",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="low_fitting_confidence",
                    message="Algorithm fitting confidence is below the minimum fitting threshold.",
                    metrics_delta=metrics_delta,
                    policy_hint="use a narrower scope or simpler primitive",
                    retry_allowed=True,
                    retry_constraints={"min_required_confidence": self.config.min_fitting_confidence},
                ),
            )

        if self._inlier_ratio(preview, command) < self.config.min_inlier_ratio:
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Preview inlier ratio is below the minimum threshold.",
                risk_flags=("low_inlier_ratio",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="low_inlier_ratio",
                    message="Preview inlier ratio is below the minimum threshold.",
                    metrics_delta=metrics_delta,
                    policy_hint="retry with adjusted breakpoints or a different primitive",
                    retry_allowed=True,
                    retry_constraints={"min_inlier_ratio": self.config.min_inlier_ratio},
                ),
            )

        if int(policy_metrics.get("constraint_violation_delta", 0)) > self.config.max_constraint_violation_increase:
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Preview increases constraint violations beyond the allowed threshold.",
                risk_flags=("constraint_violation_increase",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="constraint_violation_increase",
                    message="Preview increases constraint violations beyond the allowed threshold.",
                    metrics_delta=metrics_delta,
                    policy_hint="reduce affected scope or preserve constraints",
                    retry_allowed=True,
                    retry_constraints={
                        "max_constraint_violation_increase": self.config.max_constraint_violation_increase,
                    },
                ),
            )

        if not self._coordinate_system_consistent(before_document, after_document):
            return PreviewDecision(
                command=dict(command),
                preview_result=preview,
                decision="reject",
                reason="Preview changes the coordinate system unexpectedly.",
                risk_flags=("coordinate_system_inconsistent",),
                decision_kind=DecisionKind.AUTO_REJECT,
                risk_level=risk_level,
                policy_feedback=PolicyFeedback(
                    reason_code="coordinate_system_inconsistent",
                    message="Preview changes the coordinate system unexpectedly.",
                    metrics_delta=metrics_delta,
                    policy_hint="preserve coordinate system invariants",
                    retry_allowed=False,
                ),
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

    def _risk_level(self, command: dict[str, Any]) -> RiskLevel:
        explicit = command.get("risk_level")
        if explicit is not None:
            return RiskLevel(str(explicit))
        tool = str(command.get("tool", ""))
        if "bezier" in tool:
            return RiskLevel.HIGH
        if tool == BATCH_TOOL:
            return RiskLevel.MEDIUM_HIGH
        if "ellipse" in tool:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _policy_metrics(self, command: dict[str, Any]) -> dict[str, Any]:
        raw = command.get("policy_metrics")
        if isinstance(raw, dict):
            return dict(raw)
        return {}

    def _algorithm_fitting_confidence(self, preview: CommandPreviewResult) -> float | None:
        if preview.algorithm_fitting_confidence is None:
            return None
        return float(preview.algorithm_fitting_confidence)

    def _inlier_ratio(self, preview: CommandPreviewResult, command: dict[str, Any]) -> float:
        if preview.inlier_ratio is not None:
            return float(preview.inlier_ratio)
        value = self._policy_metrics(command).get("inlier_ratio", 1.0)
        return float(value) if isinstance(value, (int, float)) else 1.0

    def _fit_error(self, preview: CommandPreviewResult, command: dict[str, Any]) -> float | None:
        if preview.fit_error is not None:
            return float(preview.fit_error)
        value = self._policy_metrics(command).get("fit_error")
        return float(value) if isinstance(value, (int, float)) else None

    def _refinement_feedback_reason(self, preview: CommandPreviewResult, command: dict[str, Any]) -> str | None:
        if preview.refinement_feedback_reason is not None:
            return str(preview.refinement_feedback_reason)
        value = self._policy_metrics(command).get("refinement_feedback_reason")
        return str(value) if isinstance(value, str) else None

    def _metrics_delta(self, preview: CommandPreviewResult, command: dict[str, Any]) -> dict[str, Any]:
        metrics = self._policy_metrics(command)
        return {
            "score_delta": preview.score_delta,
            "affected_segments": len(preview.affected_segments),
            "constraint_violation_delta": int(metrics.get("constraint_violation_delta", 0)),
            "fit_error_delta": float(metrics.get("fit_error_delta", 0.0)),
            "complexity_delta": float(metrics.get("complexity_delta", 0.0)),
            "edge_error_delta": float(metrics.get("edge_error_delta", 0.0)),
            "algorithm_fitting_confidence": self._algorithm_fitting_confidence(preview),
            "ai_command_confidence": float(command.get("confidence", 0.0)),
            "inlier_ratio": self._inlier_ratio(preview, command),
            "fit_error": self._fit_error(preview, command),
            "refinement_feedback_reason": self._refinement_feedback_reason(preview, command),
            "self_intersection_delta": sum(
                max(preview.self_intersection_count_after.get(path_id, 0) - preview.self_intersection_count_before.get(path_id, 0), 0)
                for path_id in preview.self_intersection_count_after
            ),
        }

    def _preview_summary(self, preview: CommandPreviewResult, command: dict[str, Any]) -> dict[str, Any]:
        return {
            "command_id": preview.command_id,
            "score_before": preview.old_score,
            "score_after": preview.predicted_new_score,
            "score_delta": preview.score_delta,
            "affected_paths": list(preview.affected_paths),
            "affected_segments": list(preview.affected_segments),
            "topology_status_before": dict(preview.topology_status_before),
            "topology_status_after": dict(preview.topology_status_after),
            "self_intersection_count_before": dict(preview.self_intersection_count_before),
            "self_intersection_count_after": dict(preview.self_intersection_count_after),
            "constraint_violation_delta": int(self._policy_metrics(command).get("constraint_violation_delta", 0)),
            "fit_error_delta": float(self._policy_metrics(command).get("fit_error_delta", 0.0)),
            "complexity_delta": float(self._policy_metrics(command).get("complexity_delta", 0.0)),
            "edge_error_delta": float(self._policy_metrics(command).get("edge_error_delta", 0.0)),
            "algorithm_fitting_confidence": self._algorithm_fitting_confidence(preview),
            "ai_command_confidence": float(command.get("confidence", 0.0)),
            "inlier_ratio": self._inlier_ratio(preview, command),
            "fit_error": self._fit_error(preview, command),
            "refinement_feedback_reason": self._refinement_feedback_reason(preview, command),
        }

    def _edge_error_gain(self, policy_metrics: dict[str, Any]) -> float:
        edge_error_delta = float(policy_metrics.get("edge_error_delta", 0.0))
        return max(-edge_error_delta, 0.0)

    def _coordinate_system_consistent(
        self,
        before_document: VectorDocument,
        after_document: VectorDocument,
    ) -> bool:
        return before_document.coordinate_system == after_document.coordinate_system

    def _external_decision_request(
        self,
        *,
        command: dict[str, Any],
        reason: str,
        risk_flags: tuple[str, ...],
        risk_level: RiskLevel,
        preview_summary: dict[str, Any],
        policy_feedback: PolicyFeedback,
    ) -> ExternalDecisionRequest:
        command_id = str(command.get("command_id", "external_decision"))
        return ExternalDecisionRequest(
            decision_id=f"{command_id}:external_decision",
            reason=reason,
            risk_flags=tuple(dict.fromkeys(risk_flags + (risk_level.value,))),
            available_actions=("apply", "reject", "defer"),
            command=dict(command),
            candidate_id=command.get("candidate_id"),
            preview_summary=preview_summary,
            policy_feedback=policy_feedback,
        )


def _decision_kind_from_legacy(value: DecisionType) -> DecisionKind:
    return DecisionKind.from_legacy(value)


def _forbidden_repeated_commands(command: dict[str, Any], risk_flags: list[str]) -> tuple[str, ...]:
    if "risk_level_requires_escalation" not in risk_flags and "manual_only_mode" not in risk_flags:
        return ()
    tool = str(command.get("tool", "unknown_tool"))
    target = str(command.get("path_id") or command.get("segment_id") or command.get("candidate_id") or "unknown_target")
    return (f"{tool}:{target}",)


__all__ = [
    "DecisionType",
    "PreviewAndAutoAcceptPolicy",
    "PreviewAndAutoAcceptPolicyConfig",
    "PreviewDecision",
    "PreviewPolicyResult",
]
