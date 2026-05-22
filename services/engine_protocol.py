from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from core.document import from_dict as document_from_dict
from core.document import to_dict as document_to_dict
from core.types import VectorDocument


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


def _tuple_of_strings(values: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(str(item) for item in (values or ()))


def _json_safe(value: Any) -> Any:
    if isinstance(value, _StringEnum):
        return value.value
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(_json_safe(dict(value or {})))


class AutonomyLevel(_StringEnum):
    MANUAL_ONLY = "manual_only"
    ASSISTED = "assisted"
    AUTONOMOUS_SAFE = "autonomous_safe"
    AUTONOMOUS_FULL = "autonomous_full"


class DecisionKind(_StringEnum):
    AUTO_APPLY = "auto_apply"
    AUTO_REJECT = "auto_reject"
    REQUIRES_EXTERNAL_DECISION = "requires_external_decision"

    @classmethod
    def from_legacy(cls, value: str) -> "DecisionKind":
        normalized = str(value).strip().lower()
        mapping = {
            "auto_accept": cls.AUTO_APPLY,
            "auto_apply": cls.AUTO_APPLY,
            "reject": cls.AUTO_REJECT,
            "auto_reject": cls.AUTO_REJECT,
            "user_confirm": cls.REQUIRES_EXTERNAL_DECISION,
            "requires_external_decision": cls.REQUIRES_EXTERNAL_DECISION,
        }
        if normalized not in mapping:
            raise ValueError(f"unsupported legacy decision: {value}")
        return mapping[normalized]

    def to_legacy(self) -> str:
        mapping = {
            DecisionKind.AUTO_APPLY: "auto_accept",
            DecisionKind.AUTO_REJECT: "reject",
            DecisionKind.REQUIRES_EXTERNAL_DECISION: "user_confirm",
        }
        return mapping[self]


class RiskLevel(_StringEnum):
    LOW = "low"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"

    @property
    def sort_key(self) -> int:
        order = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.MEDIUM_HIGH: 2,
            RiskLevel.HIGH: 3,
        }
        return order[self]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.sort_key < other.sort_key

    def __le__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.sort_key <= other.sort_key

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.sort_key > other.sort_key

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.sort_key >= other.sort_key


class EngineStatus(_StringEnum):
    COMPLETED = "completed"
    REQUIRES_EXTERNAL_DECISION = "requires_external_decision"
    FAILED = "failed"

    @classmethod
    def from_legacy(cls, value: str) -> "EngineStatus":
        normalized = str(value).strip().lower()
        mapping = {
            "completed": cls.COMPLETED,
            "failed": cls.FAILED,
            "requires_external_decision": cls.REQUIRES_EXTERNAL_DECISION,
            "needs_external_decision": cls.REQUIRES_EXTERNAL_DECISION,
        }
        if normalized not in mapping:
            raise ValueError(f"unsupported legacy engine status: {value}")
        return mapping[normalized]


@dataclass(frozen=True, slots=True)
class PolicyFeedback:
    reason_code: str
    message: str
    metrics_delta: dict[str, Any] = field(default_factory=dict)
    policy_hint: str | None = None
    retry_allowed: bool = False
    retry_constraints: dict[str, Any] = field(default_factory=dict)
    forbidden_repeated_commands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics_delta", _copy_mapping(self.metrics_delta))
        object.__setattr__(self, "retry_constraints", _copy_mapping(self.retry_constraints))
        object.__setattr__(
            self,
            "forbidden_repeated_commands",
            _tuple_of_strings(self.forbidden_repeated_commands),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "message": self.message,
            "metrics_delta": _json_safe(self.metrics_delta),
            "policy_hint": self.policy_hint,
            "retry_allowed": self.retry_allowed,
            "retry_constraints": _json_safe(self.retry_constraints),
            "forbidden_repeated_commands": list(self.forbidden_repeated_commands),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyFeedback":
        return cls(
            reason_code=str(data["reason_code"]),
            message=str(data.get("message", "")),
            metrics_delta=dict(data.get("metrics_delta", {})),
            policy_hint=data.get("policy_hint"),
            retry_allowed=bool(data.get("retry_allowed", False)),
            retry_constraints=dict(data.get("retry_constraints", {})),
            forbidden_repeated_commands=tuple(data.get("forbidden_repeated_commands", ())),
        )


@dataclass(frozen=True, slots=True)
class RejectionMemoryItem:
    target: str
    tool: str
    reason_code: str
    retry_count: int
    last_metrics_delta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "retry_count", int(self.retry_count))
        object.__setattr__(self, "last_metrics_delta", _copy_mapping(self.last_metrics_delta))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "tool": self.tool,
            "reason_code": self.reason_code,
            "retry_count": self.retry_count,
            "last_metrics_delta": _json_safe(self.last_metrics_delta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RejectionMemoryItem":
        return cls(
            target=str(data["target"]),
            tool=str(data["tool"]),
            reason_code=str(data["reason_code"]),
            retry_count=int(data.get("retry_count", 0)),
            last_metrics_delta=dict(data.get("last_metrics_delta", {})),
        )


@dataclass(frozen=True, slots=True)
class ExternalDecisionRequest:
    decision_id: str
    reason: str
    risk_flags: tuple[str, ...] = ()
    available_actions: tuple[str, ...] = ()
    command: dict[str, Any] = field(default_factory=dict)
    candidate_id: str | None = None
    preview_summary: dict[str, Any] = field(default_factory=dict)
    policy_feedback: PolicyFeedback | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk_flags", _tuple_of_strings(self.risk_flags))
        object.__setattr__(self, "available_actions", _tuple_of_strings(self.available_actions))
        object.__setattr__(self, "command", _copy_mapping(self.command))
        object.__setattr__(self, "preview_summary", _copy_mapping(self.preview_summary))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "reason": self.reason,
            "risk_flags": list(self.risk_flags),
            "available_actions": list(self.available_actions),
            "command": _json_safe(self.command),
            "candidate_id": self.candidate_id,
            "preview_summary": _json_safe(self.preview_summary),
            "policy_feedback": None if self.policy_feedback is None else self.policy_feedback.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExternalDecisionRequest":
        policy_feedback_raw = data.get("policy_feedback")
        return cls(
            decision_id=str(data["decision_id"]),
            reason=str(data["reason"]),
            risk_flags=tuple(data.get("risk_flags", ())),
            available_actions=tuple(data.get("available_actions", ())),
            command=dict(data.get("command", {})),
            candidate_id=data.get("candidate_id"),
            preview_summary=dict(data.get("preview_summary", {})),
            policy_feedback=None
            if policy_feedback_raw is None
            else PolicyFeedback.from_dict(policy_feedback_raw),
        )


@dataclass(frozen=True, slots=True)
class DecisionPolicyResult:
    decision: DecisionKind
    reason_code: str
    risk_level: RiskLevel
    policy_feedback: PolicyFeedback | None = None
    external_decision_request: ExternalDecisionRequest | None = None

    def __post_init__(self) -> None:
        if (
            self.decision == DecisionKind.REQUIRES_EXTERNAL_DECISION
            and self.external_decision_request is None
        ):
            raise ValueError("external_decision_request is required for requires_external_decision")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason_code": self.reason_code,
            "risk_level": self.risk_level.value,
            "policy_feedback": None if self.policy_feedback is None else self.policy_feedback.to_dict(),
            "external_decision_request": None
            if self.external_decision_request is None
            else self.external_decision_request.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionPolicyResult":
        policy_feedback_raw = data.get("policy_feedback")
        external_request_raw = data.get("external_decision_request")
        return cls(
            decision=DecisionKind(str(data["decision"])),
            reason_code=str(data["reason_code"]),
            risk_level=RiskLevel(str(data["risk_level"])),
            policy_feedback=None
            if policy_feedback_raw is None
            else PolicyFeedback.from_dict(policy_feedback_raw),
            external_decision_request=None
            if external_request_raw is None
            else ExternalDecisionRequest.from_dict(external_request_raw),
        )


@dataclass(frozen=True, slots=True)
class EngineResult:
    status: EngineStatus
    document: VectorDocument | None = None
    report: dict[str, Any] = field(default_factory=dict)
    decisions: tuple[DecisionPolicyResult, ...] = ()
    external_decisions: tuple[ExternalDecisionRequest, ...] = ()
    policy_feedback: tuple[PolicyFeedback, ...] = ()
    rejection_memory: tuple[RejectionMemoryItem, ...] = ()
    errors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "external_decisions", tuple(self.external_decisions))
        object.__setattr__(self, "policy_feedback", tuple(self.policy_feedback))
        object.__setattr__(self, "rejection_memory", tuple(self.rejection_memory))
        object.__setattr__(self, "errors", _tuple_of_strings(self.errors))
        object.__setattr__(self, "report", _copy_mapping(self.report))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "document": None if self.document is None else document_to_dict(self.document),
            "report": _json_safe(self.report),
            "decisions": [item.to_dict() for item in self.decisions],
            "external_decisions": [item.to_dict() for item in self.external_decisions],
            "policy_feedback": [item.to_dict() for item in self.policy_feedback],
            "rejection_memory": [item.to_dict() for item in self.rejection_memory],
            "errors": list(self.errors),
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EngineResult":
        document_raw = data.get("document")
        return cls(
            status=EngineStatus.from_legacy(str(data["status"])),
            document=None if document_raw is None else document_from_dict(document_raw),
            report=dict(data.get("report", {})),
            decisions=tuple(
                DecisionPolicyResult.from_dict(item) for item in data.get("decisions", ())
            ),
            external_decisions=tuple(
                ExternalDecisionRequest.from_dict(item)
                for item in data.get("external_decisions", ())
            ),
            policy_feedback=tuple(
                PolicyFeedback.from_dict(item) for item in data.get("policy_feedback", ())
            ),
            rejection_memory=tuple(
                RejectionMemoryItem.from_dict(item)
                for item in data.get("rejection_memory", ())
            ),
            errors=tuple(str(item) for item in data.get("errors", ())),
            metadata=dict(data.get("metadata", {})),
        )


__all__ = [
    "AutonomyLevel",
    "DecisionKind",
    "DecisionPolicyResult",
    "EngineResult",
    "EngineStatus",
    "ExternalDecisionRequest",
    "PolicyFeedback",
    "RejectionMemoryItem",
    "RiskLevel",
]
