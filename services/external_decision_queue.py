from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from core.types import VectorDocument
from services.engine_protocol import (
    ExternalDecisionAction,
    ExternalDecisionRecord,
    ExternalDecisionRequest,
    ExternalDecisionStatus,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExternalDecisionQueue:
    def __init__(
        self,
        records: Iterable[ExternalDecisionRecord] = (),
    ) -> None:
        self._records: dict[str, ExternalDecisionRecord] = {}
        for record in records:
            self._store(record)

    def enqueue(
        self,
        request: ExternalDecisionRequest,
        *,
        preview_document: VectorDocument | None = None,
        created_at: str | None = None,
    ) -> ExternalDecisionRecord:
        if request.decision_id in self._records:
            raise ValueError(f"duplicate decision_id: {request.decision_id}")
        timestamp = created_at or _utc_now_iso()
        record = ExternalDecisionRecord(
            request=request,
            status=ExternalDecisionStatus.PENDING,
            preview_document=preview_document,
            created_at=timestamp,
            updated_at=timestamp,
            action_reason=None,
        )
        self._store(record)
        return record

    def get(self, decision_id: str) -> ExternalDecisionRecord:
        try:
            return self._records[decision_id]
        except KeyError as exc:
            raise KeyError(f"unknown decision_id: {decision_id}") from exc

    def list(self) -> tuple[ExternalDecisionRecord, ...]:
        return tuple(self._records.values())

    def apply(
        self,
        decision_id: str,
        *,
        reason: str | None = None,
        timestamp: str | None = None,
    ) -> ExternalDecisionRecord:
        record = self.get(decision_id)
        if record.preview_document is None:
            raise ValueError("apply requires preview_document to preserve preview transaction semantics")
        return self._update_record(
            record,
            status=ExternalDecisionStatus.APPLIED,
            reason=reason,
            timestamp=timestamp,
        )

    def reject(
        self,
        decision_id: str,
        *,
        reason: str,
        timestamp: str | None = None,
    ) -> ExternalDecisionRecord:
        return self._update_record(
            self.get(decision_id),
            status=ExternalDecisionStatus.REJECTED,
            reason=reason,
            timestamp=timestamp,
        )

    def defer(
        self,
        decision_id: str,
        *,
        reason: str,
        timestamp: str | None = None,
    ) -> ExternalDecisionRecord:
        return self._update_record(
            self.get(decision_id),
            status=ExternalDecisionStatus.DEFERRED,
            reason=reason,
            timestamp=timestamp,
        )

    def resolve(
        self,
        decision_id: str,
        action: str | ExternalDecisionAction,
        *,
        reason: str | None = None,
        timestamp: str | None = None,
    ) -> ExternalDecisionRecord:
        try:
            normalized_action = action if isinstance(action, ExternalDecisionAction) else ExternalDecisionAction(str(action))
        except ValueError as exc:
            raise ValueError(f"unsupported external decision action: {action}") from exc

        if normalized_action is ExternalDecisionAction.APPLY:
            return self.apply(decision_id, reason=reason, timestamp=timestamp)
        if normalized_action is ExternalDecisionAction.REJECT:
            if not reason:
                raise ValueError("reject requires reason")
            return self.reject(decision_id, reason=reason, timestamp=timestamp)
        if normalized_action is ExternalDecisionAction.DEFER:
            if not reason:
                raise ValueError("defer requires reason")
            return self.defer(decision_id, reason=reason, timestamp=timestamp)
        raise ValueError(f"unsupported external decision action: {action}")

    def to_dict(self) -> dict[str, object]:
        return {
            "records": [record.to_dict() for record in self.list()],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ExternalDecisionQueue":
        raw_records = data.get("records", [])
        if not isinstance(raw_records, list):
            raise ValueError("records must be a list")
        return cls(ExternalDecisionRecord.from_dict(item) for item in raw_records)

    @classmethod
    def from_json(cls, payload: str) -> "ExternalDecisionQueue":
        return cls.from_dict(json.loads(payload))

    @classmethod
    def load_json(cls, path: str | Path) -> "ExternalDecisionQueue":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def _store(self, record: ExternalDecisionRecord) -> None:
        decision_id = record.request.decision_id
        if decision_id in self._records:
            raise ValueError(f"duplicate decision_id: {decision_id}")
        self._records[decision_id] = record

    def _update_record(
        self,
        record: ExternalDecisionRecord,
        *,
        status: ExternalDecisionStatus,
        reason: str | None,
        timestamp: str | None,
    ) -> ExternalDecisionRecord:
        if record.status is not ExternalDecisionStatus.PENDING:
            raise ValueError(f"decision already resolved: {record.request.decision_id}")
        updated = replace(
            record,
            status=status,
            updated_at=timestamp or _utc_now_iso(),
            action_reason=reason,
        )
        self._records[record.request.decision_id] = updated
        return updated


__all__ = ["ExternalDecisionQueue"]
