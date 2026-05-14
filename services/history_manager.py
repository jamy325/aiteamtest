from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.document import from_dict, to_dict
from core.types import VectorDocument


@dataclass(frozen=True, slots=True)
class HistoryItem:
    version: int
    command: object
    before: dict[str, Any]
    after: dict[str, Any]
    old_score: float | None
    new_score: float | None
    timestamp: str


class HistoryManager:
    def __init__(self) -> None:
        self._items: list[HistoryItem] = []
        self._cursor: int = -1

    @property
    def items(self) -> tuple[HistoryItem, ...]:
        return tuple(self._items)

    @property
    def cursor(self) -> int:
        return self._cursor

    def record(
        self,
        *,
        command: object,
        before_document: VectorDocument,
        after_document: VectorDocument,
        old_score: float | None = None,
        new_score: float | None = None,
        timestamp: str | None = None,
    ) -> HistoryItem:
        if self._cursor < len(self._items) - 1:
            self._items = self._items[: self._cursor + 1]

        item = HistoryItem(
            version=len(self._items) + 1,
            command=deepcopy(command),
            before=to_dict(before_document),
            after=to_dict(after_document),
            old_score=old_score,
            new_score=new_score,
            timestamp=timestamp or self._timestamp(),
        )
        self._items.append(item)
        self._cursor = len(self._items) - 1
        return item

    def undo(self) -> VectorDocument:
        if self._cursor < 0:
            raise ValueError("no history available for undo")
        item = self._items[self._cursor]
        self._cursor -= 1
        return from_dict(deepcopy(item.before))

    def redo(self) -> VectorDocument:
        if self._cursor + 1 >= len(self._items):
            raise ValueError("no history available for redo")
        self._cursor += 1
        item = self._items[self._cursor]
        return from_dict(deepcopy(item.after))

    def get_by_command_id(self, command_id: str) -> tuple[HistoryItem, ...]:
        return tuple(item for item in self._items if self._command_id(item.command) == command_id)

    def _command_id(self, command: object) -> str | None:
        if isinstance(command, dict) and "command_id" in command:
            return str(command["command_id"])
        return None

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "HistoryItem",
    "HistoryManager",
]
