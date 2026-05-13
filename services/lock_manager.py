from __future__ import annotations

from dataclasses import dataclass

from core.document import set_segment_locked
from core.types import VectorDocument


@dataclass(frozen=True, slots=True)
class LockManager:
    def lock_segment(self, document: VectorDocument, segment_id: str) -> VectorDocument:
        return set_segment_locked(document, segment_id, locked=True)


def lock_segment(document: VectorDocument, segment_id: str) -> VectorDocument:
    return set_segment_locked(document, segment_id, locked=True)


__all__ = ["LockManager", "lock_segment"]
