from __future__ import annotations

from typing import Any

from core.types import VectorDocument


class CanvasWidget:
    def __init__(self, *, locked_ids: tuple[str, ...] = ()) -> None:
        self._locked_ids = tuple(str(item) for item in locked_ids)
        self._image_path: str | None = None
        self._document: VectorDocument | None = None
        self._review_summary: str = ""
        self._review_issues: tuple[dict[str, Any], ...] = ()
        self._review_commands: tuple[dict[str, Any], ...] = ()

    @property
    def locked_ids(self) -> tuple[str, ...]:
        return self._locked_ids

    @property
    def image_path(self) -> str | None:
        return self._image_path

    @property
    def document(self) -> VectorDocument | None:
        return self._document

    @property
    def review_summary(self) -> str:
        return self._review_summary

    @property
    def review_issues(self) -> tuple[dict[str, Any], ...]:
        return self._review_issues

    @property
    def review_commands(self) -> tuple[dict[str, Any], ...]:
        return self._review_commands

    def set_locked_ids(self, locked_ids: tuple[str, ...] | list[str]) -> None:
        self._locked_ids = tuple(str(item) for item in locked_ids)

    def set_image_path(self, image_path: str | None) -> None:
        self._image_path = None if image_path is None else str(image_path)

    def set_document(self, document: VectorDocument | None) -> None:
        self._document = document

    def set_review_display(
        self,
        *,
        summary: str,
        issues: tuple[dict[str, Any], ...],
        proposed_commands: tuple[dict[str, Any], ...],
    ) -> None:
        self._review_summary = str(summary)
        self._review_issues = tuple(dict(item) for item in issues)
        self._review_commands = tuple(dict(item) for item in proposed_commands)

    def lock_id(self, item_id: str) -> None:
        if item_id in self._locked_ids:
            return
        self._locked_ids = self._locked_ids + (str(item_id),)

    def unlock_id(self, item_id: str) -> None:
        self._locked_ids = tuple(existing for existing in self._locked_ids if existing != item_id)
