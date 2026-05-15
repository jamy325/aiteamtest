from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.types import VectorDocument


@dataclass(frozen=True, slots=True)
class AISuggestionOverlayTarget:
    target_type: str
    target_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    exists: bool = True
    locked: bool = False


@dataclass(frozen=True, slots=True)
class AISuggestionOverlay:
    overlay_id: str
    source_type: str
    title: str
    detail: str
    tool: str | None = None
    confidence: float | None = None
    path_id: str | None = None
    segment_ids: tuple[str, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    targets: tuple[AISuggestionOverlayTarget, ...] = ()
    locked_target_ids: tuple[str, ...] = ()
    unknown_target_ids: tuple[str, ...] = ()


class CanvasWidget:
    def __init__(self, *, locked_ids: tuple[str, ...] = ()) -> None:
        self._locked_ids = tuple(str(item) for item in locked_ids)
        self._image_path: str | None = None
        self._document: VectorDocument | None = None
        self._review_summary: str = ""
        self._review_issues: tuple[dict[str, Any], ...] = ()
        self._review_commands: tuple[dict[str, Any], ...] = ()
        self._suggestion_overlays: tuple[AISuggestionOverlay, ...] = ()

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

    @property
    def suggestion_overlays(self) -> tuple[AISuggestionOverlay, ...]:
        return self._suggestion_overlays

    def set_locked_ids(self, locked_ids: tuple[str, ...] | list[str]) -> None:
        self._locked_ids = tuple(str(item) for item in locked_ids)
        self._suggestion_overlays = self._build_suggestion_overlays()

    def set_image_path(self, image_path: str | None) -> None:
        self._image_path = None if image_path is None else str(image_path)

    def set_document(self, document: VectorDocument | None) -> None:
        self._document = document
        self._suggestion_overlays = self._build_suggestion_overlays()

    def set_review_display(
        self,
        *,
        summary: str,
        issues: tuple[dict[str, Any], ...],
        proposed_commands: tuple[dict[str, Any], ...],
    ) -> tuple[AISuggestionOverlay, ...]:
        self._review_summary = str(summary)
        self._review_issues = tuple(dict(item) for item in issues)
        self._review_commands = tuple(dict(item) for item in proposed_commands)
        self._suggestion_overlays = self._build_suggestion_overlays()
        return self._suggestion_overlays

    def lock_id(self, item_id: str) -> None:
        if item_id in self._locked_ids:
            return
        self._locked_ids = self._locked_ids + (str(item_id),)
        self._suggestion_overlays = self._build_suggestion_overlays()

    def unlock_id(self, item_id: str) -> None:
        self._locked_ids = tuple(existing for existing in self._locked_ids if existing != item_id)
        self._suggestion_overlays = self._build_suggestion_overlays()

    def _build_suggestion_overlays(self) -> tuple[AISuggestionOverlay, ...]:
        overlays: list[AISuggestionOverlay] = []
        for index, issue in enumerate(self._review_issues):
            overlays.append(self._overlay_from_item(source_type="issue", item=issue, index=index))
        for index, command in enumerate(self._review_commands):
            overlays.extend(self._overlays_from_command_item(command, index))
        return tuple(overlays)

    def _overlays_from_command_item(self, item: dict[str, Any], index: int) -> tuple[AISuggestionOverlay, ...]:
        if item.get("tool") != "propose_batch_refinement":
            return (self._overlay_from_item(source_type="command", item=item, index=index),)

        nested_commands = item.get("commands")
        if not isinstance(nested_commands, list):
            return (self._overlay_from_item(source_type="command", item=item, index=index),)

        overlays: list[AISuggestionOverlay] = []
        batch_summary = str(item.get("summary", ""))
        batch_confidence = self._coerce_optional_float(item.get("confidence"))

        for nested_index, nested_command in enumerate(nested_commands):
            if not isinstance(nested_command, dict):
                continue
            nested_overlay = self._overlay_from_item(
                source_type="command",
                item={
                    **nested_command,
                    "confidence": nested_command.get("confidence", batch_confidence),
                    "reason": nested_command.get("reason", batch_summary),
                    "_overlay_id_suffix": f"{index}:{nested_index}",
                },
                index=nested_index,
            )
            overlays.append(nested_overlay)

        if overlays:
            return tuple(overlays)
        return (self._overlay_from_item(source_type="command", item=item, index=index),)

    def _overlay_from_item(self, *, source_type: str, item: dict[str, Any], index: int) -> AISuggestionOverlay:
        path_id = self._coerce_optional_string(item.get("path_id"))
        segment_ids = self._resolve_segment_ids(item, path_id)
        bbox = self._coerce_bbox(item.get("bbox"))
        targets: list[AISuggestionOverlayTarget] = []
        unknown_target_ids: list[str] = []
        locked_target_ids: list[str] = []

        if bbox is not None:
            targets.append(AISuggestionOverlayTarget(target_type="bbox", bbox=bbox, locked=False))

        if path_id is not None:
            path_exists = self._path_exists(path_id)
            path_locked = path_id in self._locked_ids
            targets.append(
                AISuggestionOverlayTarget(
                    target_type="path",
                    target_id=path_id,
                    exists=path_exists,
                    locked=path_locked,
                )
            )
            if not path_exists:
                unknown_target_ids.append(path_id)
            if path_locked:
                locked_target_ids.append(path_id)

        for segment_id in segment_ids:
            segment_exists = self._segment_exists(segment_id)
            segment_locked = segment_id in self._locked_ids
            targets.append(
                AISuggestionOverlayTarget(
                    target_type="segment",
                    target_id=segment_id,
                    exists=segment_exists,
                    locked=segment_locked,
                )
            )
            if not segment_exists:
                unknown_target_ids.append(segment_id)
            if segment_locked:
                locked_target_ids.append(segment_id)

        explicit_segment_id = self._coerce_optional_string(item.get("segment_id"))
        if explicit_segment_id is not None and explicit_segment_id not in segment_ids:
            segment_exists = self._segment_exists(explicit_segment_id)
            segment_locked = explicit_segment_id in self._locked_ids
            targets.append(
                AISuggestionOverlayTarget(
                    target_type="segment",
                    target_id=explicit_segment_id,
                    exists=segment_exists,
                    locked=segment_locked,
                )
            )
            if not segment_exists:
                unknown_target_ids.append(explicit_segment_id)
            if segment_locked:
                locked_target_ids.append(explicit_segment_id)

        for anchor_id in self._coerce_string_list(item.get("locked_anchor_ids")):
            if anchor_id in self._locked_ids:
                locked_target_ids.append(anchor_id)

        title = self._overlay_title(source_type, item, index)
        detail = self._overlay_detail(source_type, item)
        overlay_id = self._overlay_id(source_type, item, index)
        return AISuggestionOverlay(
            overlay_id=overlay_id,
            source_type=source_type,
            title=title,
            detail=detail,
            tool=self._coerce_optional_string(item.get("tool")),
            confidence=self._coerce_optional_float(item.get("confidence")),
            path_id=path_id,
            segment_ids=tuple(segment_ids),
            bbox=bbox,
            targets=tuple(targets),
            locked_target_ids=tuple(dict.fromkeys(locked_target_ids)),
            unknown_target_ids=tuple(dict.fromkeys(unknown_target_ids)),
        )

    def _resolve_segment_ids(self, item: dict[str, Any], path_id: str | None) -> tuple[str, ...]:
        explicit_segment_id = self._coerce_optional_string(item.get("segment_id"))
        if explicit_segment_id is not None:
            return (explicit_segment_id,)

        segment_range = item.get("segment_range")
        if not isinstance(segment_range, list) or len(segment_range) != 2:
            return ()
        if not all(isinstance(index, int) and not isinstance(index, bool) for index in segment_range):
            return ()
        if path_id is None or self._document is None:
            return ()
        path = next((value for value in self._document.paths if value.path_id == path_id), None)
        if path is None:
            return ()
        start_index, end_index = segment_range
        if start_index < 0 or end_index < start_index or end_index >= len(path.segments):
            return ()
        return tuple(path.segments[start_index : end_index + 1])

    def _path_exists(self, path_id: str) -> bool:
        if self._document is None:
            return False
        return any(path.path_id == path_id for path in self._document.paths)

    def _segment_exists(self, segment_id: str) -> bool:
        if self._document is None:
            return False
        return any(segment.segment_id == segment_id for segment in self._document.segments)

    @staticmethod
    def _coerce_optional_string(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _coerce_string_list(value: Any) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(str(item) for item in value)

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _overlay_id(source_type: str, item: dict[str, Any], index: int) -> str:
        suffix = item.get("_overlay_id_suffix")
        if suffix is not None:
            return f"{source_type}:{item.get('tool', index)}:{suffix}"
        key_name = "issue_id" if source_type == "issue" else "tool"
        key_value = item.get(key_name, index)
        return f"{source_type}:{key_value}"

    @staticmethod
    def _overlay_title(source_type: str, item: dict[str, Any], index: int) -> str:
        if source_type == "issue":
            category = str(item.get("category", "issue"))
            severity = str(item.get("severity", "unknown"))
            return f"{category}:{severity}:{index}"
        tool = item.get("tool", "unknown_tool")
        return str(tool)

    @staticmethod
    def _overlay_detail(source_type: str, item: dict[str, Any]) -> str:
        if source_type == "issue":
            return str(item.get("summary", ""))
        if item.get("tool") == "propose_batch_refinement":
            return str(item.get("summary", ""))
        return str(item.get("reason", ""))
