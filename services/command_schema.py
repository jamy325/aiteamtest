from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.document import from_dict
from core.types import Constraint, Segment, VectorDocument


SEGMENT_REPLACE_TOOLS = {
    "propose_replace_segment_with_line",
    "propose_replace_segment_with_arc",
    "propose_replace_segment_with_circle",
    "propose_replace_segment_with_ellipse",
}
PATH_REPLACE_TOOLS = {
    "propose_replace_path_with_circle",
    "propose_replace_path_with_ellipse",
}
BATCH_TOOL = "propose_batch_refinement"
REPLACE_TOOLS = SEGMENT_REPLACE_TOOLS | PATH_REPLACE_TOOLS
ALLOWED_TOOLS = REPLACE_TOOLS | {BATCH_TOOL}
FORBIDDEN_PRECISE_KEYS = {
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "rotation",
    "start",
    "end",
    "start_angle",
    "end_angle",
    "control1",
    "control2",
    "control_points",
    "in_handle",
    "out_handle",
    "shared_tangent",
}


@dataclass(frozen=True, slots=True)
class CommandValidationResult:
    tool: str
    target_path_id: str | None = None
    target_segment_ids: tuple[str, ...] = ()


class CommandValidationError(ValueError):
    pass


def validate_command(command: dict[str, Any], document: VectorDocument | dict[str, Any]) -> CommandValidationResult:
    if not isinstance(command, dict):
        raise CommandValidationError("command must be a dictionary")
    vector_document = _coerce_document(document)
    _validate_coordinate_system(vector_document)
    _reject_precise_geometry(command)

    tool = command.get("tool")
    if tool not in ALLOWED_TOOLS:
        raise CommandValidationError(f"unknown tool: {tool}")

    if tool == BATCH_TOOL:
        return _validate_batch_command(command, vector_document)
    return _validate_replace_command(command, vector_document)


def validate_commands(commands: list[dict[str, Any]] | tuple[dict[str, Any], ...], document: VectorDocument | dict[str, Any]) -> tuple[CommandValidationResult, ...]:
    return tuple(validate_command(command, document) for command in commands)


def _validate_batch_command(command: dict[str, Any], document: VectorDocument) -> CommandValidationResult:
    _require_fields(command, ("summary", "commands", "confidence", "requires_user_confirmation"))
    nested_commands = command["commands"]
    if not isinstance(nested_commands, list) or not nested_commands:
        raise CommandValidationError("batch command requires a non-empty commands list")
    _validate_common_fields(command)

    nested_results = tuple(_validate_replace_command(item, document) for item in nested_commands)
    path_ids = tuple(result.target_path_id for result in nested_results if result.target_path_id is not None)
    return CommandValidationResult(
        tool=BATCH_TOOL,
        target_path_id=path_ids[0] if path_ids else None,
        target_segment_ids=tuple(segment_id for result in nested_results for segment_id in result.target_segment_ids),
    )


def _validate_replace_command(command: dict[str, Any], document: VectorDocument) -> CommandValidationResult:
    tool = command.get("tool")
    if tool in SEGMENT_REPLACE_TOOLS:
        return _validate_replace_segment_command(command, document)
    if tool in PATH_REPLACE_TOOLS:
        return _validate_replace_path_command(command, document)
    raise CommandValidationError(f"unknown tool: {tool}")


def _validate_replace_segment_command(command: dict[str, Any], document: VectorDocument) -> CommandValidationResult:
    _require_fields(command, ("path_id", "segment_range", "reason", "confidence", "requires_user_confirmation"))
    _validate_common_fields(command)

    path = _path_by_id(document, str(command["path_id"]))
    if path.locked:
        raise CommandValidationError(f"locked path cannot be modified: {path.path_id}")

    segment_range = command["segment_range"]
    if not isinstance(segment_range, list) or len(segment_range) != 2:
        raise CommandValidationError("segment_range must contain exactly two indices")
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in segment_range):
        raise CommandValidationError("segment_range indices must be integers")
    start_index = segment_range[0]
    end_index = segment_range[1]
    if start_index < 0 or end_index < start_index:
        raise CommandValidationError("segment_range is invalid")
    if end_index >= len(path.segments):
        raise CommandValidationError(f"segment_range exceeds path segment count for {path.path_id}")

    target_segment_ids = path.segments[start_index : end_index + 1]
    target_segments = tuple(_segment_by_id(document, segment_id) for segment_id in target_segment_ids)
    for segment in target_segments:
        if segment.locked:
            raise CommandValidationError(f"locked segment cannot be modified: {segment.segment_id}")

    target_anchor_ids = {anchor_id for segment in target_segments for anchor_id in segment.anchors}
    for anchor in document.anchors:
        if anchor.anchor_id in target_anchor_ids and anchor.locked:
            raise CommandValidationError(f"locked anchor cannot be modified: {anchor.anchor_id}")

    for constraint in document.constraints:
        if constraint.locked and _constraint_targets_locked(constraint, path.path_id, target_segment_ids, target_anchor_ids):
            raise CommandValidationError(f"locked constraint blocks modification: {constraint.constraint_id}")

    locked_anchor_ids = command.get("locked_anchor_ids", [])
    if not isinstance(locked_anchor_ids, list):
        raise CommandValidationError("locked_anchor_ids must be a list when provided")
    for anchor_id in locked_anchor_ids:
        if _anchor_exists(document, str(anchor_id)) is None:
            raise CommandValidationError(f"unknown anchor_id in locked_anchor_ids: {anchor_id}")

    return CommandValidationResult(
        tool=str(command["tool"]),
        target_path_id=path.path_id,
        target_segment_ids=tuple(target_segment_ids),
    )


def _validate_replace_path_command(command: dict[str, Any], document: VectorDocument) -> CommandValidationResult:
    _require_fields(command, ("path_id", "reason", "confidence", "requires_user_confirmation"))
    _validate_common_fields(command)
    if "segment_range" in command:
        raise CommandValidationError("path replacement does not accept segment_range")
    if "segment_id" in command:
        raise CommandValidationError("path replacement does not accept segment_id")

    path = _path_by_id(document, str(command["path_id"]))
    if path.locked:
        raise CommandValidationError(f"locked path cannot be modified: {path.path_id}")
    if not path.closed:
        raise CommandValidationError(f"path must be closed for path replacement: {path.path_id}")
    if not path.segments:
        raise CommandValidationError(f"path has no segments for path replacement: {path.path_id}")

    target_segment_ids = tuple(path.segments)
    target_segments = tuple(_segment_by_id(document, segment_id) for segment_id in target_segment_ids)
    for segment in target_segments:
        if segment.locked:
            raise CommandValidationError(f"locked segment cannot be modified: {segment.segment_id}")

    target_anchor_ids = {anchor_id for segment in target_segments for anchor_id in segment.anchors}
    for anchor in document.anchors:
        if anchor.anchor_id in target_anchor_ids and anchor.locked:
            raise CommandValidationError(f"locked anchor cannot be modified: {anchor.anchor_id}")

    for constraint in document.constraints:
        if constraint.locked and _constraint_targets_locked(constraint, path.path_id, target_segment_ids, target_anchor_ids):
            raise CommandValidationError(f"locked constraint blocks modification: {constraint.constraint_id}")

    return CommandValidationResult(
        tool=str(command["tool"]),
        target_path_id=path.path_id,
        target_segment_ids=target_segment_ids,
    )


def _validate_common_fields(command: dict[str, Any]) -> None:
    reason = command.get("reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise CommandValidationError("reason must be a non-empty string")

    confidence = command["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        raise CommandValidationError("confidence must be within [0, 1]")
    if not isinstance(command["requires_user_confirmation"], bool):
        raise CommandValidationError("requires_user_confirmation must be boolean")
    candidate_id = command.get("candidate_id")
    if candidate_id is not None and (not isinstance(candidate_id, str) or not candidate_id.strip()):
        raise CommandValidationError("candidate_id must be a non-empty string when provided")
    semantic_source = command.get("semantic_source")
    if semantic_source is not None and (not isinstance(semantic_source, str) or not semantic_source.strip()):
        raise CommandValidationError("semantic_source must be a non-empty string when provided")
    semantic_confidence = command.get("semantic_confidence")
    if semantic_confidence is not None:
        if (
            isinstance(semantic_confidence, bool)
            or not isinstance(semantic_confidence, (int, float))
            or not (0.0 <= float(semantic_confidence) <= 1.0)
        ):
            raise CommandValidationError("semantic_confidence must be within [0, 1]")


def _validate_coordinate_system(document: VectorDocument) -> None:
    if document.coordinate_system.internal_space != "vector":
        raise CommandValidationError("document coordinate system must use Vector Space")


def _reject_precise_geometry(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_PRECISE_KEYS:
                raise CommandValidationError(f"precise geometry parameter is forbidden: {key}")
            _reject_precise_geometry(nested)
        return
    if isinstance(value, list):
        for item in value:
            _reject_precise_geometry(item)


def _constraint_targets_locked(
    constraint: Constraint,
    path_id: str,
    segment_ids: tuple[str, ...],
    anchor_ids: set[str],
) -> bool:
    target_ids = set(constraint.targets)
    return bool(target_ids & ({path_id} | set(segment_ids) | anchor_ids))


def _require_fields(command: dict[str, Any], field_names: tuple[str, ...]) -> None:
    missing = [field for field in field_names if field not in command]
    if missing:
        raise CommandValidationError(f"missing required fields: {', '.join(missing)}")


def _coerce_document(document: VectorDocument | dict[str, Any]) -> VectorDocument:
    if isinstance(document, VectorDocument):
        return document
    return from_dict(document)


def _path_by_id(document: VectorDocument, path_id: str):
    for path in document.paths:
        if path.path_id == path_id:
            return path
    raise CommandValidationError(f"unknown path_id: {path_id}")


def _segment_by_id(document: VectorDocument, segment_id: str) -> Segment:
    for segment in document.segments:
        if segment.segment_id == segment_id:
            return segment
    raise CommandValidationError(f"unknown segment_id: {segment_id}")


def _anchor_exists(document: VectorDocument, anchor_id: str):
    for anchor in document.anchors:
        if anchor.anchor_id == anchor_id:
            return anchor
    return None


__all__ = [
    "ALLOWED_TOOLS",
    "BATCH_TOOL",
    "CommandValidationError",
    "CommandValidationResult",
    "FORBIDDEN_PRECISE_KEYS",
    "PATH_REPLACE_TOOLS",
    "REPLACE_TOOLS",
    "SEGMENT_REPLACE_TOOLS",
    "validate_command",
    "validate_commands",
]
