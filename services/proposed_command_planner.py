from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.types import Path, ShapeCandidate, VectorDocument
from services.command_schema import CommandValidationError, validate_command


PATH_LEVEL_TOOL_BY_TARGET = {
    "circle": "propose_replace_path_with_circle",
    "ellipse": "propose_replace_path_with_ellipse",
}
SEGMENT_LEVEL_TOOL_BY_TARGET = {
    "circle": "propose_replace_segment_with_circle",
    "ellipse": "propose_replace_segment_with_ellipse",
    "line": "propose_replace_segment_with_line",
    "arc": "propose_replace_segment_with_arc",
}


@dataclass(frozen=True, slots=True)
class ProposedCommandPlannerConfig:
    min_candidate_confidence: float = 0.35
    batch_min_command_count: int = 2
    max_batch_commands: int = 6
    confidence_weight: float = 0.8
    complexity_weight: float = 0.2
    path_replacement_bonus: float = 0.05
    semantic_source_prefix: str = "shape_candidate_detector"
    include_batch_command: bool = True


@dataclass(frozen=True, slots=True)
class _PlannedCommand:
    command: dict[str, Any]
    score: float
    dedupe_key: tuple[Any, ...]
    confidence: float


class ProposedCommandPlanner:
    def __init__(self, config: ProposedCommandPlannerConfig | None = None) -> None:
        self.config = config or ProposedCommandPlannerConfig()

    def plan_commands(
        self,
        document: VectorDocument,
        candidates: tuple[ShapeCandidate, ...] | list[ShapeCandidate],
    ) -> tuple[dict[str, Any], ...]:
        planned_commands: list[_PlannedCommand] = []
        path_by_id = {path.path_id: path for path in document.paths}

        for candidate in candidates:
            if candidate.confidence < self.config.min_candidate_confidence:
                continue
            path = path_by_id.get(candidate.path_id)
            if path is None:
                continue
            for planned in self._plan_candidate(candidate, path):
                try:
                    validate_command(planned.command, document)
                except CommandValidationError:
                    continue
                planned_commands.append(planned)

        selected = self._dedupe_and_sort(planned_commands)
        commands = [dict(planned.command) for planned in selected]

        if self.config.include_batch_command:
            batch_command = self._batch_command(commands, document)
            if batch_command is not None:
                commands.append(batch_command)

        return tuple(commands)

    def _plan_candidate(self, candidate: ShapeCandidate, path: Path) -> tuple[_PlannedCommand, ...]:
        if candidate.target_type == "rectangle":
            return self._plan_rectangle_candidate(candidate, path)

        tool = self._tool_for_candidate(candidate, path)
        if tool is None:
            return ()
        command = self._base_command(candidate, tool)
        score = self._candidate_score(candidate, tool)
        return (
            _PlannedCommand(
                command=command,
                score=score,
                dedupe_key=self._dedupe_key(command),
                confidence=candidate.confidence,
            ),
        )

    def _plan_rectangle_candidate(self, candidate: ShapeCandidate, path: Path) -> tuple[_PlannedCommand, ...]:
        start_index, end_index = candidate.segment_range
        if not path.segments:
            return ()

        commands: list[_PlannedCommand] = []
        score = self._candidate_score(candidate, "propose_replace_segment_with_line")
        for segment_index in range(start_index, min(end_index, len(path.segments) - 1) + 1):
            command = self._base_command(
                candidate,
                "propose_replace_segment_with_line",
                segment_range=(segment_index, segment_index),
                reason=(
                    candidate.reason
                    or "Rectangle candidate suggests this edge should be refined as a straight line."
                ),
            )
            commands.append(
                _PlannedCommand(
                    command=command,
                    score=score,
                    dedupe_key=self._dedupe_key(command),
                    confidence=candidate.confidence,
                )
            )
        return tuple(commands)

    def _tool_for_candidate(self, candidate: ShapeCandidate, path: Path) -> str | None:
        if candidate.target_type in PATH_LEVEL_TOOL_BY_TARGET and self._covers_full_path(candidate, path):
            return PATH_LEVEL_TOOL_BY_TARGET[candidate.target_type]
        return SEGMENT_LEVEL_TOOL_BY_TARGET.get(candidate.target_type)

    def _base_command(
        self,
        candidate: ShapeCandidate,
        tool: str,
        *,
        segment_range: tuple[int, int] | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        command = {
            "tool": tool,
            "path_id": candidate.path_id,
            "reason": reason or self._default_reason(candidate),
            "confidence": candidate.confidence,
            "requires_user_confirmation": True,
            "candidate_id": candidate.candidate_id,
            "semantic_source": self._semantic_source(candidate),
            "semantic_confidence": candidate.confidence,
        }
        if tool in SEGMENT_LEVEL_TOOL_BY_TARGET.values():
            command["segment_range"] = list(segment_range or candidate.segment_range)
        return command

    def _candidate_score(self, candidate: ShapeCandidate, tool: str) -> float:
        complexity_delta = float(candidate.evidence.get("model_complexity_delta", 0.0) or 0.0)
        complexity_score = max(0.0, min(1.0, complexity_delta / 4.0))
        score = (
            (candidate.confidence * self.config.confidence_weight)
            + (complexity_score * self.config.complexity_weight)
        )
        if tool in PATH_LEVEL_TOOL_BY_TARGET.values():
            score += self.config.path_replacement_bonus
        return min(1.0, score)

    def _dedupe_and_sort(self, planned_commands: list[_PlannedCommand]) -> tuple[_PlannedCommand, ...]:
        best_by_key: dict[tuple[Any, ...], _PlannedCommand] = {}
        for planned in planned_commands:
            existing = best_by_key.get(planned.dedupe_key)
            if existing is None or self._is_better(planned, existing):
                best_by_key[planned.dedupe_key] = planned
        return tuple(
            sorted(
                best_by_key.values(),
                key=lambda item: (
                    -item.score,
                    -item.confidence,
                    item.command["path_id"],
                    item.command["tool"],
                    tuple(item.command.get("segment_range", ())),
                ),
            )
        )

    def _is_better(self, candidate: _PlannedCommand, current: _PlannedCommand) -> bool:
        if candidate.score != current.score:
            return candidate.score > current.score
        if candidate.confidence != current.confidence:
            return candidate.confidence > current.confidence
        return str(candidate.command["tool"]) < str(current.command["tool"])

    def _dedupe_key(self, command: dict[str, Any]) -> tuple[Any, ...]:
        segment_range = command.get("segment_range")
        if segment_range is None:
            return ("path", command["path_id"])
        return ("segment", command["path_id"], int(segment_range[0]), int(segment_range[1]))

    def _batch_command(self, commands: list[dict[str, Any]], document: VectorDocument) -> dict[str, Any] | None:
        batch_commands = [dict(command) for command in commands[: self.config.max_batch_commands]]
        if len(batch_commands) < self.config.batch_min_command_count:
            return None

        confidence = sum(float(command["confidence"]) for command in batch_commands) / len(batch_commands)
        target_paths = sorted({str(command["path_id"]) for command in batch_commands})
        batch_command = {
            "tool": "propose_batch_refinement",
            "summary": f"Review {len(batch_commands)} proposed primitive refinements across {len(target_paths)} path(s).",
            "commands": batch_commands,
            "confidence": confidence,
            "requires_user_confirmation": True,
            "semantic_source": self.config.semantic_source_prefix,
            "semantic_confidence": confidence,
        }
        try:
            validate_command(batch_command, document)
        except CommandValidationError:
            return None
        return batch_command

    def _covers_full_path(self, candidate: ShapeCandidate, path: Path) -> bool:
        return path.closed and candidate.segment_range == (0, max(len(path.segments) - 1, 0))

    def _semantic_source(self, candidate: ShapeCandidate) -> str:
        if candidate.source:
            return f"{self.config.semantic_source_prefix}:{candidate.source}"
        return self.config.semantic_source_prefix

    def _default_reason(self, candidate: ShapeCandidate) -> str:
        if candidate.target_type == "circle":
            return "This region reads as a circular primitive and should be refined as a circle."
        if candidate.target_type == "ellipse":
            return "This region reads as an elliptical primitive and should be refined as an ellipse."
        if candidate.target_type == "rectangle":
            return "This region reads as a rectangle and its edges should be refined as straight lines."
        if candidate.target_type == "line":
            return "This region reads as a straight edge and should be refined as a line."
        if candidate.target_type == "arc":
            return "This region reads as a circular arc and should be refined as an arc."
        return "This region should be refined as a standard geometric primitive."


__all__ = [
    "ProposedCommandPlanner",
    "ProposedCommandPlannerConfig",
]
