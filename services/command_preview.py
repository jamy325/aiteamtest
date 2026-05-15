from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import uuid
from xml.etree import ElementTree as ET

from core.types import VectorDocument
from services.command_executor import CommandExecutionResult, CommandExecutor
from services.dxf_exporter import DxfExporter
from services.json_exporter import JsonExporter
from services.svg_exporter import SvgExporter


@dataclass(frozen=True, slots=True)
class ExportImpactSummary:
    before: dict[str, int]
    after: dict[str, int]
    delta: dict[str, int]


@dataclass(frozen=True, slots=True)
class CommandPreviewResult:
    success: bool
    command_id: str
    reason: str | None
    old_score: float | None
    predicted_new_score: float | None
    score_delta: float | None
    affected_paths: tuple[str, ...]
    affected_segments: tuple[str, ...]
    topology_status_before: dict[str, str]
    topology_status_after: dict[str, str]
    self_intersection_count_before: dict[str, int]
    self_intersection_count_after: dict[str, int]
    segment_type_summary: dict[str, dict[str, int]]
    export_impact_summary: ExportImpactSummary


@dataclass(frozen=True, slots=True)
class BatchCommandPreviewResult:
    batch_id: str
    success_count: int
    failure_count: int
    previews: tuple[CommandPreviewResult, ...]


class CommandPreviewService:
    def __init__(
        self,
        *,
        command_executor: CommandExecutor | None = None,
        json_exporter: JsonExporter | None = None,
        svg_exporter: SvgExporter | None = None,
        dxf_exporter: DxfExporter | None = None,
    ) -> None:
        self.command_executor = command_executor or CommandExecutor()
        self.json_exporter = json_exporter or JsonExporter()
        self.svg_exporter = svg_exporter or SvgExporter()
        self.dxf_exporter = dxf_exporter or DxfExporter()

    def preview(self, command: object, document: VectorDocument) -> CommandPreviewResult:
        before_document = deepcopy(document)
        execution_result = self.command_executor.execute(command, deepcopy(document))
        after_document = execution_result.document if execution_result.success else before_document
        return self._preview_result(before_document, after_document, execution_result)

    def preview_batch(
        self,
        commands: list[object] | tuple[object, ...],
        document: VectorDocument,
        *,
        continue_on_failure: bool = True,
    ) -> BatchCommandPreviewResult:
        batch_id = f"preview_batch_{uuid.uuid4().hex[:8]}"
        current_document = deepcopy(document)
        previews: list[CommandPreviewResult] = []

        for command in commands:
            before_document = deepcopy(current_document)
            execution_result = self.command_executor.execute(command, deepcopy(current_document))
            after_document = execution_result.document if execution_result.success else before_document
            previews.append(self._preview_result(before_document, after_document, execution_result))
            if execution_result.success:
                current_document = execution_result.document
                continue
            if not continue_on_failure:
                break

        success_count = sum(1 for preview in previews if preview.success)
        failure_count = len(previews) - success_count
        return BatchCommandPreviewResult(
            batch_id=batch_id,
            success_count=success_count,
            failure_count=failure_count,
            previews=tuple(previews),
        )

    def _preview_result(
        self,
        before_document: VectorDocument,
        after_document: VectorDocument,
        execution_result: CommandExecutionResult,
    ) -> CommandPreviewResult:
        old_score = execution_result.old_score
        predicted_new_score = execution_result.new_score
        score_delta = None
        if old_score is not None and predicted_new_score is not None:
            score_delta = predicted_new_score - old_score

        affected_paths = execution_result.affected_paths
        topology_status_before = self._path_topology_summary(before_document, affected_paths)
        topology_status_after = self._path_topology_summary(after_document, affected_paths)
        self_intersection_count_before = self._path_intersection_summary(before_document, affected_paths)
        self_intersection_count_after = self._path_intersection_summary(after_document, affected_paths)

        before_counts = self._segment_type_counts(before_document)
        after_counts = self._segment_type_counts(after_document)
        segment_types = sorted(set(before_counts) | set(after_counts))
        delta_counts = {
            segment_type: after_counts.get(segment_type, 0) - before_counts.get(segment_type, 0)
            for segment_type in segment_types
        }

        return CommandPreviewResult(
            success=execution_result.success,
            command_id=execution_result.command_id,
            reason=execution_result.reason,
            old_score=old_score,
            predicted_new_score=predicted_new_score,
            score_delta=score_delta,
            affected_paths=affected_paths,
            affected_segments=execution_result.affected_segments,
            topology_status_before=topology_status_before,
            topology_status_after=topology_status_after,
            self_intersection_count_before=self_intersection_count_before,
            self_intersection_count_after=self_intersection_count_after,
            segment_type_summary={
                "before": before_counts,
                "after": after_counts,
                "delta": delta_counts,
            },
            export_impact_summary=self._export_impact_summary(before_document, after_document),
        )

    def _path_topology_summary(
        self,
        document: VectorDocument,
        affected_paths: tuple[str, ...],
    ) -> dict[str, str]:
        if not affected_paths:
            return {}
        path_ids = set(affected_paths)
        return {
            path.path_id: path.topology_status
            for path in document.paths
            if path.path_id in path_ids
        }

    def _path_intersection_summary(
        self,
        document: VectorDocument,
        affected_paths: tuple[str, ...],
    ) -> dict[str, int]:
        if not affected_paths:
            return {}
        path_ids = set(affected_paths)
        return {
            path.path_id: int(path.self_intersection_count)
            for path in document.paths
            if path.path_id in path_ids
        }

    def _segment_type_counts(self, document: VectorDocument) -> dict[str, int]:
        counts: dict[str, int] = {}
        for segment in document.segments:
            counts[segment.type] = counts.get(segment.type, 0) + 1
        return counts

    def _export_impact_summary(
        self,
        before_document: VectorDocument,
        after_document: VectorDocument,
    ) -> ExportImpactSummary:
        before = self._export_metrics(before_document)
        after = self._export_metrics(after_document)
        keys = sorted(set(before) | set(after))
        delta = {
            key: after.get(key, 0) - before.get(key, 0)
            for key in keys
        }
        return ExportImpactSummary(before=before, after=after, delta=delta)

    def _export_metrics(self, document: VectorDocument) -> dict[str, int]:
        json_payload = self.json_exporter.export_document(document)
        try:
            svg_payload = self.svg_exporter.export_document(document)
            svg_element_count = self._svg_element_count(svg_payload)
        except Exception:
            svg_element_count = 0

        try:
            dxf_payload = self.dxf_exporter.export_document(document)
            dxf_entity_count = self._dxf_entity_count(dxf_payload)
        except Exception:
            dxf_entity_count = 0

        return {
            "json_char_count": len(json_payload),
            "svg_element_count": svg_element_count,
            "dxf_entity_count": dxf_entity_count,
        }

    def _svg_element_count(self, svg_payload: str) -> int:
        root = ET.fromstring(svg_payload)
        return sum(1 for node in root.iter() if node is not root)

    def _dxf_entity_count(self, dxf_payload: str) -> int:
        entity_types = {"LINE", "ARC", "CIRCLE", "LWPOLYLINE"}
        lines = dxf_payload.splitlines()
        count = 0
        for index in range(0, len(lines) - 1, 2):
            if lines[index] == "0" and lines[index + 1] in entity_types:
                count += 1
        return count


__all__ = [
    "BatchCommandPreviewResult",
    "CommandPreviewResult",
    "CommandPreviewService",
    "ExportImpactSummary",
]
