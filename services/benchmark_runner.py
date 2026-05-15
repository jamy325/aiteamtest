from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from xml.etree import ElementTree as ET

from core.types import Point, VectorDocument
from services.command_executor import CommandExecutionResult, CommandExecutor
from services.contour_extractor import BinaryContour, ExtractedContours
from services.dxf_exporter import DxfExporter
from services.edge_error import EdgeErrorCalculator, EdgeErrorResult
from services.json_exporter import JsonExporter
from services.minimal_pipeline import MinimalPipeline, MinimalPipelineResult
from services.scorer import ScoreResult, Scorer
from services.segment_sampler import SegmentSampler
from services.svg_exporter import SvgExporter


GeometryCounts = dict[str, int]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    image_path: str
    expected_geometry: dict[str, int] = field(default_factory=dict)
    expected_constraints: dict[str, int] = field(default_factory=dict)
    expected_export: dict[str, Any] = field(default_factory=dict)
    proposed_commands: tuple[dict[str, Any], ...] = ()
    segment_type: str = "line"
    document_id: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    case_id: str
    image_path: str
    success: bool
    document_id: str
    segment_type: str
    stats: dict[str, float | int]
    actual_geometry: dict[str, int]
    geometry_hits: dict[str, int]
    actual_constraints: dict[str, int]
    constraint_hits: dict[str, int]
    export_summary: dict[str, Any]
    command_results: tuple[dict[str, Any], ...]
    total_score: float
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "image_path": self.image_path,
            "success": self.success,
            "document_id": self.document_id,
            "segment_type": self.segment_type,
            "stats": dict(self.stats),
            "actual_geometry": dict(self.actual_geometry),
            "geometry_hits": dict(self.geometry_hits),
            "actual_constraints": dict(self.actual_constraints),
            "constraint_hits": dict(self.constraint_hits),
            "export_summary": json.loads(json.dumps(self.export_summary)),
            "command_results": [dict(item) for item in self.command_results],
            "total_score": self.total_score,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    cases: tuple[BenchmarkCaseResult, ...]
    summary: dict[str, Any]
    generated_at: str | None = None
    runner_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "runner_version": self.runner_version,
            "cases": [case.to_dict() for case in self.cases],
            "summary": json.loads(json.dumps(self.summary)),
        }
        if self.generated_at is not None:
            payload["generated_at"] = self.generated_at
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


PipelineFactory = Callable[[BenchmarkCase], MinimalPipeline]


class BenchmarkRunner:
    def __init__(
        self,
        *,
        pipeline_factory: PipelineFactory | None = None,
        command_executor: CommandExecutor | None = None,
        scorer: Scorer | None = None,
        edge_error_calculator: EdgeErrorCalculator | None = None,
        segment_sampler: SegmentSampler | None = None,
        json_exporter: JsonExporter | None = None,
        svg_exporter: SvgExporter | None = None,
        dxf_exporter: DxfExporter | None = None,
    ) -> None:
        self.pipeline_factory = pipeline_factory or self._default_pipeline_factory
        self.command_executor = command_executor or CommandExecutor()
        self.scorer = scorer or Scorer()
        self.edge_error_calculator = edge_error_calculator or EdgeErrorCalculator()
        self.segment_sampler = segment_sampler or SegmentSampler()
        self.json_exporter = json_exporter or JsonExporter()
        self.svg_exporter = svg_exporter or SvgExporter()
        self.dxf_exporter = dxf_exporter or DxfExporter()

    def load_manifest(self, manifest_path: str | Path) -> tuple[BenchmarkCase, ...]:
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_payloads = payload.get("cases")
        if not isinstance(case_payloads, list):
            raise ValueError("manifest must contain a list of cases")

        cases: list[BenchmarkCase] = []
        for item in case_payloads:
            if not isinstance(item, dict):
                raise ValueError("benchmark case entries must be objects")
            case_id = item.get("case_id")
            image_path = item.get("image_path")
            if not case_id or not image_path:
                raise ValueError("benchmark case requires case_id and image_path")
            resolved_image_path = self._resolve_manifest_path(path.parent, str(image_path))
            expected_geometry = self._coerce_counts(item.get("expected_geometry"))
            expected_constraints = self._coerce_counts(item.get("expected_constraints"))
            expected_export = self._coerce_dict(item.get("expected_export"))
            proposed_commands = self._coerce_command_list(item.get("proposed_commands") or item.get("commands"))
            segment_type = str(item.get("segment_type", "line"))
            document_id = str(item["document_id"]) if "document_id" in item else None
            cases.append(
                BenchmarkCase(
                    case_id=str(case_id),
                    image_path=str(resolved_image_path),
                    expected_geometry=expected_geometry,
                    expected_constraints=expected_constraints,
                    expected_export=expected_export,
                    proposed_commands=proposed_commands,
                    segment_type=segment_type,
                    document_id=document_id,
                )
            )
        return tuple(cases)

    def run_case(
        self,
        case: BenchmarkCase,
        *,
        execute_proposed_commands: bool = False,
    ) -> BenchmarkCaseResult:
        document_id = case.document_id or f"benchmark_{case.case_id}"
        pipeline = self.pipeline_factory(case)
        pipeline_result = pipeline.run_from_file(case.image_path, document_id=document_id)

        document = pipeline_result.document
        command_results: list[dict[str, Any]] = []
        if execute_proposed_commands:
            for command in case.proposed_commands:
                execution = self.command_executor.execute(command, document)
                command_results.append(self._command_result_payload(execution))
                if execution.success:
                    document = execution.document

        edge_error = self._edge_error(pipeline_result.extracted_contours, document)
        score = self.scorer.score_document(document, edge_error=edge_error)
        actual_geometry = self._geometry_counts(document)
        actual_constraints = self._constraint_counts(document)
        export_summary = self._export_summary(document, case.expected_export)
        stats = self._case_stats(document, score, edge_error)

        return BenchmarkCaseResult(
            case_id=case.case_id,
            image_path=case.image_path,
            success=True,
            document_id=document.document_id,
            segment_type=case.segment_type,
            stats=stats,
            actual_geometry=actual_geometry,
            geometry_hits=self._count_hits(case.expected_geometry, actual_geometry),
            actual_constraints=actual_constraints,
            constraint_hits=self._count_hits(case.expected_constraints, actual_constraints),
            export_summary=export_summary,
            command_results=tuple(command_results),
            total_score=score.total_score,
            failure_reason=None,
        )

    def run_manifest(
        self,
        manifest: str | Path | Sequence[BenchmarkCase],
        *,
        execute_proposed_commands: bool = False,
    ) -> BenchmarkReport:
        cases = self.load_manifest(manifest) if isinstance(manifest, (str, Path)) else tuple(manifest)
        results = tuple(
            self.run_case(case, execute_proposed_commands=execute_proposed_commands)
            for case in cases
        )
        return BenchmarkReport(cases=results, summary=self._summary(results))

    def _default_pipeline_factory(self, case: BenchmarkCase) -> MinimalPipeline:
        return MinimalPipeline(segment_type=case.segment_type)

    def _edge_error(self, contours: ExtractedContours, document: VectorDocument) -> EdgeErrorResult:
        source_points = tuple(
            point
            for contour in contours.binary_contours + contours.skeleton_contours
            for point in contour.points
        )
        vector_points = tuple(
            point
            for segment in document.segments
            for point in self.segment_sampler.sample_segment(segment)
        )
        return self.edge_error_calculator.calculate(source_points, vector_points)

    def _geometry_counts(self, document: VectorDocument) -> GeometryCounts:
        counts: GeometryCounts = {}
        for segment in document.segments:
            counts[segment.type] = counts.get(segment.type, 0) + 1
        for segment_type in ("line", "arc", "circle", "ellipse"):
            counts.setdefault(segment_type, 0)
        return counts

    def _constraint_counts(self, document: VectorDocument) -> dict[str, int]:
        counts: dict[str, int] = {}
        for constraint in document.constraints:
            counts[constraint.type] = counts.get(constraint.type, 0) + 1
        return counts

    def _count_hits(self, expected: Mapping[str, int], actual: Mapping[str, int]) -> dict[str, int]:
        return {
            key: min(max(int(value), 0), max(int(actual.get(key, 0)), 0))
            for key, value in expected.items()
        }

    def _export_summary(self, document: VectorDocument, expected_export: Mapping[str, Any]) -> dict[str, Any]:
        json_payload = self.json_exporter.export_document(document)
        svg_payload = self.svg_exporter.export_document(document)
        dxf_payload = self.dxf_exporter.export_document(document)
        summary = {
            "json": {
                "char_count": len(json_payload),
            },
            "svg": {
                "element_count": self._svg_element_count(svg_payload),
            },
            "dxf": {
                "entity_count": self._dxf_entity_count(dxf_payload),
            },
        }
        if expected_export:
            summary["expected"] = json.loads(json.dumps(expected_export))
        return summary

    def _case_stats(
        self,
        document: VectorDocument,
        score: ScoreResult,
        edge_error: EdgeErrorResult,
    ) -> dict[str, float | int]:
        geometry_counts = self._geometry_counts(document)
        return {
            "total_score": score.total_score,
            "edge_error": score.breakdown.edge_error_score,
            "geometry_complexity": score.breakdown.geometry_complexity_score,
            "topology_error_count": sum(1 for path in document.paths if path.topology_status == "topology_error"),
            "self_intersection_count": sum(int(path.self_intersection_count) for path in document.paths),
            "segment_count": len(document.segments),
            "control_point_count": self._control_point_count(document),
            "line_count": geometry_counts["line"],
            "arc_count": geometry_counts["arc"],
            "circle_count": geometry_counts["circle"],
            "ellipse_count": geometry_counts["ellipse"],
            "source_point_count": edge_error.source_point_count,
            "vector_point_count": edge_error.vector_point_count,
        }

    def _control_point_count(self, document: VectorDocument) -> int:
        total = 0
        for segment in document.segments:
            if segment.type == "bezier":
                total += int("control1" in segment.params) + int("control2" in segment.params)
            elif segment.type == "bspline":
                total += max(0, len(segment.params.get("points", ())) - 2)
        return total

    def _summary(self, results: Sequence[BenchmarkCaseResult]) -> dict[str, Any]:
        total_cases = len(results)
        if total_cases == 0:
            return {
                "total_cases": 0,
                "average_total_score": 0.0,
                "total_segments": 0,
                "total_control_points": 0,
                "geometry_hit_totals": {},
            }

        total_score = sum(float(item.stats["total_score"]) for item in results)
        total_segments = sum(int(item.stats["segment_count"]) for item in results)
        total_control_points = sum(int(item.stats["control_point_count"]) for item in results)
        geometry_hit_totals: dict[str, int] = {}
        for item in results:
            for key, value in item.geometry_hits.items():
                geometry_hit_totals[key] = geometry_hit_totals.get(key, 0) + int(value)

        return {
            "total_cases": total_cases,
            "average_total_score": total_score / total_cases,
            "total_segments": total_segments,
            "total_control_points": total_control_points,
            "geometry_hit_totals": geometry_hit_totals,
            "case_ids": [item.case_id for item in results],
        }

    def _command_result_payload(self, result: CommandExecutionResult) -> dict[str, Any]:
        return {
            "command_id": result.command_id,
            "success": result.success,
            "reason": result.reason,
            "affected_paths": list(result.affected_paths),
            "affected_segments": list(result.affected_segments),
            "topology_status": result.topology_status,
            "self_intersection_count": result.self_intersection_count,
            "requires_rerender": result.requires_rerender,
            "old_score": result.old_score,
            "new_score": result.new_score,
        }

    def _svg_element_count(self, svg_payload: str) -> int:
        root = ET.fromstring(svg_payload)
        return sum(1 for _ in root.iter() if _ is not root)

    def _dxf_entity_count(self, dxf_payload: str) -> int:
        entity_types = {"LINE", "ARC", "CIRCLE", "LWPOLYLINE"}
        lines = dxf_payload.splitlines()
        count = 0
        for index in range(0, len(lines) - 1, 2):
            if lines[index] == "0" and lines[index + 1] in entity_types:
                count += 1
        return count

    def _resolve_manifest_path(self, manifest_dir: Path, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return manifest_dir / path

    def _coerce_counts(self, value: object) -> dict[str, int]:
        payload = self._coerce_dict(value)
        counts: dict[str, int] = {}
        for key, item in payload.items():
            counts[str(key)] = int(item)
        return counts

    def _coerce_dict(self, value: object) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("manifest mapping fields must be objects")
        return dict(value)

    def _coerce_command_list(self, value: object) -> tuple[dict[str, Any], ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise ValueError("proposed_commands must be a list")
        commands: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("proposed_commands entries must be objects")
            commands.append(dict(item))
        return tuple(commands)


def benchmark_report_to_json(report: BenchmarkReport) -> str:
    return report.to_json()


__all__ = [
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkReport",
    "BenchmarkRunner",
    "benchmark_report_to_json",
]
