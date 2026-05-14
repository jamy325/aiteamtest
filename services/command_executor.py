from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from core.precision import PrecisionUtility
from core.types import Path, Point, Segment, VectorDocument, updated
from services.breakpoint_optimizer import BreakPointOptimizer, BreakPointRequest
from services.command_schema import BATCH_TOOL, CommandValidationError, validate_command
from services.fitting_confidence import FittingConfidenceInputs, FittingConfidenceMetric
from services.refiner import (
    PreciseArcFitter,
    PreciseCircleFitter,
    PreciseLineFitter,
    PreciseLineResult,
    RansacArcConfig,
    RansacArcFitter,
    RansacCircleConfig,
    RansacCircleFitter,
    RansacEllipseConfig,
    RansacEllipseFitter,
    RansacLineConfig,
    RansacLineFitter,
)
from services.refinement_feedback import RefinementFeedback, RefinementFeedbackInputs
from services.scorer import Scorer
from services.segment_sampler import SegmentSampler
from services.self_intersection import SelfIntersectionDetector
from services.topology import TopologyEngine

SEGMENT_REPLACE_TOOL_TO_TYPE = {
    "propose_replace_segment_with_line": "line",
    "propose_replace_segment_with_arc": "arc",
    "propose_replace_segment_with_circle": "circle",
    "propose_replace_segment_with_ellipse": "ellipse",
}
PATH_REPLACE_TOOL_TO_TYPE = {
    "propose_replace_path_with_circle": "circle",
}
REPLACE_TOOL_TO_TYPE = SEGMENT_REPLACE_TOOL_TO_TYPE | PATH_REPLACE_TOOL_TO_TYPE


@dataclass(frozen=True, slots=True)
class CommandExecutionResult:
    success: bool
    command_id: str
    document: VectorDocument
    affected_paths: tuple[str, ...]
    affected_segments: tuple[str, ...]
    old_score: float | None
    new_score: float | None
    topology_status: str | None
    self_intersection_count: int
    requires_rerender: bool
    reason: str | None = None


class CommandExecutor:
    def __init__(
        self,
        *,
        breakpoint_optimizer: BreakPointOptimizer | None = None,
        topology_engine: TopologyEngine | None = None,
        self_intersection_detector: SelfIntersectionDetector | None = None,
        scorer: Scorer | None = None,
        segment_sampler: SegmentSampler | None = None,
        fitting_confidence_metric: FittingConfidenceMetric | None = None,
        refinement_feedback: RefinementFeedback | None = None,
    ) -> None:
        self.breakpoint_optimizer = breakpoint_optimizer or BreakPointOptimizer()
        self.topology_engine = topology_engine or TopologyEngine()
        self.self_intersection_detector = self_intersection_detector or SelfIntersectionDetector()
        self.scorer = scorer or Scorer()
        self.segment_sampler = segment_sampler or SegmentSampler()
        self.fitting_confidence_metric = fitting_confidence_metric or FittingConfidenceMetric()
        self.refinement_feedback = refinement_feedback or RefinementFeedback()

    def execute(self, command: object, document: VectorDocument) -> CommandExecutionResult:
        command_id = self._command_id(command)
        old_score = self.scorer.score_document(document).total_score
        affected_path_id = self._command_path_id(command)

        try:
            execution_command, target_type = self._normalize_execution_command(command, document)
            validation = validate_command(execution_command, document)
            if validation.tool == BATCH_TOOL:
                raise ValueError("batch command execution is not supported")
            if validation.target_path_id is None or not validation.target_segment_ids:
                raise ValueError("command does not resolve to an executable segment range")

            path = self._path_by_id(document, validation.target_path_id)
            target_segments = tuple(self._segment_by_id(document, segment_id) for segment_id in validation.target_segment_ids)
            self._ensure_no_dangling_constraints(document, validation.target_segment_ids[1:])
            sampled_points = self._sample_segment_range(target_segments)
            fit_points = self._fit_points_for_command(command, sampled_points, target_type)
            replacement_segment = self._replacement_segment(
                command=command,
                target_type=target_type,
                path_closed=path.closed,
                segments=target_segments,
                points=fit_points,
                support_points=sampled_points,
            )
            replaced_document = self._replace_segment_range(
                document,
                path_id=path.path_id,
                target_segment_ids=validation.target_segment_ids,
                replacement_segment=replacement_segment,
            )
            topology_result = self.topology_engine.enforce_path_topology(replaced_document, path.path_id)
            intersection_result = self.self_intersection_detector.detect_path_self_intersections(
                topology_result.document,
                path.path_id,
            )
            new_score = self.scorer.score_document(intersection_result.document).total_score
            final_path = self._path_by_id(intersection_result.document, path.path_id)

            return CommandExecutionResult(
                success=True,
                command_id=command_id,
                document=intersection_result.document,
                affected_paths=(path.path_id,),
                affected_segments=validation.target_segment_ids,
                old_score=old_score,
                new_score=new_score,
                topology_status=final_path.topology_status,
                self_intersection_count=final_path.self_intersection_count,
                requires_rerender=True,
                reason=None,
            )
        except (CommandValidationError, KeyError, ValueError) as exc:
            current_path = self._try_path(document, affected_path_id)
            return CommandExecutionResult(
                success=False,
                command_id=command_id,
                document=document,
                affected_paths=(affected_path_id,) if current_path is not None else (),
                affected_segments=(),
                old_score=old_score,
                new_score=None,
                topology_status=None if current_path is None else current_path.topology_status,
                self_intersection_count=0 if current_path is None else current_path.self_intersection_count,
                requires_rerender=False,
                reason=str(exc),
            )

    def _replacement_segment(
        self,
        *,
        command: object,
        target_type: str,
        path_closed: bool,
        segments: tuple[Segment, ...],
        points: tuple[Point, ...],
        support_points: tuple[Point, ...],
    ) -> Segment:
        primary_segment = segments[0]
        first_point = points[0]
        last_point = points[-1]
        if path_closed and target_type in {"circle", "ellipse"}:
            last_point = first_point
        scale = self._point_scale(points)
        inlier_threshold = max(0.05, scale * 0.35)
        min_inlier_ratio = 0.6 if len(points) >= 5 else 0.5
        start_anchor_id = primary_segment.anchors[0] if primary_segment.anchors else None
        end_anchor_id = segments[-1].anchors[-1] if segments[-1].anchors else None
        anchors = tuple(dict.fromkeys(anchor_id for anchor_id in (start_anchor_id, end_anchor_id) if anchor_id is not None))

        if target_type == "line":
            ransac = RansacLineFitter(
                RansacLineConfig(
                    iterations=128,
                    inlier_threshold=inlier_threshold,
                    min_inlier_ratio=min_inlier_ratio,
                    random_seed=0,
                )
            )
            initial = ransac.fit(points)
            inlier_points = tuple(points[index] for index in initial.inlier_indexes)
            refined = PreciseLineFitter().fit(inlier_points, initial.params)
            params = self._line_params_covering_support_points(refined.params, support_points)
            confidence = initial.inlier_ratio
            fit_error = refined.rmse
            feedback = self._line_refinement_feedback(
                initial_inlier_ratio=initial.inlier_ratio,
                refined=refined,
                params=params,
            )
            if self._should_reject_line_feedback(feedback.reason, refined.rmse):
                raise ValueError(feedback.reason or feedback.suggestion)
        elif target_type == "arc":
            ransac = RansacArcFitter(
                RansacArcConfig(
                    iterations=128,
                    inlier_threshold=inlier_threshold,
                    min_inlier_ratio=min_inlier_ratio,
                    random_seed=0,
                    min_arc_angle=math.pi / 18.0,
                    max_radial_error=max(inlier_threshold, scale * 0.4),
                )
            )
            initial = ransac.fit(points)
            inlier_points = tuple(points[index] for index in initial.inlier_indexes)
            refined = PreciseArcFitter().fit(inlier_points, initial.params)
            params = dict(refined.params)
            arc_coverage = self._arc_support_coverage(
                support_points,
                (float(params["cx"]), float(params["cy"])),
            )
            confidence = initial.inlier_ratio
            fit_error = refined.rmse
            feedback = self._arc_refinement_feedback(
                initial_inlier_ratio=initial.inlier_ratio,
                initial_fit_error=initial.fit_error,
                refined=refined,
                support_coverage=arc_coverage,
            )
            if self._should_reject_arc_feedback(
                feedback.reason,
                refined.rmse,
                arc_coverage,
            ):
                raise ValueError(feedback.reason or feedback.suggestion)
        elif target_type == "circle":
            ransac = RansacCircleFitter(
                RansacCircleConfig(
                    iterations=128,
                    inlier_threshold=inlier_threshold,
                    min_inlier_ratio=min_inlier_ratio,
                    random_seed=0,
                )
            )
            initial = ransac.fit(points)
            inlier_points = tuple(points[index] for index in initial.inlier_indexes)
            refined = PreciseCircleFitter().fit(inlier_points, initial.params)
            params = dict(refined.params)
            params["start"] = [first_point[0], first_point[1]]
            params["end"] = [last_point[0], last_point[1]]
            confidence = initial.inlier_ratio
            fit_error = refined.rmse
            feedback = self._circle_refinement_feedback(
                initial_inlier_ratio=initial.inlier_ratio,
                initial_fit_error=initial.fit_error,
                refined=refined,
            )
            if self._should_reject_circle_feedback(feedback.reason):
                raise ValueError(feedback.reason or feedback.suggestion)
        elif target_type == "ellipse":
            ransac = RansacEllipseFitter(
                RansacEllipseConfig(
                    max_iterations=200,
                    sample_size=5,
                    max_error=max(inlier_threshold, scale * 0.45),
                    min_inlier_ratio=min_inlier_ratio,
                    random_seed=0,
                )
            )
            refined = ransac.fit(points)
            params = {
                "cx": refined.cx,
                "cy": refined.cy,
                "rx": refined.rx,
                "ry": refined.ry,
                "rotation": refined.rotation,
                "start": [first_point[0], first_point[1]],
                "end": [last_point[0], last_point[1]],
            }
            confidence = refined.inlier_ratio
            fit_error = refined.fit_error
        else:
            raise ValueError(f"unsupported command target type: {target_type}")

        metadata = dict(primary_segment.metadata)
        metadata["executor"] = {
            "tool": self._command_tool(command),
            "command_id": self._command_id(command),
        }
        return Segment(
            segment_id=primary_segment.segment_id,
            path_id=primary_segment.path_id,
            type=target_type,
            params=params,
            anchors=anchors,
            fit_error=fit_error,
            confidence=confidence,
            rigidity=primary_segment.rigidity,
            locked=primary_segment.locked,
            metadata=metadata,
        )

    def _line_refinement_feedback(
        self,
        *,
        initial_inlier_ratio: float,
        refined: PreciseLineResult,
        params: dict[str, object],
    ) -> object:
        start = self._coerce_point(params["start"])
        end = self._coerce_point(params["end"])
        segment_length = PrecisionUtility.distance_between_points(start, end)
        confidence_result = self.fitting_confidence_metric.evaluate(
            FittingConfidenceInputs(
                segment_type="line",
                inlier_ratio=initial_inlier_ratio,
                rmse=refined.rmse,
                segment_length=segment_length,
                parameter_delta=refined.parameter_delta,
            )
        )
        return self.refinement_feedback.evaluate(
            RefinementFeedbackInputs(
                segment_type="line",
                inlier_ratio=initial_inlier_ratio,
                fit_error=refined.rmse,
                confidence_result=confidence_result,
            )
        )

    def _should_reject_line_feedback(self, reason: str | None, rmse: float) -> bool:
        if reason is None:
            return False
        # Short but otherwise stable line replacements can score below the generic
        # confidence threshold; only treat that as fatal when residuals are also high.
        if reason == "low_confidence" and rmse <= (self.refinement_feedback.config.max_fit_error * 0.6):
            return False
        return True

    def _arc_refinement_feedback(
        self,
        *,
        initial_inlier_ratio: float,
        initial_fit_error: float,
        refined: object,
        support_coverage: float,
    ) -> object:
        params = refined.params
        segment_length = support_coverage * float(params["r"])
        confidence_result = self.fitting_confidence_metric.evaluate(
            FittingConfidenceInputs(
                segment_type="arc",
                inlier_ratio=initial_inlier_ratio,
                rmse=refined.rmse,
                segment_length=segment_length,
                parameter_delta=refined.parameter_delta,
                radial_error=initial_fit_error,
                arc_angle_coverage=support_coverage,
            )
        )
        return self.refinement_feedback.evaluate(
            RefinementFeedbackInputs(
                segment_type="arc",
                inlier_ratio=initial_inlier_ratio,
                fit_error=refined.rmse,
                confidence_result=confidence_result,
            )
        )

    def _circle_refinement_feedback(
        self,
        *,
        initial_inlier_ratio: float,
        initial_fit_error: float,
        refined: object,
    ) -> object:
        radius = float(refined.params["r"])
        confidence_result = self.fitting_confidence_metric.evaluate(
            FittingConfidenceInputs(
                segment_type="circle",
                inlier_ratio=initial_inlier_ratio,
                rmse=refined.rmse,
                segment_length=math.tau * radius,
                parameter_delta=refined.parameter_delta,
                radial_error=initial_fit_error,
            )
        )
        return self.refinement_feedback.evaluate(
            RefinementFeedbackInputs(
                segment_type="circle",
                inlier_ratio=initial_inlier_ratio,
                fit_error=refined.rmse,
                confidence_result=confidence_result,
            )
        )

    def _should_reject_arc_feedback(self, reason: str | None, rmse: float, coverage: float) -> bool:
        if reason is None:
            return False
        if (
            reason == "low_confidence"
            and rmse <= (self.refinement_feedback.config.max_fit_error * 0.6)
            and coverage >= self.fitting_confidence_metric.config.min_arc_angle_coverage
        ):
            return False
        return True

    def _should_reject_circle_feedback(self, reason: str | None) -> bool:
        return reason is not None

    def _arc_support_coverage(self, points: tuple[Point, ...], center: Point) -> float:
        if len(points) < 2:
            return 0.0
        raw_angles = tuple(math.atan2(point[1] - center[1], point[0] - center[0]) for point in points)
        unwrapped = [raw_angles[0]]
        for angle in raw_angles[1:]:
            delta = (angle - raw_angles[len(unwrapped) - 1] + math.pi) % (2.0 * math.pi) - math.pi
            unwrapped.append(unwrapped[-1] + delta)
        return abs(unwrapped[-1] - unwrapped[0])

    def _line_params_covering_support_points(
        self,
        refined_params: dict[str, object],
        support_points: tuple[Point, ...],
    ) -> dict[str, object]:
        direction = self._coerce_point(refined_params["direction"])
        origin = self._coerce_point(refined_params["start"])
        offsets = tuple(
            ((point[0] - origin[0]) * direction[0]) + ((point[1] - origin[1]) * direction[1])
            for point in support_points
        )
        start_offset = min(offsets)
        end_offset = max(offsets)
        start = (origin[0] + (direction[0] * start_offset), origin[1] + (direction[1] * start_offset))
        end = (origin[0] + (direction[0] * end_offset), origin[1] + (direction[1] * end_offset))

        params = dict(refined_params)
        params["start"] = [start[0], start[1]]
        params["end"] = [end[0], end[1]]
        return params

    def _fit_points(self, points: tuple[Point, ...], target_type: str) -> tuple[Point, ...]:
        if len(points) < self._min_points(target_type):
            raise ValueError("insufficient sampled points for command execution")

        optimized = self.breakpoint_optimizer.optimize(
            BreakPointRequest(
                points=points,
                rough_range=(0, len(points) - 1),
                target_type=target_type,
            )
        )
        start_index, end_index = optimized.optimized_range
        candidate = self._dedupe_points(points[start_index : end_index + 1])
        if target_type == "arc" and len(candidate) < min(len(points), 5):
            candidate = self._dedupe_points(points)
        if len(candidate) < self._min_points(target_type):
            candidate = self._dedupe_points(points)
        if len(candidate) < self._min_points(target_type):
            raise ValueError("optimized point range is too small for command execution")
        return candidate

    def _fit_points_for_command(
        self,
        command: object,
        sampled_points: tuple[Point, ...],
        target_type: str,
    ) -> tuple[Point, ...]:
        if self._command_tool(command) in PATH_REPLACE_TOOL_TO_TYPE:
            return self._dedupe_points(sampled_points)
        return self._fit_points(sampled_points, target_type)

    def _sample_segment_range(self, segments: tuple[Segment, ...]) -> tuple[Point, ...]:
        sampled_points: list[Point] = []
        for segment in segments:
            current = tuple(self.segment_sampler.sample_segment(segment))
            if not current:
                continue
            if sampled_points and PrecisionUtility.points_close(sampled_points[-1], current[0]):
                sampled_points.extend(current[1:])
            else:
                sampled_points.extend(current)
        return self._dedupe_points(tuple(sampled_points))

    def _replace_segment_range(
        self,
        document: VectorDocument,
        *,
        path_id: str,
        target_segment_ids: tuple[str, ...],
        replacement_segment: Segment,
    ) -> VectorDocument:
        path_index = self._find_path_index(document, path_id)
        if path_index is None:
            raise ValueError(f"unknown path_id: {path_id}")

        path = document.paths[path_index]
        start_index = None
        for index in range(len(path.segments) - len(target_segment_ids) + 1):
            if tuple(path.segments[index : index + len(target_segment_ids)]) == target_segment_ids:
                start_index = index
                break
        if start_index is None:
            raise ValueError("target segment range is not contiguous in path")

        updated_path_segment_ids = (
            path.segments[:start_index]
            + (replacement_segment.segment_id,)
            + path.segments[start_index + len(target_segment_ids) :]
        )
        updated_path = updated(path, segments=updated_path_segment_ids)

        target_segment_id_set = set(target_segment_ids)
        updated_segments: list[Segment] = []
        inserted = False
        for segment in document.segments:
            if segment.segment_id == replacement_segment.segment_id:
                updated_segments.append(replacement_segment)
                inserted = True
                continue
            if segment.segment_id in target_segment_id_set:
                continue
            updated_segments.append(segment)
        if not inserted:
            raise ValueError(f"unable to replace segment {replacement_segment.segment_id}")

        paths = list(document.paths)
        paths[path_index] = updated_path
        return updated(document, paths=tuple(paths), segments=tuple(updated_segments))

    def _ensure_no_dangling_constraints(self, document: VectorDocument, removed_segment_ids: tuple[str, ...]) -> None:
        if not removed_segment_ids:
            return
        removed_segment_id_set = set(removed_segment_ids)
        for constraint in document.constraints:
            if any(target in removed_segment_id_set for target in constraint.targets):
                raise ValueError("segment range is referenced by a constraint and cannot be replaced safely")

    def _path_by_id(self, document: VectorDocument, path_id: str) -> Path:
        path = self._try_path(document, path_id)
        if path is None:
            raise ValueError(f"unknown path_id: {path_id}")
        return path

    def _try_path(self, document: VectorDocument, path_id: str | None) -> Path | None:
        if path_id is None:
            return None
        for path in document.paths:
            if path.path_id == path_id:
                return path
        return None

    def _segment_by_id(self, document: VectorDocument, segment_id: str) -> Segment:
        for segment in document.segments:
            if segment.segment_id == segment_id:
                return segment
        raise ValueError(f"unknown segment_id: {segment_id}")

    def _find_path_index(self, document: VectorDocument, path_id: str) -> int | None:
        for index, path in enumerate(document.paths):
            if path.path_id == path_id:
                return index
        return None

    def _dedupe_points(self, points: tuple[Point, ...]) -> tuple[Point, ...]:
        if not points:
            return ()
        deduped = [points[0]]
        for point in points[1:]:
            if PrecisionUtility.points_close(deduped[-1], point):
                continue
            deduped.append(point)
        return tuple(deduped)

    def _point_scale(self, points: tuple[Point, ...]) -> float:
        if len(points) < 2:
            return 1.0
        lengths = [
            PrecisionUtility.distance_between_points(points[index], points[index + 1])
            for index in range(len(points) - 1)
            if not PrecisionUtility.points_close(points[index], points[index + 1])
        ]
        if lengths:
            return sum(lengths) / len(lengths)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)

    def _min_points(self, target_type: str) -> int:
        if target_type == "line":
            return 2
        if target_type in {"arc", "circle"}:
            return 3
        if target_type == "ellipse":
            return 5
        raise ValueError(f"unsupported command target type: {target_type}")

    def _command_id(self, command: object) -> str:
        if isinstance(command, dict):
            if "command_id" in command:
                return str(command["command_id"])
            if "tool" in command:
                return str(command["tool"])
        return "unknown_command"

    def _command_path_id(self, command: object) -> str | None:
        if isinstance(command, dict) and "path_id" in command:
            return str(command["path_id"])
        return None

    def _command_tool(self, command: object) -> str:
        if isinstance(command, dict) and "tool" in command:
            return str(command["tool"])
        return "unknown_tool"

    def _normalize_execution_command(self, command: object, document: VectorDocument) -> tuple[object, str]:
        tool = self._command_tool(command)
        if tool == "propose_replace_path_with_circle":
            if not isinstance(command, dict):
                raise ValueError("command must be a dictionary")
            if "segment_range" in command:
                raise ValueError("path circle replacement does not accept segment_range")
            path_id = command.get("path_id")
            if path_id is None:
                raise ValueError("path_id is required for circle path replacement")
            path = self._path_by_id(document, str(path_id))
            if not path.closed:
                raise ValueError(f"path must be closed for circle replacement: {path.path_id}")
            if not path.segments:
                raise ValueError(f"path has no segments for circle replacement: {path.path_id}")
            normalized_command = dict(command)
            normalized_command["tool"] = "propose_replace_segment_with_circle"
            normalized_command["segment_range"] = [0, len(path.segments) - 1]
            return normalized_command, "circle"
        if tool in SEGMENT_REPLACE_TOOL_TO_TYPE:
            return command, SEGMENT_REPLACE_TOOL_TO_TYPE[tool]
        return command, REPLACE_TOOL_TO_TYPE.get(tool, "")

    def _coerce_point(self, value: object) -> Point:
        x, y = value  # type: ignore[misc]
        return (float(x), float(y))


__all__ = [
    "CommandExecutionResult",
    "CommandExecutor",
    "SEGMENT_REPLACE_TOOL_TO_TYPE",
]
