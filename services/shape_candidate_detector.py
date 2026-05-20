from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from core.precision import PrecisionUtility
from core.types import Path, Point, Segment, ShapeCandidate, VectorDocument
from services.ellipse_fitter import RansacEllipseConfig, RansacEllipseFitter, RansacEllipseResult
from services.fitting_confidence import FittingConfidenceInputs, FittingConfidenceMetric
from services.refiner import (
    PreciseArcFitter,
    PreciseCircleFitter,
    PreciseLineFitter,
    RansacArcConfig,
    RansacArcFitter,
    RansacCircleConfig,
    RansacCircleFitter,
    RansacLineConfig,
    RansacLineFitter,
)
from services.segment_sampler import SegmentSampler


@dataclass(frozen=True, slots=True)
class ShapeCandidateDetectorConfig:
    min_path_points: int = 2
    min_circle_points: int = 24
    min_ellipse_points: int = 24
    min_rectangle_points: int = 4
    min_arc_points: int = 5
    min_bbox_extent: float = 4.0
    min_path_diagonal: float = 6.0
    min_line_length: float = 8.0
    min_arc_length: float = 10.0
    min_arc_angle_coverage: float = math.pi / 10.0
    max_arc_angle_coverage: float = math.tau * 0.9
    max_circle_aspect_delta: float = 0.18
    circle_aspect_hard_limit: float = 0.28
    min_circle_confidence: float = 0.62
    min_ellipse_confidence: float = 0.7
    min_rectangle_confidence: float = 0.72
    min_line_confidence: float = 0.7
    min_arc_confidence: float = 0.68
    rectangle_rdp_epsilon_ratio: float = 0.025
    rectangle_min_rdp_epsilon: float = 1.0
    rectangle_angle_tolerance: float = math.pi / 9.0
    rectangle_parallel_tolerance: float = math.pi / 10.0
    rectangle_edge_ratio_tolerance: float = 0.35
    max_segment_window: int = 4
    filter_tiny_paths: bool = True
    filter_open_paths_for_closed_candidates: bool = True
    prefer_raw_source_points: bool = True
    line_ransac_config: RansacLineConfig = field(default_factory=RansacLineConfig)
    circle_ransac_config: RansacCircleConfig = field(default_factory=RansacCircleConfig)
    arc_ransac_config: RansacArcConfig = field(default_factory=RansacArcConfig)
    ellipse_ransac_config: RansacEllipseConfig = field(default_factory=RansacEllipseConfig)


class ShapeCandidateDetector:
    def __init__(
        self,
        config: ShapeCandidateDetectorConfig | None = None,
        *,
        segment_sampler: SegmentSampler | None = None,
        fitting_confidence_metric: FittingConfidenceMetric | None = None,
    ) -> None:
        self.config = config or ShapeCandidateDetectorConfig()
        self.segment_sampler = segment_sampler or SegmentSampler()
        self.fitting_confidence_metric = fitting_confidence_metric or FittingConfidenceMetric()
        self.line_ransac_fitter = RansacLineFitter(self.config.line_ransac_config)
        self.circle_ransac_fitter = RansacCircleFitter(self.config.circle_ransac_config)
        self.arc_ransac_fitter = RansacArcFitter(self.config.arc_ransac_config)
        self.ellipse_ransac_fitter = RansacEllipseFitter(self.config.ellipse_ransac_config)
        self.line_precise_fitter = PreciseLineFitter()
        self.circle_precise_fitter = PreciseCircleFitter()
        self.arc_precise_fitter = PreciseArcFitter()

    def detect_candidates(self, document: VectorDocument) -> tuple[ShapeCandidate, ...]:
        candidates: list[ShapeCandidate] = []

        for path in document.paths:
            segments = self._path_segments(document, path)
            if not segments:
                continue
            if self.config.filter_tiny_paths and self._is_tiny_path(segments):
                continue

            path_candidates = self._detect_path_candidates(document, path, segments)
            range_candidates = self._detect_segment_range_candidates(document, path, segments)
            candidates.extend(path_candidates)
            candidates.extend(range_candidates)

        deduped = self._dedupe_candidates(candidates)
        return tuple(sorted(deduped, key=lambda item: (-item.confidence, item.candidate_id)))

    def _detect_path_candidates(
        self,
        document: VectorDocument,
        path: Path,
        segments: tuple[Segment, ...],
    ) -> tuple[ShapeCandidate, ...]:
        if self.config.filter_open_paths_for_closed_candidates and not path.closed:
            return ()

        points, source, raw_point_count = self._path_points(document, path, segments)
        if len(points) < self.config.min_rectangle_points:
            return ()

        bbox = self._bbox(points)
        if bbox["width"] < self.config.min_bbox_extent or bbox["height"] < self.config.min_bbox_extent:
            return ()

        circle_candidate = self._circle_candidate(path, segments, points, source, raw_point_count, bbox)
        ellipse_candidate = self._ellipse_candidate(path, segments, points, source, raw_point_count, bbox)
        rectangle_candidate = self._rectangle_candidate(path, segments, points, source, raw_point_count, bbox)

        result = [candidate for candidate in (circle_candidate, ellipse_candidate, rectangle_candidate) if candidate is not None]
        if circle_candidate is not None and ellipse_candidate is not None:
            aspect_ratio = float(circle_candidate.evidence["aspect_ratio"])
            if abs(aspect_ratio - 1.0) <= self.config.max_circle_aspect_delta and circle_candidate.confidence >= ellipse_candidate.confidence:
                result = [candidate for candidate in result if candidate.target_type != "ellipse"]
        return tuple(result)

    def _detect_segment_range_candidates(
        self,
        document: VectorDocument,
        path: Path,
        segments: tuple[Segment, ...],
    ) -> tuple[ShapeCandidate, ...]:
        candidates: list[ShapeCandidate] = []
        max_window = min(len(segments), max(1, self.config.max_segment_window))

        for window in range(1, max_window + 1):
            for start_index in range(0, len(segments) - window + 1):
                end_index = start_index + window - 1
                points, source, raw_point_count = self._segment_range_points(document, path, segments, start_index, end_index)
                if len(points) < self.config.min_path_points:
                    continue
                bbox = self._bbox(points)
                if bbox["diagonal"] < self.config.min_path_diagonal:
                    continue

                line_candidate = self._line_candidate(
                    path,
                    segments,
                    points,
                    source,
                    raw_point_count,
                    bbox,
                    start_index,
                    end_index,
                )
                arc_candidate = self._arc_candidate(
                    path,
                    segments,
                    points,
                    source,
                    raw_point_count,
                    bbox,
                    start_index,
                    end_index,
                )
                if line_candidate is not None:
                    candidates.append(line_candidate)
                if arc_candidate is not None:
                    candidates.append(arc_candidate)

        return tuple(candidates)

    def _circle_candidate(
        self,
        path: Path,
        segments: tuple[Segment, ...],
        points: tuple[Point, ...],
        source: str,
        raw_point_count: int,
        bbox: dict[str, float],
    ) -> ShapeCandidate | None:
        aspect_ratio = self._aspect_ratio(bbox)
        if len(points) < self.config.min_circle_points:
            return None
        if abs(aspect_ratio - 1.0) > self.config.circle_aspect_hard_limit:
            return None

        try:
            ransac = self.circle_ransac_fitter.fit(points)
            inlier_points = tuple(points[index] for index in ransac.inlier_indexes)
            precise = self.circle_precise_fitter.fit(inlier_points, ransac.params)
        except ValueError:
            return None

        aspect_score = max(0.0, 1.0 - (abs(aspect_ratio - 1.0) / max(self.config.max_circle_aspect_delta, 1e-9)))
        fit_score = self._lower_is_better(precise.rmse, target=0.06, maximum=0.18)
        inlier_score = self._higher_is_better(ransac.inlier_ratio, minimum=0.65, target=0.95)
        confidence = max(0.0, min(1.0, (fit_score * 0.55) + (inlier_score * 0.3) + (aspect_score * 0.15)))
        if confidence < self.config.min_circle_confidence:
            return None

        return ShapeCandidate(
            candidate_id=f"{path.path_id}:circle:0-{len(segments) - 1}",
            target_type="circle",
            path_id=path.path_id,
            segment_range=(0, len(segments) - 1),
            source=source,
            confidence=confidence,
            evidence=self._base_evidence(
                source=source,
                raw_point_count=raw_point_count,
                fit_point_count=len(points),
                segment_count=len(segments),
                bbox=bbox,
                fit_error=precise.rmse,
                inlier_ratio=ransac.inlier_ratio,
                model_complexity_delta=max(0, len(segments) - 1),
                aspect_ratio=aspect_ratio,
                extra={
                    "center": [float(precise.params["cx"]), float(precise.params["cy"])],
                    "radius": float(precise.params["r"]),
                    "fit_score": fit_score,
                    "aspect_score": aspect_score,
                    "inlier_score": inlier_score,
                },
            ),
            reason="closed path has near-square bbox and high-confidence circle fit",
        )

    def _ellipse_candidate(
        self,
        path: Path,
        segments: tuple[Segment, ...],
        points: tuple[Point, ...],
        source: str,
        raw_point_count: int,
        bbox: dict[str, float],
    ) -> ShapeCandidate | None:
        if len(points) < self.config.min_ellipse_points:
            return None
        try:
            result = self.ellipse_ransac_fitter.fit(points)
        except ValueError:
            return None

        axis_ratio = min(result.rx, result.ry) / max(result.rx, result.ry)
        fit_score = self._lower_is_better(result.fit_error, target=0.03, maximum=self.config.ellipse_ransac_config.max_error)
        inlier_score = self._higher_is_better(
            result.inlier_ratio,
            minimum=self.config.ellipse_ransac_config.min_inlier_ratio,
            target=0.95,
        )
        axis_score = max(0.0, min(1.0, axis_ratio / 0.9))
        confidence = max(0.0, min(1.0, (fit_score * 0.45) + (inlier_score * 0.4) + (axis_score * 0.15)))
        if confidence < self.config.min_ellipse_confidence:
            return None

        return ShapeCandidate(
            candidate_id=f"{path.path_id}:ellipse:0-{len(segments) - 1}",
            target_type="ellipse",
            path_id=path.path_id,
            segment_range=(0, len(segments) - 1),
            source=source,
            confidence=confidence,
            evidence=self._base_evidence(
                source=source,
                raw_point_count=raw_point_count,
                fit_point_count=len(points),
                segment_count=len(segments),
                bbox=bbox,
                fit_error=result.fit_error,
                inlier_ratio=result.inlier_ratio,
                model_complexity_delta=max(0, len(segments) - 1),
                aspect_ratio=self._aspect_ratio(bbox),
                extra={
                    "center": [result.cx, result.cy],
                    "axes": [result.rx, result.ry],
                    "rotation": result.rotation,
                    "axis_ratio": axis_ratio,
                },
            ),
            reason="closed path has high-confidence ellipse fit",
        )

    def _rectangle_candidate(
        self,
        path: Path,
        segments: tuple[Segment, ...],
        points: tuple[Point, ...],
        source: str,
        raw_point_count: int,
        bbox: dict[str, float],
    ) -> ShapeCandidate | None:
        if len(points) < self.config.min_rectangle_points:
            return None
        simplified = self._simplify_closed_polygon(points, bbox["diagonal"])
        if len(simplified) != 4:
            return None

        edges = [self._edge(simplified[index], simplified[(index + 1) % 4]) for index in range(4)]
        if any(edge["length"] < self.config.min_bbox_extent for edge in edges):
            return None

        angle_errors = [
            abs((self._angle_between(edges[index]["vector"], edges[(index + 1) % 4]["vector"])) - (math.pi / 2.0))
            for index in range(4)
        ]
        right_angle_score = 1.0 - max(angle_errors) / max(self.config.rectangle_angle_tolerance, 1e-9)
        parallel_errors = [
            self._parallel_error(edges[0]["vector"], edges[2]["vector"]),
            self._parallel_error(edges[1]["vector"], edges[3]["vector"]),
        ]
        parallel_score = 1.0 - max(parallel_errors) / max(self.config.rectangle_parallel_tolerance, 1e-9)
        opposite_length_score = min(
            self._length_pair_score(edges[0]["length"], edges[2]["length"]),
            self._length_pair_score(edges[1]["length"], edges[3]["length"]),
        )
        confidence = max(
            0.0,
            min(1.0, (right_angle_score * 0.45) + (parallel_score * 0.35) + (opposite_length_score * 0.2)),
        )
        if confidence < self.config.min_rectangle_confidence:
            return None

        return ShapeCandidate(
            candidate_id=f"{path.path_id}:rectangle:0-{len(segments) - 1}",
            target_type="rectangle",
            path_id=path.path_id,
            segment_range=(0, len(segments) - 1),
            source=source,
            confidence=confidence,
            evidence=self._base_evidence(
                source=source,
                raw_point_count=raw_point_count,
                fit_point_count=len(points),
                segment_count=len(segments),
                bbox=bbox,
                fit_error=max(angle_errors + parallel_errors),
                inlier_ratio=1.0,
                model_complexity_delta=max(0, len(segments) - 4),
                aspect_ratio=self._aspect_ratio(bbox),
                extra={
                    "corner_count": 4,
                    "corner_points": [list(point) for point in simplified],
                    "right_angle_error": max(angle_errors),
                    "parallel_error": max(parallel_errors),
                    "opposite_length_score": opposite_length_score,
                },
            ),
            reason="closed path simplifies to four near-orthogonal, opposite-parallel edges",
        )

    def _line_candidate(
        self,
        path: Path,
        segments: tuple[Segment, ...],
        points: tuple[Point, ...],
        source: str,
        raw_point_count: int,
        bbox: dict[str, float],
        start_index: int,
        end_index: int,
    ) -> ShapeCandidate | None:
        if len(points) < 2:
            return None
        segment_length = _polyline_length(points)
        if segment_length < self.config.min_line_length:
            return None
        try:
            ransac = self.line_ransac_fitter.fit(points)
            inlier_points = tuple(points[index] for index in ransac.inlier_indexes)
            precise = self.line_precise_fitter.fit(inlier_points, ransac.params)
        except ValueError:
            return None

        chord_length = PrecisionUtility.distance_between_points(points[0], points[-1])
        straightness_score = max(0.0, min(1.0, chord_length / max(segment_length, 1e-9)))
        fit_score = self._lower_is_better(precise.rmse, target=0.02, maximum=0.15)
        inlier_score = self._higher_is_better(ransac.inlier_ratio, minimum=0.6, target=0.95)
        confidence = max(0.0, min(1.0, (fit_score * 0.5) + (straightness_score * 0.35) + (inlier_score * 0.15)))
        if confidence < self.config.min_line_confidence:
            return None

        range_segment_count = end_index - start_index + 1
        return ShapeCandidate(
            candidate_id=f"{path.path_id}:line:{start_index}-{end_index}",
            target_type="line",
            path_id=path.path_id,
            segment_range=(start_index, end_index),
            source=source,
            confidence=confidence,
            evidence=self._base_evidence(
                source=source,
                raw_point_count=raw_point_count,
                fit_point_count=len(points),
                segment_count=range_segment_count,
                bbox=bbox,
                fit_error=precise.rmse,
                inlier_ratio=ransac.inlier_ratio,
                model_complexity_delta=max(0, range_segment_count - 1),
                aspect_ratio=self._aspect_ratio(bbox),
                extra={
                    "line_length": segment_length,
                    "chord_length": chord_length,
                    "straightness_score": straightness_score,
                    "fit_score": fit_score,
                    "inlier_score": inlier_score,
                },
            ),
            reason="segment range has low-error line fit and sufficient length",
        )

    def _arc_candidate(
        self,
        path: Path,
        segments: tuple[Segment, ...],
        points: tuple[Point, ...],
        source: str,
        raw_point_count: int,
        bbox: dict[str, float],
        start_index: int,
        end_index: int,
    ) -> ShapeCandidate | None:
        if len(points) < self.config.min_arc_points:
            return None
        segment_length = _polyline_length(points)
        if segment_length < self.config.min_arc_length:
            return None
        try:
            ransac = self.arc_ransac_fitter.fit(points)
            inlier_points = tuple(points[index] for index in ransac.inlier_indexes)
            precise = self.arc_precise_fitter.fit(inlier_points, ransac.params)
        except ValueError:
            return None

        arc_coverage = _arc_angle_coverage(precise.params)
        if arc_coverage < self.config.min_arc_angle_coverage or arc_coverage > self.config.max_arc_angle_coverage:
            return None

        metric = self.fitting_confidence_metric.evaluate(
            FittingConfidenceInputs(
                segment_type="arc",
                inlier_ratio=ransac.inlier_ratio,
                rmse=precise.rmse,
                segment_length=segment_length,
                radial_error=precise.rmse,
                arc_angle_coverage=arc_coverage,
                parameter_delta=precise.parameter_delta,
            )
        )
        if metric.confidence < self.config.min_arc_confidence:
            return None

        range_segment_count = end_index - start_index + 1
        return ShapeCandidate(
            candidate_id=f"{path.path_id}:arc:{start_index}-{end_index}",
            target_type="arc",
            path_id=path.path_id,
            segment_range=(start_index, end_index),
            source=source,
            confidence=metric.confidence,
            evidence=self._base_evidence(
                source=source,
                raw_point_count=raw_point_count,
                fit_point_count=len(points),
                segment_count=range_segment_count,
                bbox=bbox,
                fit_error=precise.rmse,
                inlier_ratio=ransac.inlier_ratio,
                model_complexity_delta=max(0, range_segment_count - 1),
                aspect_ratio=self._aspect_ratio(bbox),
                extra={
                    "failure_reason": metric.failure_reason,
                    "arc_angle_coverage": arc_coverage,
                    "radius": float(precise.params["r"]),
                },
            ),
            reason="segment range has high-confidence arc fit and is not a full circle",
        )

    def _path_segments(self, document: VectorDocument, path: Path) -> tuple[Segment, ...]:
        by_id = {segment.segment_id: segment for segment in document.segments}
        return tuple(by_id[segment_id] for segment_id in path.segments if segment_id in by_id)

    def _path_points(
        self,
        document: VectorDocument,
        path: Path,
        segments: tuple[Segment, ...],
    ) -> tuple[tuple[Point, ...], str, int]:
        raw_points = self._raw_source_points_for_path(document, path)
        if self.config.prefer_raw_source_points and raw_points is not None and len(raw_points) >= self.config.min_path_points:
            return (
                self._prepared_raw_fitting_points(raw_points, path_closed=path.closed),
                "raw_contour_points",
                len(raw_points),
            )

        sampled = self._sample_segment_range(segments)
        return (sampled, "segment_samples_fallback", 0 if raw_points is None else len(raw_points))

    def _segment_range_points(
        self,
        document: VectorDocument,
        path: Path,
        segments: tuple[Segment, ...],
        start_index: int,
        end_index: int,
    ) -> tuple[tuple[Point, ...], str, int]:
        selected_segments = segments[start_index : end_index + 1]
        support_points = self._sample_segment_range(selected_segments)
        raw_points = self._raw_source_points_for_path(document, path)
        if self.config.prefer_raw_source_points and raw_points is not None:
            raw_range = self._raw_points_for_segment_range(path, selected_segments, raw_points, support_points)
            if raw_range is not None and len(raw_range) >= self.config.min_path_points:
                return (
                    self._prepared_raw_fitting_points(raw_range, path_closed=path.closed),
                    "raw_contour_points",
                    len(raw_range),
                )
        return (support_points, "segment_samples_fallback", 0 if raw_points is None else len(raw_points))

    def _raw_source_points_for_path(self, document: VectorDocument, path: Path) -> tuple[Point, ...] | None:
        contour_id = path.metadata.get("source_contour_id")
        if contour_id is None:
            return None
        pipeline = document.metadata.get("pipeline")
        if not isinstance(pipeline, dict):
            return None
        contour_group = pipeline.get("source_contours")
        if not isinstance(contour_group, dict):
            return None

        for group_name in ("binary_contours", "skeleton_contours"):
            contours = contour_group.get(group_name, ())
            if not isinstance(contours, list):
                continue
            for contour in contours:
                if not isinstance(contour, dict) or contour.get("contour_id") != contour_id:
                    continue
                points = contour.get("points")
                if not isinstance(points, list):
                    return None
                coerced = self._coerce_points(points)
                if not coerced:
                    return None
                deduped = self._dedupe_points(coerced)
                if len(deduped) > 1 and PrecisionUtility.points_close(deduped[0], deduped[-1]):
                    deduped = deduped[:-1]
                return deduped
        return None

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

    def _raw_points_for_segment_range(
        self,
        path: Path,
        segments: tuple[Segment, ...],
        raw_points: tuple[Point, ...],
        support_points: tuple[Point, ...],
    ) -> tuple[Point, ...] | None:
        start_point = self._segment_start_point(segments[0])
        end_point = self._segment_end_point(segments[-1])
        if start_point is None or end_point is None:
            return None
        start_index = self._nearest_point_index(raw_points, start_point)
        end_index = self._nearest_point_index(raw_points, end_point)
        if start_index is None or end_index is None:
            return None

        if path.closed:
            forward = self._closed_point_slice(raw_points, start_index, end_index)
            backward = tuple(reversed(self._closed_point_slice(raw_points, end_index, start_index)))
            candidate = self._best_oriented_sequence((forward, backward), support_points)
        else:
            low = min(start_index, end_index)
            high = max(start_index, end_index)
            base = raw_points[low : high + 1]
            forward = base if start_index <= end_index else tuple(reversed(base))
            backward = tuple(reversed(forward))
            candidate = self._best_oriented_sequence((forward, backward), support_points)

        deduped = self._dedupe_points(candidate)
        return deduped if deduped else None

    def _best_oriented_sequence(
        self,
        candidates: tuple[tuple[Point, ...], ...],
        support_points: tuple[Point, ...],
    ) -> tuple[Point, ...]:
        if not support_points:
            return candidates[0]
        best = candidates[0]
        best_score = math.inf
        for candidate in candidates:
            if not candidate:
                continue
            score = (
                PrecisionUtility.distance_between_points(candidate[0], support_points[0])
                + PrecisionUtility.distance_between_points(candidate[-1], support_points[-1])
            )
            if len(candidate) > 2 and len(support_points) > 2:
                score += PrecisionUtility.distance_between_points(
                    candidate[len(candidate) // 2],
                    support_points[len(support_points) // 2],
                )
            if score < best_score:
                best = candidate
                best_score = score
        return best

    def _prepared_raw_fitting_points(
        self,
        points: tuple[Point, ...],
        *,
        path_closed: bool,
    ) -> tuple[Point, ...]:
        deduped = self._dedupe_points(points)
        if len(deduped) <= 8:
            return deduped
        window = self._raw_smoothing_window(len(deduped))
        if window <= 1:
            return deduped
        return self._smooth_point_sequence(deduped, window=window, closed=path_closed)

    def _raw_smoothing_window(self, point_count: int) -> int:
        if point_count < 64:
            return 1
        window = max(3, min(15, point_count // 24))
        if window % 2 == 0:
            window += 1
        return window

    def _smooth_point_sequence(
        self,
        points: tuple[Point, ...],
        *,
        window: int,
        closed: bool,
    ) -> tuple[Point, ...]:
        if window <= 1 or len(points) <= 2:
            return points
        radius = window // 2
        smoothed: list[Point] = []
        last_index = len(points) - 1
        for index in range(len(points)):
            if closed:
                samples = [points[(index + offset) % len(points)] for offset in range(-radius, radius + 1)]
            else:
                samples = [points[min(max(index + offset, 0), last_index)] for offset in range(-radius, radius + 1)]
            smoothed.append(
                (
                    sum(sample[0] for sample in samples) / len(samples),
                    sum(sample[1] for sample in samples) / len(samples),
                )
            )
        return tuple(smoothed)

    def _is_tiny_path(self, segments: tuple[Segment, ...]) -> bool:
        points = self._sample_segment_range(segments)
        if len(points) < 2:
            return True
        bbox = self._bbox(points)
        polyline_length = _polyline_length(points)
        if bbox["diagonal"] >= self.config.min_path_diagonal or polyline_length >= self.config.min_line_length:
            return False
        return bbox["width"] < self.config.min_bbox_extent and bbox["height"] < self.config.min_bbox_extent

    def _bbox(self, points: tuple[Point, ...]) -> dict[str, float]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        width = max_x - min_x
        height = max_y - min_y
        return {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "width": width,
            "height": height,
            "diagonal": math.hypot(width, height),
        }

    def _aspect_ratio(self, bbox: dict[str, float]) -> float:
        width = bbox["width"]
        height = bbox["height"]
        if PrecisionUtility.near_zero(width) or PrecisionUtility.near_zero(height):
            return math.inf
        return max(width, height) / min(width, height)

    def _simplify_closed_polygon(self, points: tuple[Point, ...], diagonal: float) -> tuple[Point, ...]:
        ring = self._dedupe_points(points)
        if len(ring) < 4:
            return ()
        if not PrecisionUtility.points_close(ring[0], ring[-1]):
            ring = ring + (ring[0],)
        epsilon = max(self.config.rectangle_min_rdp_epsilon, diagonal * self.config.rectangle_rdp_epsilon_ratio)
        simplified = self._rdp(ring, epsilon)
        deduped = self._dedupe_points(simplified)
        if len(deduped) > 1 and PrecisionUtility.points_close(deduped[0], deduped[-1]):
            deduped = deduped[:-1]
        return deduped if len(deduped) == 4 else ()

    def _rdp(self, points: tuple[Point, ...], epsilon: float) -> tuple[Point, ...]:
        if len(points) <= 2:
            return points
        first = points[0]
        last = points[-1]
        best_distance = -1.0
        best_index = -1
        for index in range(1, len(points) - 1):
            distance = self._point_to_line_distance(points[index], first, last)
            if distance > best_distance:
                best_distance = distance
                best_index = index
        if best_distance <= epsilon or best_index < 0:
            return (first, last)
        left = self._rdp(points[: best_index + 1], epsilon)
        right = self._rdp(points[best_index:], epsilon)
        return left[:-1] + right

    def _point_to_line_distance(self, point: Point, start: Point, end: Point) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if PrecisionUtility.near_zero(dx) and PrecisionUtility.near_zero(dy):
            return PrecisionUtility.distance_between_points(point, start)
        numerator = abs((dy * point[0]) - (dx * point[1]) + (end[0] * start[1]) - (end[1] * start[0]))
        denominator = math.hypot(dx, dy)
        return numerator / denominator

    def _edge(self, start: Point, end: Point) -> dict[str, object]:
        vector = (end[0] - start[0], end[1] - start[1])
        return {"start": start, "end": end, "vector": vector, "length": math.hypot(vector[0], vector[1])}

    def _angle_between(self, left: Point, right: Point) -> float:
        left_vector = PrecisionUtility.normalize_vector(left)
        right_vector = PrecisionUtility.normalize_vector(right)
        if left_vector is None or right_vector is None:
            return 0.0
        dot = max(-1.0, min(1.0, (left_vector[0] * right_vector[0]) + (left_vector[1] * right_vector[1])))
        return math.acos(abs(dot))

    def _parallel_error(self, left: Point, right: Point) -> float:
        left_vector = PrecisionUtility.normalize_vector(left)
        right_vector = PrecisionUtility.normalize_vector(right)
        if left_vector is None or right_vector is None:
            return math.pi
        dot = max(-1.0, min(1.0, abs((left_vector[0] * right_vector[0]) + (left_vector[1] * right_vector[1]))))
        return math.acos(dot)

    def _length_pair_score(self, first: float, second: float) -> float:
        largest = max(first, second)
        if PrecisionUtility.near_zero(largest):
            return 0.0
        ratio = abs(first - second) / largest
        return max(0.0, 1.0 - (ratio / self.config.rectangle_edge_ratio_tolerance))

    def _base_evidence(
        self,
        *,
        source: str,
        raw_point_count: int,
        fit_point_count: int,
        segment_count: int,
        bbox: dict[str, float],
        fit_error: float,
        inlier_ratio: float,
        model_complexity_delta: int,
        aspect_ratio: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "point_source": source,
            "raw_point_count": raw_point_count,
            "fit_point_count": fit_point_count,
            "segment_count": segment_count,
            "bbox": {
                "min_x": bbox["min_x"],
                "min_y": bbox["min_y"],
                "max_x": bbox["max_x"],
                "max_y": bbox["max_y"],
                "width": bbox["width"],
                "height": bbox["height"],
                "diagonal": bbox["diagonal"],
            },
            "aspect_ratio": aspect_ratio,
            "fit_error": fit_error,
            "inlier_ratio": inlier_ratio,
            "model_complexity_delta": model_complexity_delta,
        }
        if extra:
            payload.update(extra)
        return payload

    def _segment_start_point(self, segment: Segment) -> Point | None:
        if "start" in segment.params:
            return self._coerce_point(segment.params["start"])
        points = segment.params.get("points")
        if isinstance(points, list) and points:
            return self._coerce_point(points[0])
        return None

    def _segment_end_point(self, segment: Segment) -> Point | None:
        if "end" in segment.params:
            return self._coerce_point(segment.params["end"])
        points = segment.params.get("points")
        if isinstance(points, list) and points:
            return self._coerce_point(points[-1])
        return None

    def _nearest_point_index(self, points: tuple[Point, ...], target: Point) -> int | None:
        if not points:
            return None
        return min(range(len(points)), key=lambda index: PrecisionUtility.distance_between_points(points[index], target))

    def _closed_point_slice(self, points: tuple[Point, ...], start_index: int, end_index: int) -> tuple[Point, ...]:
        if start_index <= end_index:
            return points[start_index : end_index + 1]
        return points[start_index:] + points[: end_index + 1]

    def _coerce_points(self, values: list[object]) -> tuple[Point, ...]:
        points: list[Point] = []
        for value in values:
            try:
                points.append(self._coerce_point(value))
            except (TypeError, ValueError):
                return ()
        return tuple(points)

    def _coerce_point(self, value: object) -> Point:
        x, y = value  # type: ignore[misc]
        return (float(x), float(y))

    def _dedupe_points(self, points: tuple[Point, ...]) -> tuple[Point, ...]:
        if not points:
            return ()
        deduped = [points[0]]
        for point in points[1:]:
            if PrecisionUtility.points_close(deduped[-1], point):
                continue
            deduped.append(point)
        return tuple(deduped)

    def _dedupe_candidates(self, candidates: list[ShapeCandidate]) -> tuple[ShapeCandidate, ...]:
        best_by_key: dict[tuple[str, str, tuple[int, int]], ShapeCandidate] = {}
        for candidate in candidates:
            key = (candidate.path_id, candidate.target_type, candidate.segment_range)
            existing = best_by_key.get(key)
            if existing is None or candidate.confidence > existing.confidence:
                best_by_key[key] = candidate
        return tuple(best_by_key.values())

    def _higher_is_better(self, value: float, *, minimum: float, target: float) -> float:
        if target <= minimum:
            return 1.0 if value >= target else 0.0
        return max(0.0, min(1.0, (value - minimum) / (target - minimum)))

    def _lower_is_better(self, value: float, *, target: float, maximum: float) -> float:
        if value <= target:
            return 1.0
        if value >= maximum:
            return 0.0
        return max(0.0, min(1.0, (maximum - value) / (maximum - target)))


def _polyline_length(points: tuple[Point, ...]) -> float:
    return sum(
        PrecisionUtility.distance_between_points(points[index], points[index + 1])
        for index in range(len(points) - 1)
    )


def _arc_angle_coverage(params: dict[str, object]) -> float:
    start = float(params["start_angle"])
    end = float(params["end_angle"])
    direction = str(params["direction"]).lower()
    if direction == "cw":
        sweep = start - end
        if sweep <= 0.0:
            sweep += math.tau
        return sweep
    sweep = end - start
    if sweep <= 0.0:
        sweep += math.tau
    return sweep


__all__ = ["ShapeCandidateDetector", "ShapeCandidateDetectorConfig"]
