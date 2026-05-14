from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from core.precision import PrecisionUtility
from core.types import Point
from services.breakpoint_optimizer import (
    BreakPointOptimizer,
    BreakPointOptimizerConfig,
    BreakPointRequest,
    BreakPointResult,
)
from services.ellipse_fitter import (
    PreciseEllipseFitter,
    PreciseEllipseResult,
    RansacEllipseConfig,
    RansacEllipseFitter,
    RansacEllipseResult,
)
from services.fitting_confidence import (
    FittingConfidenceConfig,
    FittingConfidenceInputs,
    FittingConfidenceMetric,
    FittingConfidenceResult,
)
from services.refinement_feedback import (
    RefinementFeedback,
    RefinementFeedbackConfig,
    RefinementFeedbackInputs,
    RefinementFeedbackResult,
)


@dataclass(frozen=True, slots=True)
class RansacLineConfig:
    iterations: int = 64
    inlier_threshold: float = 0.2
    min_inlier_ratio: float = 0.5
    random_seed: int = 0


@dataclass(frozen=True, slots=True)
class RansacLineResult:
    params: dict[str, object]
    inlier_ratio: float
    fit_error: float
    inlier_indexes: tuple[int, ...]
    outlier_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RansacCircleConfig:
    iterations: int = 96
    inlier_threshold: float = 0.25
    min_inlier_ratio: float = 0.5
    random_seed: int = 0


@dataclass(frozen=True, slots=True)
class RansacCircleResult:
    params: dict[str, float]
    inlier_ratio: float
    fit_error: float
    inlier_indexes: tuple[int, ...]
    outlier_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RansacArcConfig:
    iterations: int = 96
    inlier_threshold: float = 0.25
    min_inlier_ratio: float = 0.5
    random_seed: int = 0
    min_arc_angle: float = math.pi / 12.0
    max_radial_error: float = 0.25


@dataclass(frozen=True, slots=True)
class RansacArcResult:
    params: dict[str, object]
    inlier_ratio: float
    fit_error: float
    inlier_indexes: tuple[int, ...]
    outlier_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PreciseLineResult:
    params: dict[str, object]
    mse: float
    rmse: float
    parameter_delta: dict[str, object]


@dataclass(frozen=True, slots=True)
class PreciseCircleResult:
    params: dict[str, float]
    mse: float
    rmse: float
    parameter_delta: dict[str, object]


@dataclass(frozen=True, slots=True)
class PreciseArcResult:
    params: dict[str, object]
    mse: float
    rmse: float
    parameter_delta: dict[str, object]


@dataclass(frozen=True, slots=True)
class RefinementEngineConfig:
    breakpoint_optimizer_config: BreakPointOptimizerConfig = field(default_factory=BreakPointOptimizerConfig)
    fitting_confidence_config: FittingConfidenceConfig = field(default_factory=FittingConfidenceConfig)
    refinement_feedback_config: RefinementFeedbackConfig = field(default_factory=RefinementFeedbackConfig)
    line_ransac_config: RansacLineConfig = field(default_factory=RansacLineConfig)
    circle_ransac_config: RansacCircleConfig = field(default_factory=RansacCircleConfig)
    arc_ransac_config: RansacArcConfig = field(default_factory=RansacArcConfig)


@dataclass(frozen=True, slots=True)
class RefinementRequest:
    points: tuple[Point, ...]
    rough_range: tuple[int, int]
    target_type: str
    residuals: tuple[float, ...] = ()
    tangent_vectors: tuple[tuple[float, float] | None, ...] = ()
    user_breakpoints: tuple[int, ...] = ()
    adjacent_endpoints: tuple[int, ...] = ()
    ai_marked_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple((float(x), float(y)) for x, y in self.points))
        object.__setattr__(self, "rough_range", (int(self.rough_range[0]), int(self.rough_range[1])))
        object.__setattr__(self, "target_type", str(self.target_type).lower())
        object.__setattr__(self, "residuals", tuple(float(value) for value in self.residuals))
        object.__setattr__(
            self,
            "tangent_vectors",
            tuple(None if vector is None else (float(vector[0]), float(vector[1])) for vector in self.tangent_vectors),
        )
        object.__setattr__(self, "user_breakpoints", tuple(int(index) for index in self.user_breakpoints))
        object.__setattr__(self, "adjacent_endpoints", tuple(int(index) for index in self.adjacent_endpoints))
        if self.ai_marked_range is not None:
            object.__setattr__(self, "ai_marked_range", (int(self.ai_marked_range[0]), int(self.ai_marked_range[1])))


@dataclass(frozen=True, slots=True)
class RefinementResult:
    target_type: str
    optimized_range: tuple[int, int]
    breakpoint_result: BreakPointResult
    params: dict[str, object] | None
    inlier_ratio: float
    fit_error: float
    confidence_result: FittingConfidenceResult
    feedback: RefinementFeedbackResult
    inlier_indexes: tuple[int, ...]
    outlier_indexes: tuple[int, ...]
    failure_message: str | None = None


class RefinementEngine:
    def __init__(
        self,
        config: RefinementEngineConfig | None = None,
        *,
        breakpoint_optimizer: BreakPointOptimizer | None = None,
        fitting_confidence_metric: FittingConfidenceMetric | None = None,
        refinement_feedback: RefinementFeedback | None = None,
    ) -> None:
        self.config = config or RefinementEngineConfig()
        self.breakpoint_optimizer = breakpoint_optimizer or BreakPointOptimizer(self.config.breakpoint_optimizer_config)
        self.fitting_confidence_metric = fitting_confidence_metric or FittingConfidenceMetric(
            self.config.fitting_confidence_config
        )
        self.refinement_feedback = refinement_feedback or RefinementFeedback(self.config.refinement_feedback_config)
        self.line_ransac_fitter = RansacLineFitter(self.config.line_ransac_config)
        self.circle_ransac_fitter = RansacCircleFitter(self.config.circle_ransac_config)
        self.arc_ransac_fitter = RansacArcFitter(self.config.arc_ransac_config)
        self.line_precise_fitter = PreciseLineFitter()
        self.circle_precise_fitter = PreciseCircleFitter()
        self.arc_precise_fitter = PreciseArcFitter()

    def refine(self, request: RefinementRequest) -> RefinementResult:
        if request.target_type == "line":
            return self.refine_line(request)
        if request.target_type == "circle":
            return self.refine_circle(request)
        if request.target_type == "arc":
            return self.refine_arc(request)
        raise ValueError(f"unsupported refinement target type: {request.target_type}")

    def refine_line(self, request: RefinementRequest) -> RefinementResult:
        return self._refine(request, target_type="line")

    def refine_circle(self, request: RefinementRequest) -> RefinementResult:
        return self._refine(request, target_type="circle")

    def refine_arc(self, request: RefinementRequest) -> RefinementResult:
        return self._refine(request, target_type="arc")

    def _refine(self, request: RefinementRequest, *, target_type: str) -> RefinementResult:
        breakpoint_result = self.breakpoint_optimizer.optimize(
            BreakPointRequest(
                points=request.points,
                rough_range=request.rough_range,
                target_type=target_type,
                residuals=request.residuals,
                tangent_vectors=request.tangent_vectors,
                user_breakpoints=request.user_breakpoints,
                adjacent_endpoints=request.adjacent_endpoints,
                ai_marked_range=request.ai_marked_range,
            )
        )
        selected_points, global_indexes = self._slice_points(request.points, breakpoint_result.optimized_range)

        try:
            if target_type == "line":
                ransac_result = self.line_ransac_fitter.fit(selected_points)
                inlier_points = tuple(selected_points[index] for index in ransac_result.inlier_indexes)
                precise_result = self.line_precise_fitter.fit(inlier_points, ransac_result.params)
                confidence_inputs = self._line_confidence_inputs(ransac_result, precise_result, inlier_points)
            elif target_type == "circle":
                ransac_result = self.circle_ransac_fitter.fit(selected_points)
                inlier_points = tuple(selected_points[index] for index in ransac_result.inlier_indexes)
                precise_result = self.circle_precise_fitter.fit(inlier_points, ransac_result.params)
                confidence_inputs = self._circle_confidence_inputs(ransac_result, precise_result, inlier_points)
            elif target_type == "arc":
                ransac_result = self.arc_ransac_fitter.fit(selected_points)
                inlier_points = tuple(selected_points[index] for index in ransac_result.inlier_indexes)
                precise_result = self.arc_precise_fitter.fit(inlier_points, ransac_result.params)
                confidence_inputs = self._arc_confidence_inputs(ransac_result, precise_result, inlier_points)
            else:
                raise ValueError(f"unsupported refinement target type: {target_type}")
        except ValueError as exc:
            return self._failure_result(
                target_type=target_type,
                breakpoint_result=breakpoint_result,
                message=str(exc),
            )

        confidence_result = self.fitting_confidence_metric.evaluate(confidence_inputs)
        feedback = self.refinement_feedback.evaluate(
            RefinementFeedbackInputs(
                segment_type=target_type,
                inlier_ratio=ransac_result.inlier_ratio,
                fit_error=precise_result.rmse,
                confidence_result=confidence_result,
            )
        )
        return RefinementResult(
            target_type=target_type,
            optimized_range=breakpoint_result.optimized_range,
            breakpoint_result=breakpoint_result,
            params=precise_result.params,
            inlier_ratio=ransac_result.inlier_ratio,
            fit_error=precise_result.rmse,
            confidence_result=confidence_result,
            feedback=feedback,
            inlier_indexes=tuple(global_indexes[index] for index in ransac_result.inlier_indexes),
            outlier_indexes=tuple(global_indexes[index] for index in ransac_result.outlier_indexes),
            failure_message=None,
        )

    def _line_confidence_inputs(
        self,
        ransac_result: RansacLineResult,
        precise_result: PreciseLineResult,
        inlier_points: tuple[Point, ...],
    ) -> FittingConfidenceInputs:
        return FittingConfidenceInputs(
            segment_type="line",
            inlier_ratio=ransac_result.inlier_ratio,
            rmse=precise_result.rmse,
            segment_length=_polyline_length(inlier_points),
            parameter_delta=precise_result.parameter_delta,
        )

    def _circle_confidence_inputs(
        self,
        ransac_result: RansacCircleResult,
        precise_result: PreciseCircleResult,
        inlier_points: tuple[Point, ...],
    ) -> FittingConfidenceInputs:
        return FittingConfidenceInputs(
            segment_type="circle",
            inlier_ratio=ransac_result.inlier_ratio,
            rmse=precise_result.rmse,
            segment_length=_polyline_length(inlier_points),
            radial_error=precise_result.rmse,
            parameter_delta=precise_result.parameter_delta,
        )

    def _arc_confidence_inputs(
        self,
        ransac_result: RansacArcResult,
        precise_result: PreciseArcResult,
        inlier_points: tuple[Point, ...],
    ) -> FittingConfidenceInputs:
        return FittingConfidenceInputs(
            segment_type="arc",
            inlier_ratio=ransac_result.inlier_ratio,
            rmse=precise_result.rmse,
            segment_length=_polyline_length(inlier_points),
            radial_error=precise_result.rmse,
            arc_angle_coverage=_arc_angle_coverage(precise_result.params),
            parameter_delta=precise_result.parameter_delta,
        )

    def _slice_points(
        self,
        points: tuple[Point, ...],
        optimized_range: tuple[int, int],
    ) -> tuple[tuple[Point, ...], tuple[int, ...]]:
        start, end = optimized_range
        selected_points = points[start : end + 1]
        global_indexes = tuple(range(start, end + 1))
        return (selected_points, global_indexes)

    def _failure_result(
        self,
        *,
        target_type: str,
        breakpoint_result: BreakPointResult,
        message: str,
    ) -> RefinementResult:
        if "inlier ratio" in message:
            reason = "low_inlier_ratio"
            inlier_ratio = 0.0
            fit_error = 0.0
        else:
            reason = "high_fit_error"
            inlier_ratio = 1.0
            fit_error = self.refinement_feedback.config.max_fit_error + 1.0

        confidence_result = FittingConfidenceResult(confidence=0.0, failure_reason=reason)
        feedback = self.refinement_feedback.evaluate(
            RefinementFeedbackInputs(
                segment_type=target_type,
                inlier_ratio=inlier_ratio,
                fit_error=fit_error,
                confidence_result=confidence_result,
            )
        )
        return RefinementResult(
            target_type=target_type,
            optimized_range=breakpoint_result.optimized_range,
            breakpoint_result=breakpoint_result,
            params=None,
            inlier_ratio=inlier_ratio,
            fit_error=fit_error,
            confidence_result=confidence_result,
            feedback=feedback,
            inlier_indexes=(),
            outlier_indexes=(),
            failure_message=message,
        )


class RansacLineFitter:
    def __init__(self, config: RansacLineConfig | None = None) -> None:
        self.config = config or RansacLineConfig()

    def fit(self, points: tuple[Point, ...] | list[Point]) -> RansacLineResult:
        point_sequence = tuple((float(x), float(y)) for x, y in points)
        if len(point_sequence) < 2:
            raise ValueError("at least two Vector Space points are required")

        rng = random.Random(self.config.random_seed)
        best: tuple[tuple[int, ...], tuple[float, float, float], float] | None = None

        for first_index, second_index in self._sample_pairs(len(point_sequence), rng):
            line = self._line_from_points(point_sequence[first_index], point_sequence[second_index])
            if line is None:
                continue

            inlier_indexes = tuple(
                index
                for index, point in enumerate(point_sequence)
                if self._point_line_distance(point, line) <= self.config.inlier_threshold
            )
            if len(inlier_indexes) < 2:
                continue

            fit_error = self._fit_error(point_sequence, inlier_indexes, line)
            candidate = (inlier_indexes, line, fit_error)
            if best is None or self._is_better_candidate(candidate, best, len(point_sequence)):
                best = candidate

        if best is None:
            raise ValueError("unable to fit a robust line from the provided points")

        inlier_indexes, line, fit_error = best
        inlier_ratio = len(inlier_indexes) / len(point_sequence)
        if inlier_ratio < self.config.min_inlier_ratio:
            raise ValueError("insufficient inlier ratio for robust line fit")

        params = self._segment_params(point_sequence, inlier_indexes, line)
        outlier_indexes = tuple(index for index in range(len(point_sequence)) if index not in set(inlier_indexes))

        return RansacLineResult(
            params=params,
            inlier_ratio=inlier_ratio,
            fit_error=fit_error,
            inlier_indexes=inlier_indexes,
            outlier_indexes=outlier_indexes,
        )

    def _sample_pairs(self, point_count: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
        if point_count == 2:
            return ((0, 1),)

        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        max_unique_pairs = point_count * (point_count - 1) // 2
        target_iterations = min(self.config.iterations, max_unique_pairs)

        while len(pairs) < target_iterations:
            first_index, second_index = sorted(rng.sample(range(point_count), 2))
            pair = (first_index, second_index)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)

        return tuple(pairs)

    def _line_from_points(self, first: Point, second: Point) -> tuple[float, float, float] | None:
        if PrecisionUtility.points_close(first, second):
            return None
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = math.hypot(dx, dy)
        if PrecisionUtility.near_zero(length):
            return None
        a = dy / length
        b = -dx / length
        c = -((a * first[0]) + (b * first[1]))
        return (a, b, c)

    def _point_line_distance(self, point: Point, line: tuple[float, float, float]) -> float:
        a, b, c = line
        return abs((a * point[0]) + (b * point[1]) + c)

    def _fit_error(
        self,
        points: tuple[Point, ...],
        inlier_indexes: tuple[int, ...],
        line: tuple[float, float, float],
    ) -> float:
        distances = [self._point_line_distance(points[index], line) for index in inlier_indexes]
        if not distances:
            return math.inf
        return sum(distances) / len(distances)

    def _is_better_candidate(
        self,
        candidate: tuple[tuple[int, ...], tuple[float, float, float], float],
        current_best: tuple[tuple[int, ...], tuple[float, float, float], float],
        point_count: int,
    ) -> bool:
        candidate_inliers, _, candidate_error = candidate
        best_inliers, _, best_error = current_best
        if len(candidate_inliers) != len(best_inliers):
            return len(candidate_inliers) > len(best_inliers)
        if not PrecisionUtility.almost_equal(candidate_error, best_error):
            return candidate_error < best_error
        return (len(candidate_inliers) / point_count) > (len(best_inliers) / point_count)

    def _segment_params(
        self,
        points: tuple[Point, ...],
        inlier_indexes: tuple[int, ...],
        line: tuple[float, float, float],
    ) -> dict[str, object]:
        inlier_points = tuple(points[index] for index in inlier_indexes)
        direction = self._principal_direction(inlier_points)
        origin = inlier_points[0]
        projected = tuple(
            ((point[0] - origin[0]) * direction[0]) + ((point[1] - origin[1]) * direction[1])
            for point in inlier_points
        )
        start_offset = min(projected)
        end_offset = max(projected)
        start = (origin[0] + (direction[0] * start_offset), origin[1] + (direction[1] * start_offset))
        end = (origin[0] + (direction[0] * end_offset), origin[1] + (direction[1] * end_offset))
        a, b, c = line

        return {
            "start": [start[0], start[1]],
            "end": [end[0], end[1]],
            "direction": [direction[0], direction[1]],
            "line": {"a": a, "b": b, "c": c},
        }

    def _principal_direction(self, points: tuple[Point, ...]) -> tuple[float, float]:
        if len(points) == 2:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            direction = PrecisionUtility.normalize_vector((dx, dy))
            if direction is None:
                raise ValueError("degenerate inlier set")
            return direction

        mean_x = sum(point[0] for point in points) / len(points)
        mean_y = sum(point[1] for point in points) / len(points)
        sxx = sum((point[0] - mean_x) ** 2 for point in points)
        syy = sum((point[1] - mean_y) ** 2 for point in points)
        sxy = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points)
        angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
        direction = PrecisionUtility.normalize_vector((math.cos(angle), math.sin(angle)))
        if direction is None:
            raise ValueError("degenerate inlier set")
        return direction


class RansacCircleFitter:
    def __init__(self, config: RansacCircleConfig | None = None) -> None:
        self.config = config or RansacCircleConfig()

    def fit(self, points: tuple[Point, ...] | list[Point]) -> RansacCircleResult:
        point_sequence = tuple((float(x), float(y)) for x, y in points)
        if len(point_sequence) < 3:
            raise ValueError("at least three Vector Space points are required")

        rng = random.Random(self.config.random_seed)
        best: tuple[tuple[int, ...], tuple[float, float, float], float] | None = None

        for first_index, second_index, third_index in self._sample_triplets(len(point_sequence), rng):
            circle = self._circle_from_points(
                point_sequence[first_index],
                point_sequence[second_index],
                point_sequence[third_index],
            )
            if circle is None:
                continue

            inlier_indexes = tuple(
                index
                for index, point in enumerate(point_sequence)
                if self._radial_error(point, circle) <= self.config.inlier_threshold
            )
            if len(inlier_indexes) < 3:
                continue

            fit_error = self._fit_error(point_sequence, inlier_indexes, circle)
            candidate = (inlier_indexes, circle, fit_error)
            if best is None or self._is_better_candidate(candidate, best, len(point_sequence)):
                best = candidate

        if best is None:
            raise ValueError("unable to fit a robust circle from the provided points")

        inlier_indexes, circle, fit_error = best
        inlier_ratio = len(inlier_indexes) / len(point_sequence)
        if inlier_ratio < self.config.min_inlier_ratio:
            raise ValueError("insufficient inlier ratio for robust circle fit")

        outlier_index_set = set(range(len(point_sequence))) - set(inlier_indexes)
        cx, cy, radius = circle
        return RansacCircleResult(
            params={"cx": cx, "cy": cy, "r": radius},
            inlier_ratio=inlier_ratio,
            fit_error=fit_error,
            inlier_indexes=inlier_indexes,
            outlier_indexes=tuple(sorted(outlier_index_set)),
        )

    def _sample_triplets(self, point_count: int, rng: random.Random) -> tuple[tuple[int, int, int], ...]:
        if point_count == 3:
            return ((0, 1, 2),)

        triplets: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        max_unique_triplets = math.comb(point_count, 3)
        target_iterations = min(self.config.iterations, max_unique_triplets)

        while len(triplets) < target_iterations:
            triplet = tuple(sorted(rng.sample(range(point_count), 3)))
            if triplet in seen:
                continue
            seen.add(triplet)
            triplets.append(triplet)

        return tuple(triplets)

    def _circle_from_points(self, first: Point, second: Point, third: Point) -> tuple[float, float, float] | None:
        x1, y1 = first
        x2, y2 = second
        x3, y3 = third

        determinant = 2.0 * (
            (x1 * (y2 - y3))
            + (x2 * (y3 - y1))
            + (x3 * (y1 - y2))
        )
        if PrecisionUtility.near_zero(determinant):
            return None

        x1_sq_y1_sq = (x1 * x1) + (y1 * y1)
        x2_sq_y2_sq = (x2 * x2) + (y2 * y2)
        x3_sq_y3_sq = (x3 * x3) + (y3 * y3)
        cx = (
            (x1_sq_y1_sq * (y2 - y3))
            + (x2_sq_y2_sq * (y3 - y1))
            + (x3_sq_y3_sq * (y1 - y2))
        ) / determinant
        cy = (
            (x1_sq_y1_sq * (x3 - x2))
            + (x2_sq_y2_sq * (x1 - x3))
            + (x3_sq_y3_sq * (x2 - x1))
        ) / determinant
        radius = math.hypot(x1 - cx, y1 - cy)
        if PrecisionUtility.near_zero(radius):
            return None

        return (cx, cy, radius)

    def _radial_error(self, point: Point, circle: tuple[float, float, float]) -> float:
        cx, cy, radius = circle
        return abs(math.hypot(point[0] - cx, point[1] - cy) - radius)

    def _fit_error(
        self,
        points: tuple[Point, ...],
        inlier_indexes: tuple[int, ...],
        circle: tuple[float, float, float],
    ) -> float:
        errors = [self._radial_error(points[index], circle) for index in inlier_indexes]
        if not errors:
            return math.inf
        return sum(errors) / len(errors)

    def _is_better_candidate(
        self,
        candidate: tuple[tuple[int, ...], tuple[float, float, float], float],
        current_best: tuple[tuple[int, ...], tuple[float, float, float], float],
        point_count: int,
    ) -> bool:
        candidate_inliers, _, candidate_error = candidate
        best_inliers, _, best_error = current_best
        if len(candidate_inliers) != len(best_inliers):
            return len(candidate_inliers) > len(best_inliers)
        if not PrecisionUtility.almost_equal(candidate_error, best_error):
            return candidate_error < best_error
        return (len(candidate_inliers) / point_count) > (len(best_inliers) / point_count)


class RansacArcFitter:
    def __init__(self, config: RansacArcConfig | None = None) -> None:
        self.config = config or RansacArcConfig()
        self._circle_fitter = RansacCircleFitter(
            RansacCircleConfig(
                iterations=self.config.iterations,
                inlier_threshold=self.config.inlier_threshold,
                min_inlier_ratio=self.config.min_inlier_ratio,
                random_seed=self.config.random_seed,
            )
        )

    def fit(self, points: tuple[Point, ...] | list[Point]) -> RansacArcResult:
        point_sequence = tuple((float(x), float(y)) for x, y in points)
        if len(point_sequence) < 3:
            raise ValueError("at least three Vector Space points are required")

        circle_result = self._circle_fitter.fit(point_sequence)
        circle = (
            float(circle_result.params["cx"]),
            float(circle_result.params["cy"]),
            float(circle_result.params["r"]),
        )
        radial_errors = tuple(
            self._circle_fitter._radial_error(point_sequence[index], circle)
            for index in circle_result.inlier_indexes
        )
        if radial_errors and max(radial_errors) > self.config.max_radial_error:
            raise ValueError("arc radial fit error exceeds maximum radial error")

        cx, cy, radius = circle
        inlier_points = tuple(point_sequence[index] for index in circle_result.inlier_indexes)
        start_angle, end_angle, direction, arc_span = self._arc_angles(inlier_points, (cx, cy))
        if arc_span < self.config.min_arc_angle:
            raise ValueError("arc span is below minimum arc angle")

        return RansacArcResult(
            params={
                "cx": cx,
                "cy": cy,
                "r": radius,
                "start_angle": start_angle,
                "end_angle": end_angle,
                "direction": direction,
                "start": [inlier_points[0][0], inlier_points[0][1]],
                "end": [inlier_points[-1][0], inlier_points[-1][1]],
            },
            inlier_ratio=circle_result.inlier_ratio,
            fit_error=circle_result.fit_error,
            inlier_indexes=circle_result.inlier_indexes,
            outlier_indexes=circle_result.outlier_indexes,
        )

    def _arc_angles(
        self,
        points: tuple[Point, ...],
        center: tuple[float, float],
    ) -> tuple[float, float, str, float]:
        raw_angles = tuple(math.atan2(point[1] - center[1], point[0] - center[0]) for point in points)
        unwrapped = [raw_angles[0]]

        for angle in raw_angles[1:]:
            delta = (angle - raw_angles[len(unwrapped) - 1] + math.pi) % (2.0 * math.pi) - math.pi
            unwrapped.append(unwrapped[-1] + delta)

        net_delta = unwrapped[-1] - unwrapped[0]
        direction = "ccw" if net_delta >= 0.0 else "cw"
        return (
            self._normalize_angle(unwrapped[0]),
            self._normalize_angle(unwrapped[-1]),
            direction,
            abs(net_delta),
        )

    def _normalize_angle(self, angle: float) -> float:
        normalized = angle % (2.0 * math.pi)
        if PrecisionUtility.almost_equal(normalized, 2.0 * math.pi):
            return 0.0
        return normalized


class PreciseLineFitter:
    def fit(self, points: tuple[Point, ...] | list[Point], initial_params: dict[str, object]) -> PreciseLineResult:
        point_sequence = tuple((float(x), float(y)) for x, y in points)
        if len(point_sequence) < 2:
            raise ValueError("at least two Vector Space points are required")

        direction = self._principal_direction(point_sequence)
        initial_direction = _coerce_vector(initial_params["direction"])
        if _dot(direction, initial_direction) < 0.0:
            direction = (-direction[0], -direction[1])

        centroid = (
            sum(point[0] for point in point_sequence) / len(point_sequence),
            sum(point[1] for point in point_sequence) / len(point_sequence),
        )
        projected = tuple(
            ((point[0] - centroid[0]) * direction[0]) + ((point[1] - centroid[1]) * direction[1])
            for point in point_sequence
        )
        start_offset = min(projected)
        end_offset = max(projected)
        start = (centroid[0] + (direction[0] * start_offset), centroid[1] + (direction[1] * start_offset))
        end = (centroid[0] + (direction[0] * end_offset), centroid[1] + (direction[1] * end_offset))
        line = _line_from_segment(start, end)
        if line is None:
            raise ValueError("degenerate inlier set")

        distances = tuple(_point_line_distance(point, line) for point in point_sequence)
        mse = sum(distance * distance for distance in distances) / len(distances)
        rmse = math.sqrt(mse)
        a, b, c = line
        return PreciseLineResult(
            params={
                "start": [start[0], start[1]],
                "end": [end[0], end[1]],
                "direction": [direction[0], direction[1]],
                "line": {"a": a, "b": b, "c": c},
            },
            mse=mse,
            rmse=rmse,
            parameter_delta={
                "start_distance": PrecisionUtility.distance_between_points(start, _coerce_point(initial_params["start"])),
                "end_distance": PrecisionUtility.distance_between_points(end, _coerce_point(initial_params["end"])),
                "direction_angle": _angle_delta(direction, initial_direction),
                "line_offset": abs(c - float(_coerce_line(initial_params["line"])[2])),
            },
        )

    def _principal_direction(self, points: tuple[Point, ...]) -> tuple[float, float]:
        if len(points) == 2:
            direction = PrecisionUtility.normalize_vector((points[1][0] - points[0][0], points[1][1] - points[0][1]))
            if direction is None:
                raise ValueError("degenerate inlier set")
            return direction

        mean_x = sum(point[0] for point in points) / len(points)
        mean_y = sum(point[1] for point in points) / len(points)
        sxx = sum((point[0] - mean_x) ** 2 for point in points)
        syy = sum((point[1] - mean_y) ** 2 for point in points)
        sxy = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points)
        direction = PrecisionUtility.normalize_vector((math.cos(0.5 * math.atan2(2.0 * sxy, sxx - syy)), math.sin(0.5 * math.atan2(2.0 * sxy, sxx - syy))))
        if direction is None:
            raise ValueError("degenerate inlier set")
        return direction


class PreciseCircleFitter:
    def fit(self, points: tuple[Point, ...] | list[Point], initial_params: dict[str, float]) -> PreciseCircleResult:
        point_sequence = tuple((float(x), float(y)) for x, y in points)
        if len(point_sequence) < 3:
            raise ValueError("at least three Vector Space points are required")

        circle = self._least_squares_circle(point_sequence)
        cx, cy, radius = circle
        errors = tuple(self._radial_error(point, circle) for point in point_sequence)
        mse = sum(error * error for error in errors) / len(errors)
        rmse = math.sqrt(mse)
        initial_center = (float(initial_params["cx"]), float(initial_params["cy"]))
        return PreciseCircleResult(
            params={"cx": cx, "cy": cy, "r": radius},
            mse=mse,
            rmse=rmse,
            parameter_delta={
                "center_distance": PrecisionUtility.distance_between_points((cx, cy), initial_center),
                "radius_delta": abs(radius - float(initial_params["r"])),
            },
        )

    def _least_squares_circle(self, points: tuple[Point, ...]) -> tuple[float, float, float]:
        sum_x = sum(point[0] for point in points)
        sum_y = sum(point[1] for point in points)
        sum_xx = sum(point[0] * point[0] for point in points)
        sum_yy = sum(point[1] * point[1] for point in points)
        sum_xy = sum(point[0] * point[1] for point in points)
        sum_z = sum((point[0] * point[0]) + (point[1] * point[1]) for point in points)
        sum_xz = sum(point[0] * ((point[0] * point[0]) + (point[1] * point[1])) for point in points)
        sum_yz = sum(point[1] * ((point[0] * point[0]) + (point[1] * point[1])) for point in points)

        solution = _solve_3x3(
            (
                (sum_xx, sum_xy, sum_x),
                (sum_xy, sum_yy, sum_y),
                (sum_x, sum_y, float(len(points))),
            ),
            (-sum_xz, -sum_yz, -sum_z),
        )
        a, b, c = solution
        cx = -a / 2.0
        cy = -b / 2.0
        radius_sq = (cx * cx) + (cy * cy) - c
        if radius_sq <= PrecisionUtility.EPSILON:
            raise ValueError("unable to fit a precise circle from the provided points")
        return (cx, cy, math.sqrt(radius_sq))

    def _radial_error(self, point: Point, circle: tuple[float, float, float]) -> float:
        cx, cy, radius = circle
        return abs(math.hypot(point[0] - cx, point[1] - cy) - radius)


class PreciseArcFitter:
    def __init__(self) -> None:
        self._circle_fitter = PreciseCircleFitter()

    def fit(self, points: tuple[Point, ...] | list[Point], initial_params: dict[str, object]) -> PreciseArcResult:
        point_sequence = tuple((float(x), float(y)) for x, y in points)
        if len(point_sequence) < 3:
            raise ValueError("at least three Vector Space points are required")

        circle_result = self._circle_fitter.fit(
            point_sequence,
            {
                "cx": float(initial_params["cx"]),
                "cy": float(initial_params["cy"]),
                "r": float(initial_params["r"]),
            },
        )
        cx = float(circle_result.params["cx"])
        cy = float(circle_result.params["cy"])
        radius = float(circle_result.params["r"])
        start_angle, end_angle, direction = self._arc_angles(point_sequence, (cx, cy))
        return PreciseArcResult(
            params={
                "cx": cx,
                "cy": cy,
                "r": radius,
                "start_angle": start_angle,
                "end_angle": end_angle,
                "direction": direction,
                "start": [point_sequence[0][0], point_sequence[0][1]],
                "end": [point_sequence[-1][0], point_sequence[-1][1]],
            },
            mse=circle_result.mse,
            rmse=circle_result.rmse,
            parameter_delta={
                "center_distance": circle_result.parameter_delta["center_distance"],
                "radius_delta": circle_result.parameter_delta["radius_delta"],
                "start_angle_delta": _wrapped_angle_delta(start_angle, float(initial_params["start_angle"])),
                "end_angle_delta": _wrapped_angle_delta(end_angle, float(initial_params["end_angle"])),
                "direction_changed": direction != str(initial_params["direction"]),
            },
        )

    def _arc_angles(self, points: tuple[Point, ...], center: tuple[float, float]) -> tuple[float, float, str]:
        raw_angles = tuple(math.atan2(point[1] - center[1], point[0] - center[0]) for point in points)
        unwrapped = [raw_angles[0]]

        for index, angle in enumerate(raw_angles[1:], start=1):
            delta = (angle - raw_angles[index - 1] + math.pi) % (2.0 * math.pi) - math.pi
            unwrapped.append(unwrapped[-1] + delta)

        net_delta = unwrapped[-1] - unwrapped[0]
        return (_normalize_angle(unwrapped[0]), _normalize_angle(unwrapped[-1]), "ccw" if net_delta >= 0.0 else "cw")


def _coerce_point(value: object) -> tuple[float, float]:
    x, y = value  # type: ignore[misc]
    return (float(x), float(y))


def _coerce_vector(value: object) -> tuple[float, float]:
    x, y = value  # type: ignore[misc]
    return (float(x), float(y))


def _coerce_line(value: object) -> tuple[float, float, float]:
    line = value  # type: ignore[assignment]
    return (float(line["a"]), float(line["b"]), float(line["c"]))


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return (left[0] * right[0]) + (left[1] * right[1])


def _angle_delta(left: tuple[float, float], right: tuple[float, float]) -> float:
    return abs(math.atan2((left[0] * right[1]) - (left[1] * right[0]), _dot(left, right)))


def _wrapped_angle_delta(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


def _normalize_angle(angle: float) -> float:
    normalized = angle % (2.0 * math.pi)
    if PrecisionUtility.almost_equal(normalized, 2.0 * math.pi):
        return 0.0
    return normalized


def _polyline_length(points: tuple[Point, ...]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(
        PrecisionUtility.distance_between_points(points[index - 1], points[index])
        for index in range(1, len(points))
    )


def _arc_angle_coverage(params: dict[str, object]) -> float:
    start_angle = float(params["start_angle"])
    end_angle = float(params["end_angle"])
    direction = str(params["direction"])
    if direction == "cw":
        return (start_angle - end_angle) % (2.0 * math.pi)
    return (end_angle - start_angle) % (2.0 * math.pi)


def _line_from_segment(start: Point, end: Point) -> tuple[float, float, float] | None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if PrecisionUtility.near_zero(length):
        return None
    a = dy / length
    b = -dx / length
    c = -((a * start[0]) + (b * start[1]))
    return (a, b, c)


def _point_line_distance(point: Point, line: tuple[float, float, float]) -> float:
    a, b, c = line
    return abs((a * point[0]) + (b * point[1]) + c)


def _solve_3x3(matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    augmented = [
        [float(matrix[row][0]), float(matrix[row][1]), float(matrix[row][2]), float(vector[row])]
        for row in range(3)
    ]

    for pivot_index in range(3):
        pivot_row = max(range(pivot_index, 3), key=lambda row: abs(augmented[row][pivot_index]))
        if PrecisionUtility.near_zero(augmented[pivot_row][pivot_index]):
            raise ValueError("unable to fit a precise circle from the provided points")
        if pivot_row != pivot_index:
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot = augmented[pivot_index][pivot_index]
        for column in range(pivot_index, 4):
            augmented[pivot_index][column] /= pivot

        for row in range(3):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            for column in range(pivot_index, 4):
                augmented[row][column] -= factor * augmented[pivot_index][column]

    return (augmented[0][3], augmented[1][3], augmented[2][3])


__all__ = [
    "RefinementEngine",
    "RefinementEngineConfig",
    "RefinementRequest",
    "RefinementResult",
    "PreciseArcFitter",
    "PreciseArcResult",
    "PreciseCircleFitter",
    "PreciseCircleResult",
    "PreciseEllipseFitter",
    "PreciseEllipseResult",
    "PreciseLineFitter",
    "PreciseLineResult",
    "RansacArcConfig",
    "RansacArcFitter",
    "RansacArcResult",
    "RansacCircleConfig",
    "RansacCircleFitter",
    "RansacCircleResult",
    "RansacEllipseConfig",
    "RansacEllipseFitter",
    "RansacEllipseResult",
    "RansacLineConfig",
    "RansacLineFitter",
    "RansacLineResult",
]
