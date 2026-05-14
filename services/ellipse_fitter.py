from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class PreciseEllipseResult:
    cx: float
    cy: float
    rx: float
    ry: float
    rotation: float
    fit_error: float
    inlier_count: int
    outlier_count: int
    inlier_ratio: float


@dataclass(frozen=True, slots=True)
class RansacEllipseConfig:
    max_iterations: int = 500
    sample_size: int = 5
    max_error: float = 1.0
    min_inlier_ratio: float = 0.7
    random_seed: int | None = None
    min_axis_ratio_delta: float = 0.05
    duplicate_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.sample_size < 5:
            raise ValueError("sample_size must be at least 5")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_error <= 0.0:
            raise ValueError("max_error must be positive")
        if not (0.0 < self.min_inlier_ratio <= 1.0):
            raise ValueError("min_inlier_ratio must be within (0, 1]")
        if self.min_axis_ratio_delta < 0.0:
            raise ValueError("min_axis_ratio_delta must be non-negative")
        if self.duplicate_epsilon <= 0.0:
            raise ValueError("duplicate_epsilon must be positive")


@dataclass(frozen=True, slots=True)
class RansacEllipseResult:
    cx: float
    cy: float
    rx: float
    ry: float
    rotation: float
    fit_error: float
    inlier_count: int
    outlier_count: int
    inlier_ratio: float
    inlier_indexes: tuple[int, ...]
    outlier_indexes: tuple[int, ...]


class PreciseEllipseFitter:
    def __init__(self, min_axis_ratio_delta: float = 0.05, duplicate_epsilon: float = 1e-9) -> None:
        self.min_axis_ratio_delta = float(min_axis_ratio_delta)
        self.duplicate_epsilon = float(duplicate_epsilon)

    def fit(self, points: tuple[Point, ...] | list[Point]) -> PreciseEllipseResult:
        point_sequence = _coerce_points(points)
        _validate_point_set(point_sequence, duplicate_epsilon=self.duplicate_epsilon)

        conic = _fit_conic(point_sequence)
        cx, cy, rx, ry, rotation = _conic_to_geometric(
            conic,
            min_axis_ratio_delta=self.min_axis_ratio_delta,
            duplicate_epsilon=self.duplicate_epsilon,
        )
        residuals = tuple(_ellipse_residual(point, cx, cy, rx, ry, rotation) for point in point_sequence)
        fit_error = sum(residuals) / len(residuals)
        return PreciseEllipseResult(
            cx=cx,
            cy=cy,
            rx=rx,
            ry=ry,
            rotation=rotation,
            fit_error=fit_error,
            inlier_count=len(point_sequence),
            outlier_count=0,
            inlier_ratio=1.0,
        )


class RansacEllipseFitter:
    def __init__(self, config: RansacEllipseConfig | None = None) -> None:
        self.config = config or RansacEllipseConfig()
        self._precise_fitter = PreciseEllipseFitter(
            min_axis_ratio_delta=self.config.min_axis_ratio_delta,
            duplicate_epsilon=self.config.duplicate_epsilon,
        )

    def fit(self, points: tuple[Point, ...] | list[Point]) -> RansacEllipseResult:
        point_sequence = _coerce_points(points)
        _validate_point_set(point_sequence, duplicate_epsilon=self.config.duplicate_epsilon)

        rng = random.Random(self.config.random_seed)
        min_inlier_count = max(5, math.ceil(len(point_sequence) * self.config.min_inlier_ratio))
        best: tuple[tuple[int, ...], PreciseEllipseResult] | None = None

        for _ in range(self.config.max_iterations):
            sample_indexes = self._sample_indexes(point_sequence, rng)
            if sample_indexes is None:
                continue

            sample_points = tuple(point_sequence[index] for index in sample_indexes)
            try:
                candidate = self._precise_fitter.fit(sample_points)
            except ValueError:
                continue

            try:
                inlier_indexes, refined = self._refine_candidate(point_sequence, candidate, min_inlier_count)
            except ValueError:
                continue

            candidate_result = (inlier_indexes, refined)
            if best is None or self._is_better(candidate_result, best):
                best = candidate_result

        if best is None:
            try:
                self._precise_fitter.fit(point_sequence)
            except ValueError as exc:
                message = str(exc)
                if "near-circular" in message or "collinear" in message:
                    raise
            raise ValueError("unable to find a robust ellipse with sufficient inliers")

        inlier_indexes, refined = best
        outlier_indexes = tuple(index for index in range(len(point_sequence)) if index not in set(inlier_indexes))
        inlier_ratio = len(inlier_indexes) / len(point_sequence)
        if inlier_ratio < self.config.min_inlier_ratio:
            raise ValueError("insufficient inlier ratio for robust ellipse fit")

        return RansacEllipseResult(
            cx=refined.cx,
            cy=refined.cy,
            rx=refined.rx,
            ry=refined.ry,
            rotation=refined.rotation,
            fit_error=refined.fit_error,
            inlier_count=len(inlier_indexes),
            outlier_count=len(outlier_indexes),
            inlier_ratio=inlier_ratio,
            inlier_indexes=inlier_indexes,
            outlier_indexes=outlier_indexes,
        )

    def _refine_candidate(
        self,
        point_sequence: tuple[Point, ...],
        candidate: PreciseEllipseResult,
        min_inlier_count: int,
    ) -> tuple[tuple[int, ...], PreciseEllipseResult]:
        inlier_indexes = self._inlier_indexes(
            point_sequence,
            cx=candidate.cx,
            cy=candidate.cy,
            rx=candidate.rx,
            ry=candidate.ry,
            rotation=candidate.rotation,
        )
        if len(inlier_indexes) < min_inlier_count:
            raise ValueError("insufficient inlier ratio for robust ellipse fit")

        refined = candidate
        for _ in range(3):
            refined = self._precise_fitter.fit(tuple(point_sequence[index] for index in inlier_indexes))
            updated_inliers = self._inlier_indexes(
                point_sequence,
                cx=refined.cx,
                cy=refined.cy,
                rx=refined.rx,
                ry=refined.ry,
                rotation=refined.rotation,
            )
            if len(updated_inliers) < min_inlier_count:
                raise ValueError("insufficient inlier ratio for robust ellipse fit")
            if updated_inliers == inlier_indexes:
                return (inlier_indexes, refined)
            inlier_indexes = updated_inliers

        refined = self._precise_fitter.fit(tuple(point_sequence[index] for index in inlier_indexes))
        return (inlier_indexes, refined)

    def _inlier_indexes(
        self,
        point_sequence: tuple[Point, ...],
        *,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        rotation: float,
    ) -> tuple[int, ...]:
        return tuple(
            index
            for index, point in enumerate(point_sequence)
            if _ellipse_residual(point, cx, cy, rx, ry, rotation) <= self.config.max_error
        )

    def _sample_indexes(self, points: tuple[Point, ...], rng: random.Random) -> tuple[int, ...] | None:
        max_attempts = 32
        for _ in range(max_attempts):
            indexes = tuple(sorted(rng.sample(range(len(points)), self.config.sample_size)))
            sample_points = tuple(points[index] for index in indexes)
            if _has_duplicate_points(sample_points, duplicate_epsilon=self.config.duplicate_epsilon):
                continue
            return indexes
        return None

    @staticmethod
    def _is_better(
        candidate: tuple[tuple[int, ...], PreciseEllipseResult],
        current_best: tuple[tuple[int, ...], PreciseEllipseResult],
    ) -> bool:
        candidate_inliers, candidate_result = candidate
        best_inliers, best_result = current_best
        if len(candidate_inliers) != len(best_inliers):
            return len(candidate_inliers) > len(best_inliers)
        if not math.isclose(candidate_result.fit_error, best_result.fit_error, rel_tol=1e-12, abs_tol=1e-12):
            return candidate_result.fit_error < best_result.fit_error
        return candidate_result.inlier_ratio > best_result.inlier_ratio


def _coerce_points(points: tuple[Point, ...] | list[Point]) -> tuple[Point, ...]:
    return tuple((float(x), float(y)) for x, y in points)


def _has_duplicate_points(points: tuple[Point, ...], duplicate_epsilon: float) -> bool:
    for index, point in enumerate(points):
        for other in points[index + 1 :]:
            if math.dist(point, other) <= duplicate_epsilon:
                return True
    return False


def _validate_point_set(points: tuple[Point, ...], duplicate_epsilon: float) -> None:
    if len(points) < 5:
        raise ValueError("at least five Vector Space points are required")

    unique_points: list[Point] = []
    for point in points:
        if all(math.dist(point, existing) > duplicate_epsilon for existing in unique_points):
            unique_points.append(point)

    if len(unique_points) < 5:
        raise ValueError("at least five unique Vector Space points are required")

    centered = np.asarray(unique_points, dtype=np.float64)
    centered -= centered.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values[-1] <= duplicate_epsilon or singular_values[-1] / singular_values[0] < 1e-3:
        raise ValueError("point set is approximately collinear")


def _fit_conic(points: tuple[Point, ...]) -> np.ndarray:
    data = np.asarray(points, dtype=np.float64)
    mean = data.mean(axis=0)
    centered = data - mean
    scale = math.sqrt(2.0) / max(np.sqrt((centered * centered).sum(axis=1).mean()), 1e-12)
    normalized = centered * scale
    x = normalized[:, 0]
    y = normalized[:, 1]

    design_quadratic = np.column_stack((x * x, x * y, y * y))
    design_linear = np.column_stack((x, y, np.ones_like(x)))

    s1 = design_quadratic.T @ design_quadratic
    s2 = design_quadratic.T @ design_linear
    s3 = design_linear.T @ design_linear
    try:
        transform = -np.linalg.solve(s3, s2.T)
        reduced = s1 + (s2 @ transform)
        constraint = np.array(
            (
                (0.0, 0.0, 2.0),
                (0.0, -1.0, 0.0),
                (2.0, 0.0, 0.0),
            ),
            dtype=np.float64,
        )
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(constraint, reduced))
    except np.linalg.LinAlgError as exc:
        raise ValueError("unable to solve ellipse conic system") from exc

    candidates: list[np.ndarray] = []
    for index in range(eigenvectors.shape[1]):
        vector = np.real_if_close(eigenvectors[:, index]).astype(np.float64)
        if not np.all(np.isfinite(vector)):
            continue
        a, b, c = vector
        if (4.0 * a * c) - (b * b) <= 0.0:
            continue
        full_vector = np.concatenate((vector, transform @ vector))
        if np.all(np.isfinite(full_vector)):
            candidates.append(full_vector)

    if not candidates:
        raise ValueError("fitted conic is not a valid ellipse")

    conic = candidates[0]
    conic_matrix = np.array(
        (
            (conic[0], conic[1] / 2.0, conic[3] / 2.0),
            (conic[1] / 2.0, conic[2], conic[4] / 2.0),
            (conic[3] / 2.0, conic[4] / 2.0, conic[5]),
        ),
        dtype=np.float64,
    )
    normalization = np.array(
        (
            (scale, 0.0, -scale * mean[0]),
            (0.0, scale, -scale * mean[1]),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    denormalized = normalization.T @ conic_matrix @ normalization
    return np.array(
        (
            denormalized[0, 0],
            2.0 * denormalized[0, 1],
            denormalized[1, 1],
            2.0 * denormalized[0, 2],
            2.0 * denormalized[1, 2],
            denormalized[2, 2],
        ),
        dtype=np.float64,
    )


def _conic_to_geometric(
    conic: np.ndarray,
    *,
    min_axis_ratio_delta: float,
    duplicate_epsilon: float,
) -> tuple[float, float, float, float, float]:
    a, b, c, d, e, f = (float(value) for value in conic)
    if (a + c) < 0.0:
        a, b, c, d, e, f = (-a, -b, -c, -d, -e, -f)
    if (b * b) - (4.0 * a * c) >= 0.0:
        raise ValueError("fitted conic is not an ellipse")

    system = np.array(((2.0 * a, b), (b, 2.0 * c)), dtype=np.float64)
    rhs = np.array((-d, -e), dtype=np.float64)
    try:
        center = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("unable to solve ellipse center") from exc

    cx = float(center[0])
    cy = float(center[1])
    translated_constant = (
        f
        + (a * cx * cx)
        + (b * cx * cy)
        + (c * cy * cy)
        + (d * cx)
        + (e * cy)
    )
    quadratic = np.array(((a, b / 2.0), (b / 2.0, c)), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
    if np.any(~np.isfinite(eigenvalues)) or np.any(eigenvalues <= duplicate_epsilon):
        raise ValueError("ellipse quadratic form is not positive definite")
    if translated_constant >= -duplicate_epsilon:
        raise ValueError("ellipse constant term is invalid")

    axes = np.sqrt((-translated_constant) / eigenvalues)
    if np.any(~np.isfinite(axes)) or np.any(axes <= duplicate_epsilon):
        raise ValueError("ellipse axes are invalid")

    order = np.argsort(axes)[::-1]
    rx = float(axes[order[0]])
    ry = float(axes[order[1]])
    major_vector = eigenvectors[:, order[0]]
    rotation = math.atan2(float(major_vector[1]), float(major_vector[0]))

    if (rx - ry) / rx < min_axis_ratio_delta:
        raise ValueError("ellipse axes are too similar; near-circular fit is unstable")

    rotation = _normalize_rotation(rotation)
    return (cx, cy, rx, ry, rotation)


def _ellipse_residual(point: Point, cx: float, cy: float, rx: float, ry: float, rotation: float) -> float:
    translated_x = point[0] - cx
    translated_y = point[1] - cy
    cos_theta = math.cos(rotation)
    sin_theta = math.sin(rotation)
    local_x = (translated_x * cos_theta) + (translated_y * sin_theta)
    local_y = (-translated_x * sin_theta) + (translated_y * cos_theta)
    radial = math.sqrt(((local_x / rx) ** 2) + ((local_y / ry) ** 2))
    return abs(radial - 1.0) * ((rx + ry) / 2.0)


def _normalize_rotation(rotation: float) -> float:
    normalized = (rotation + (math.pi / 2.0)) % math.pi - (math.pi / 2.0)
    if math.isclose(normalized, math.pi / 2.0, abs_tol=1e-12):
        return -math.pi / 2.0
    return normalized


__all__ = [
    "PreciseEllipseFitter",
    "PreciseEllipseResult",
    "RansacEllipseConfig",
    "RansacEllipseFitter",
    "RansacEllipseResult",
]
