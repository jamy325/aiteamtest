import ast
import math
from pathlib import Path

import pytest

from services.breakpoint_optimizer import BreakPointRequest, BreakPointResult
from services.refiner import (
    RefinementEngine,
    RefinementRequest,
)


class _StubBreakPointOptimizer:
    def __init__(self, optimized_range: tuple[int, int], breakpoints: tuple[int, ...] = ()) -> None:
        self.optimized_range = optimized_range
        self.breakpoints = breakpoints
        self.last_request: BreakPointRequest | None = None

    def optimize(self, request: BreakPointRequest) -> BreakPointResult:
        self.last_request = request
        return BreakPointResult(
            optimized_range=self.optimized_range,
            breakpoints=self.breakpoints,
            confidence=0.9,
            reason="stubbed_range",
        )


def test_refinement_engine_dispatches_line_refinement_and_returns_feedback() -> None:
    points = tuple((float(index), 2.0 * float(index) + 1.0) for index in range(8))
    optimizer = _StubBreakPointOptimizer(optimized_range=(0, len(points) - 1), breakpoints=(4,))
    engine = RefinementEngine(breakpoint_optimizer=optimizer)

    result = engine.refine(
        RefinementRequest(
            points=points,
            rough_range=(0, len(points) - 1),
            target_type="line",
        )
    )

    assert optimizer.last_request is not None
    assert optimizer.last_request.target_type == "line"
    assert result.target_type == "line"
    assert result.optimized_range == (0, len(points) - 1)
    assert result.breakpoint_result.breakpoints == (4,)
    assert result.params is not None
    assert result.params["start"] == pytest.approx([0.0, 1.0], abs=0.08)
    assert result.params["end"] == pytest.approx([7.0, 15.0], abs=0.08)
    assert result.inlier_ratio == pytest.approx(1.0)
    assert result.fit_error < 0.08
    assert set(result.inlier_indexes) == set(range(8))
    assert result.outlier_indexes == ()
    assert result.confidence_result.failure_reason is None
    assert result.confidence_result.confidence > 0.65
    assert result.feedback.success is True
    assert result.feedback.retry_policy == "accept"
    assert result.failure_message is None


def test_refinement_engine_refine_circle_returns_precise_candidate() -> None:
    center_x = 10.0
    center_y = -5.0
    radius = 8.0
    angles = [index * (math.pi / 8.0) for index in range(12)]
    points = tuple(
        (
            center_x + radius * math.cos(angle) + (0.03 if index % 2 == 0 else -0.02),
            center_y + radius * math.sin(angle) + (0.02 if index % 3 == 0 else -0.01),
        )
        for index, angle in enumerate(angles)
    )
    optimizer = _StubBreakPointOptimizer(optimized_range=(0, len(points) - 1), breakpoints=(6,))
    engine = RefinementEngine(breakpoint_optimizer=optimizer)

    result = engine.refine_circle(
        RefinementRequest(
            points=points,
            rough_range=(0, len(points) - 1),
            target_type="circle",
        )
    )

    assert optimizer.last_request is not None
    assert optimizer.last_request.target_type == "circle"
    assert result.params is not None
    assert result.params["cx"] == pytest.approx(center_x, abs=0.08)
    assert result.params["cy"] == pytest.approx(center_y, abs=0.08)
    assert result.params["r"] == pytest.approx(radius, abs=0.08)
    assert result.inlier_ratio == pytest.approx(1.0)
    assert result.fit_error < 0.08
    assert result.confidence_result.failure_reason is None
    assert result.confidence_result.confidence > 0.65
    assert result.feedback.success is True


def test_refinement_engine_refine_arc_returns_precise_candidate() -> None:
    center_x = 20.0
    center_y = -10.0
    radius = 12.0
    angles = [math.radians(30.0 + (index * 12.0)) for index in range(12)]
    points = tuple(
        (
            center_x + radius * math.cos(angle) + (0.03 if index % 2 == 0 else -0.02),
            center_y + radius * math.sin(angle) + (0.02 if index % 3 == 0 else -0.01),
        )
        for index, angle in enumerate(angles)
    )
    optimizer = _StubBreakPointOptimizer(optimized_range=(0, len(points) - 1), breakpoints=(5,))
    engine = RefinementEngine(breakpoint_optimizer=optimizer)

    result = engine.refine_arc(
        RefinementRequest(
            points=points,
            rough_range=(0, len(points) - 1),
            target_type="arc",
        )
    )

    assert optimizer.last_request is not None
    assert optimizer.last_request.target_type == "arc"
    assert result.params is not None
    assert result.params["cx"] == pytest.approx(center_x, abs=0.08)
    assert result.params["cy"] == pytest.approx(center_y, abs=0.08)
    assert result.params["r"] == pytest.approx(radius, abs=0.08)
    assert result.params["direction"] == "ccw"
    assert result.params["start_angle"] == pytest.approx(angles[0], abs=0.08)
    assert result.params["end_angle"] == pytest.approx(angles[-1], abs=0.08)
    assert result.inlier_ratio == pytest.approx(1.0)
    assert result.fit_error < 0.08
    assert result.confidence_result.failure_reason is None
    assert result.confidence_result.confidence > 0.65
    assert result.feedback.success is True


def test_refinement_engine_returns_failure_feedback_when_fit_fails() -> None:
    points = (
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
        (4.0, 4.0),
    )
    optimizer = _StubBreakPointOptimizer(optimized_range=(0, len(points) - 1), breakpoints=(2,))
    engine = RefinementEngine(breakpoint_optimizer=optimizer)

    result = engine.refine_circle(
        RefinementRequest(
            points=points,
            rough_range=(0, len(points) - 1),
            target_type="circle",
        )
    )

    assert result.params is None
    assert result.confidence_result.confidence == 0.0
    assert result.confidence_result.failure_reason == "high_fit_error"
    assert result.feedback.success is False
    assert result.feedback.reason == "high_fit_error"
    assert result.feedback.retry_policy == "retry_with_new_shape"
    assert result.failure_message is not None
    assert "unable to fit a robust circle" in result.failure_message


def test_refinement_engine_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/refiner.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"cv2", "matplotlib", "PyQt5", "PyQt6", "openai", "anthropic", "ui"}

    assert imports.isdisjoint(forbidden_imports)
    assert "open(" not in source
