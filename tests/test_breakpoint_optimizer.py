import ast
import math
from pathlib import Path

from services.breakpoint_optimizer import BreakPointOptimizer, BreakPointRequest


def test_breakpoint_optimizer_finds_transition_from_line_to_arc() -> None:
    line_points = [(float(index), 0.0) for index in range(8)]
    arc_points = [
        (7.0 + math.sin(theta), 1.0 - math.cos(theta))
        for theta in (0.15, 0.35, 0.6, 0.9, 1.2, 1.45)
    ]
    points = tuple(line_points + arc_points)
    optimizer = BreakPointOptimizer()

    result = optimizer.optimize(
        BreakPointRequest(
            points=points,
            rough_range=(5, 10),
            target_type="arc",
        )
    )

    assert result.breakpoints
    assert 6 <= result.breakpoints[0] <= 8
    assert result.optimized_range[0] >= 5
    assert result.optimized_range[1] <= 10
    assert "curvature_jump" in result.reason or "angle_jump" in result.reason
    assert result.confidence > 0.0


def test_breakpoint_optimizer_finds_sharp_corner() -> None:
    points = tuple(
        [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (4.0, 2.0), (4.0, 4.0), (4.0, 6.0)]
    )
    optimizer = BreakPointOptimizer()

    result = optimizer.optimize(
        BreakPointRequest(
            points=points,
            rough_range=(1, 4),
            target_type="polyline",
            adjacent_endpoints=(2,),
        )
    )

    assert result.breakpoints == (2,)
    assert result.optimized_range == (1, 3)
    assert "angle_jump" in result.reason
    assert "adjacent_endpoint" in result.reason


def test_breakpoint_optimizer_uses_residual_peak_and_user_hint() -> None:
    points = tuple((float(index), 0.0) for index in range(10))
    residuals = (0.0, 0.02, 0.01, 0.03, 0.1, 1.2, 0.08, 0.02, 0.01, 0.0)
    optimizer = BreakPointOptimizer()

    result = optimizer.optimize(
        BreakPointRequest(
            points=points,
            rough_range=(2, 7),
            target_type="line",
            residuals=residuals,
            user_breakpoints=(5,),
            ai_marked_range=(4, 6),
        )
    )

    assert result.breakpoints == (5,)
    assert result.optimized_range == (3, 7)
    assert "residual_peak" in result.reason
    assert "user_breakpoint" in result.reason
    assert "ai_marked_region" in result.reason


def test_breakpoint_optimizer_clamps_out_of_bounds_ranges() -> None:
    points = tuple((float(index), 0.0) for index in range(6))
    optimizer = BreakPointOptimizer()

    result = optimizer.optimize(
        BreakPointRequest(
            points=points,
            rough_range=(-10, 99),
            target_type="unknown",
            user_breakpoints=(1,),
        )
    )

    assert 0 <= result.optimized_range[0] <= result.optimized_range[1] <= len(points) - 1
    assert result.breakpoints == (1,)


def test_breakpoint_optimizer_has_no_forbidden_dependencies() -> None:
    source_path = Path("services/breakpoint_optimizer.py")
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
