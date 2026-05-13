import ast
from pathlib import Path

import cv2
import numpy as np

from services.skeleton_graph import SkeletonGraphTracer


def _chebyshev_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]))


def _assert_path_is_continuous(points: tuple[tuple[int, int], ...], *, closed: bool) -> None:
    assert points
    for left, right in zip(points, points[1:]):
        assert _chebyshev_distance(left, right) <= 1
    if closed and len(points) > 1:
        assert _chebyshev_distance(points[-1], points[0]) <= 1


def test_skeleton_graph_traces_open_lines_in_endpoint_order() -> None:
    tracer = SkeletonGraphTracer()

    horizontal = np.zeros((24, 24), dtype=np.uint8)
    cv2.line(horizontal, (2, 10), (20, 10), 255, thickness=1)
    horizontal_paths = tracer.trace_mask(horizontal)
    assert len(horizontal_paths) == 1
    assert horizontal_paths[0].closed is False
    _assert_path_is_continuous(horizontal_paths[0].pixels, closed=False)
    assert horizontal_paths[0].pixels[0] == (2, 10)
    assert horizontal_paths[0].pixels[-1] == (20, 10)

    vertical = np.zeros((24, 24), dtype=np.uint8)
    cv2.line(vertical, (12, 2), (12, 20), 255, thickness=1)
    vertical_paths = tracer.trace_mask(vertical)
    assert len(vertical_paths) == 1
    assert vertical_paths[0].closed is False
    _assert_path_is_continuous(vertical_paths[0].pixels, closed=False)
    assert vertical_paths[0].pixels[0] == (12, 2)
    assert vertical_paths[0].pixels[-1] == (12, 20)

    diagonal = np.zeros((24, 24), dtype=np.uint8)
    cv2.line(diagonal, (3, 3), (19, 19), 255, thickness=1)
    diagonal_paths = tracer.trace_mask(diagonal)
    assert len(diagonal_paths) == 1
    assert diagonal_paths[0].closed is False
    _assert_path_is_continuous(diagonal_paths[0].pixels, closed=False)
    assert diagonal_paths[0].pixels[0] == (3, 3)
    assert diagonal_paths[0].pixels[-1] == (19, 19)


def test_skeleton_graph_traces_closed_loop_continuously() -> None:
    image = np.zeros((48, 48), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (36, 36), 255, thickness=1)

    paths = SkeletonGraphTracer().trace_mask(image)

    assert len(paths) == 1
    assert paths[0].closed is True
    assert len(paths[0].pixels) >= 20
    _assert_path_is_continuous(paths[0].pixels, closed=True)


def test_skeleton_graph_splits_multiple_disconnected_components() -> None:
    image = np.zeros((40, 60), dtype=np.uint8)
    cv2.line(image, (4, 10), (20, 10), 255, thickness=1)
    cv2.line(image, (35, 24), (55, 24), 255, thickness=1)

    paths = SkeletonGraphTracer().trace_mask(image)

    assert len(paths) == 2
    for path in paths:
        assert path.closed is False
        _assert_path_is_continuous(path.pixels, closed=False)
    endpoints = {(path.pixels[0], path.pixels[-1]) for path in paths}
    assert ((4, 10), (20, 10)) in endpoints
    assert ((35, 24), (55, 24)) in endpoints


def test_skeleton_graph_splits_branches_instead_of_row_major_scanning() -> None:
    image = np.zeros((40, 40), dtype=np.uint8)
    cv2.line(image, (20, 5), (20, 20), 255, thickness=1)
    cv2.line(image, (10, 20), (30, 20), 255, thickness=1)

    paths = SkeletonGraphTracer().trace_mask(image)

    assert len(paths) >= 3
    assert all(path.closed is False for path in paths)
    assert all(len(path.pixels) >= 2 for path in paths)
    for path in paths:
        _assert_path_is_continuous(path.pixels, closed=False)

    unique_endpoints = {
        path.pixels[0]
        for path in paths
    } | {
        path.pixels[-1]
        for path in paths
    }
    assert (20, 5) in unique_endpoints
    assert (10, 20) in unique_endpoints
    assert (30, 20) in unique_endpoints


def test_skeleton_graph_has_no_forbidden_dependencies() -> None:
    source = Path("services/skeleton_graph.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    forbidden_imports = {"openai", "anthropic", "PyQt5", "PyQt6", "ui", "core"}

    assert imports.isdisjoint(forbidden_imports)
