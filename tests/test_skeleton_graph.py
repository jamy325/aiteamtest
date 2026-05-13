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


def test_skeleton_graph_trace_graph_exposes_t_junction() -> None:
    image = np.zeros((40, 40), dtype=np.uint8)
    # T junction
    cv2.line(image, (10, 20), (30, 20), 255, thickness=1)
    cv2.line(image, (20, 20), (20, 30), 255, thickness=1)

    result = SkeletonGraphTracer().trace_graph(image)

    assert len(result.paths) == 3
    assert len(result.junctions) == 1
    
    junction = result.junctions[0]
    assert junction.pixel == (20, 20)
    assert junction.degree == 3
    assert len(junction.endpoints) == 3

    path_indices = {ep.path_index for ep in junction.endpoints}
    assert len(path_indices) == 3
    
    for endpoint in junction.endpoints:
        path = result.paths[endpoint.path_index]
        if endpoint.is_start:
            assert path.pixels[0] == junction.pixel
        else:
            assert path.pixels[-1] == junction.pixel


def test_skeleton_graph_trace_graph_exposes_x_crossing() -> None:
    image = np.zeros((40, 40), dtype=np.uint8)
    # X crossing
    cv2.line(image, (10, 10), (30, 30), 255, thickness=1)
    cv2.line(image, (10, 30), (30, 10), 255, thickness=1)

    result = SkeletonGraphTracer().trace_graph(image)

    # Note: OpenCV line drawing might create a cluster of pixels at the center
    # which could form multiple nearby junctions or a single degree 4 junction.
    # We just need to verify that at least one junction exists and endpoints are properly mapped.
    assert len(result.junctions) >= 1
    
    mapped_endpoints = sum(len(j.endpoints) for j in result.junctions)
    # An X crossing creates 4 outer branches, so there should be at least 4 endpoints connected to junctions
    assert mapped_endpoints >= 4
    
    for j in result.junctions:
        assert j.degree >= 3
        for endpoint in j.endpoints:
            path = result.paths[endpoint.path_index]
            pixel = path.pixels[0] if endpoint.is_start else path.pixels[-1]
            assert pixel == j.pixel


def test_skeleton_graph_trace_graph_no_junction_in_open_line() -> None:
    image = np.zeros((24, 24), dtype=np.uint8)
    cv2.line(image, (2, 10), (20, 10), 255, thickness=1)
    
    result = SkeletonGraphTracer().trace_graph(image)
    assert len(result.paths) == 1
    assert len(result.junctions) == 0


def test_skeleton_graph_trace_graph_no_junction_in_closed_loop() -> None:
    image = np.zeros((48, 48), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (36, 36), 255, thickness=1)

    result = SkeletonGraphTracer().trace_graph(image)
    assert len(result.paths) == 1
    assert result.paths[0].closed is True
    assert len(result.junctions) == 0


def test_skeleton_graph_trace_graph_preserves_path_behavior() -> None:
    # Multiple components, closed and open, with junctions
    image = np.zeros((60, 60), dtype=np.uint8)
    cv2.line(image, (10, 10), (20, 10), 255, thickness=1) # Open line
    cv2.rectangle(image, (40, 10), (50, 20), 255, thickness=1) # Closed loop
    
    # Y junction
    cv2.line(image, (20, 40), (20, 50), 255, thickness=1)
    cv2.line(image, (20, 40), (10, 30), 255, thickness=1)
    cv2.line(image, (20, 40), (30, 30), 255, thickness=1)
    
    result = SkeletonGraphTracer().trace_graph(image)
    
    # Paths: 1 for open line, 1 for loop, 3 for Y junction branches
    assert len(result.paths) == 5
    
    # Exactly 1 junction
    assert len(result.junctions) == 1
    j = result.junctions[0]
    assert j.pixel == (20, 40)
    assert j.degree == 3
    assert len(j.endpoints) == 3
    
    # Closed loops should be marked closed
    closed_paths = [p for p in result.paths if p.closed]
    assert len(closed_paths) == 1
    
    for path in result.paths:
        _assert_path_is_continuous(path.pixels, closed=path.closed)
