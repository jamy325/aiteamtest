from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Pixel = tuple[int, int]

_NEIGHBOR_OFFSETS: tuple[Pixel, ...] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


@dataclass(frozen=True, slots=True)
class SkeletonPath:
    pixels: tuple[Pixel, ...]
    closed: bool


@dataclass(frozen=True, slots=True)
class SkeletonEndpoint:
    path_index: int
    is_start: bool


@dataclass(frozen=True, slots=True)
class SkeletonJunction:
    pixel: Pixel
    junction_id: str
    degree: int
    endpoints: tuple[SkeletonEndpoint, ...]


@dataclass(frozen=True, slots=True)
class SkeletonGraphTraceResult:
    paths: tuple[SkeletonPath, ...]
    junctions: tuple[SkeletonJunction, ...]


class SkeletonGraphTracer:
    def trace_mask(self, skeleton_mask: np.ndarray) -> tuple[SkeletonPath, ...]:
        return self.trace_graph(skeleton_mask).paths

    def trace_graph(self, skeleton_mask: np.ndarray) -> SkeletonGraphTraceResult:
        normalized = self._normalize_mask(skeleton_mask)
        pixels = self._mask_pixels(normalized)
        if not pixels:
            return SkeletonGraphTraceResult(paths=(), junctions=())

        adjacency = self._build_adjacency(pixels)
        traced_paths: list[SkeletonPath] = []
        for component in self._connected_components(adjacency):
            component_set = set(component)
            component_adjacency = {
                pixel: tuple(neighbor for neighbor in adjacency[pixel] if neighbor in component_set)
                for pixel in component
            }
            traced_paths.extend(self._trace_component(component_adjacency))

        junction_pixels = {pixel for pixel, neighbors in adjacency.items() if len(neighbors) >= 3}
        junction_endpoints: dict[Pixel, list[SkeletonEndpoint]] = {p: [] for p in junction_pixels}

        for path_index, path in enumerate(traced_paths):
            if path.closed:
                continue
            
            start_p = path.pixels[0]
            if start_p in junction_pixels:
                junction_endpoints[start_p].append(SkeletonEndpoint(path_index=path_index, is_start=True))
                
            end_p = path.pixels[-1]
            if end_p in junction_pixels:
                junction_endpoints[end_p].append(SkeletonEndpoint(path_index=path_index, is_start=False))

        junctions: list[SkeletonJunction] = []
        for i, p in enumerate(sorted(junction_pixels, key=self._pixel_sort_key)):
            junctions.append(
                SkeletonJunction(
                    pixel=p,
                    junction_id=f"junction_{i}",
                    degree=len(adjacency[p]),
                    endpoints=tuple(junction_endpoints[p]),
                )
            )

        return SkeletonGraphTraceResult(paths=tuple(traced_paths), junctions=tuple(junctions))

    def _trace_component(self, adjacency: dict[Pixel, tuple[Pixel, ...]]) -> tuple[SkeletonPath, ...]:
        if not adjacency:
            return ()

        if len(adjacency) == 1:
            pixel = next(iter(adjacency))
            return (SkeletonPath(pixels=(pixel,), closed=False),)

        critical_nodes = {
            pixel
            for pixel, neighbors in adjacency.items()
            if len(neighbors) != 2
        }

        if not critical_nodes:
            return (self._trace_closed_loop(adjacency),)

        visited_edges: set[tuple[Pixel, Pixel]] = set()
        paths: list[SkeletonPath] = []

        for start in sorted(critical_nodes, key=self._pixel_sort_key):
            for neighbor in adjacency[start]:
                edge = self._edge_key(start, neighbor)
                if edge in visited_edges:
                    continue

                visited_edges.add(edge)
                path = [start]
                previous = start
                current = neighbor

                while True:
                    path.append(current)
                    if current in critical_nodes:
                        break

                    next_candidates = [item for item in adjacency[current] if item != previous]
                    if not next_candidates:
                        break

                    next_pixel = next_candidates[0]
                    next_edge = self._edge_key(current, next_pixel)
                    if next_edge in visited_edges:
                        break

                    visited_edges.add(next_edge)
                    previous, current = current, next_pixel

                closed = len(path) > 2 and path[0] == path[-1]
                if closed:
                    path = path[:-1]
                paths.append(SkeletonPath(pixels=tuple(path), closed=closed))

        return tuple(paths)

    def _trace_closed_loop(self, adjacency: dict[Pixel, tuple[Pixel, ...]]) -> SkeletonPath:
        start = min(adjacency, key=self._pixel_sort_key)
        previous = start
        current = adjacency[start][0]
        path = [start]

        while True:
            path.append(current)
            next_candidates = [item for item in adjacency[current] if item != previous]
            if not next_candidates:
                break

            next_pixel = next_candidates[0]
            previous, current = current, next_pixel
            if current == start:
                break

        return SkeletonPath(pixels=tuple(path), closed=True)

    def _build_adjacency(self, pixels: set[Pixel]) -> dict[Pixel, tuple[Pixel, ...]]:
        adjacency: dict[Pixel, tuple[Pixel, ...]] = {}
        for pixel in pixels:
            x, y = pixel
            neighbors: list[Pixel] = []
            for dx, dy in _NEIGHBOR_OFFSETS:
                candidate = (x + dx, y + dy)
                if candidate not in pixels:
                    continue

                if abs(dx) == 1 and abs(dy) == 1:
                    bridge_a = (x + dx, y)
                    bridge_b = (x, y + dy)
                    if bridge_a in pixels or bridge_b in pixels:
                        continue

                neighbors.append(candidate)

            adjacency[pixel] = tuple(sorted(neighbors, key=self._pixel_sort_key))

        return adjacency

    def _connected_components(self, adjacency: dict[Pixel, tuple[Pixel, ...]]) -> tuple[tuple[Pixel, ...], ...]:
        remaining = set(adjacency)
        components: list[tuple[Pixel, ...]] = []

        while remaining:
            start = min(remaining, key=self._pixel_sort_key)
            stack = [start]
            component: list[Pixel] = []
            remaining.remove(start)

            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)

            components.append(tuple(sorted(component, key=self._pixel_sort_key)))

        return tuple(components)

    @staticmethod
    def _normalize_mask(skeleton_mask: np.ndarray) -> np.ndarray:
        array = np.asarray(skeleton_mask)
        if array.ndim != 2:
            raise ValueError("skeleton mask must be a 2D array")
        return (array > 0).astype(np.uint8)

    @staticmethod
    def _mask_pixels(mask: np.ndarray) -> set[Pixel]:
        ys, xs = np.where(mask > 0)
        return {(int(x), int(y)) for y, x in zip(ys.tolist(), xs.tolist())}

    @staticmethod
    def _pixel_sort_key(pixel: Pixel) -> tuple[int, int]:
        return (int(pixel[1]), int(pixel[0]))

    def _edge_key(self, left: Pixel, right: Pixel) -> tuple[Pixel, Pixel]:
        if self._pixel_sort_key(left) <= self._pixel_sort_key(right):
            return (left, right)
        return (right, left)


__all__ = ["Pixel", "SkeletonEndpoint", "SkeletonGraphTraceResult", "SkeletonGraphTracer", "SkeletonJunction", "SkeletonPath"]
