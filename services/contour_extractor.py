from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.types import CoordinateSystem
from services.skeleton_graph import SkeletonGraphTracer


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class BinaryContour:
    contour_id: str
    source: str
    points: tuple[Point, ...]
    coordinate_space: str
    closed: bool
    area: float
    depth: int
    parent_contour: str | None
    children: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExtractedContours:
    binary_contours: tuple[BinaryContour, ...]
    skeleton_contours: tuple[BinaryContour, ...]


class ContourExtractor:
    def __init__(
        self,
        threshold: int = 127,
        blur_kernel_size: int = 5,
        morphology_kernel_size: int = 3,
        coordinate_transformer: CoordinateTransformer | None = None,
        skeleton_graph_tracer: SkeletonGraphTracer | None = None,
    ) -> None:
        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size
        self.morphology_kernel_size = morphology_kernel_size
        self.coordinate_transformer = coordinate_transformer or CoordinateTransformer(CoordinateSystem())
        self.skeleton_graph_tracer = skeleton_graph_tracer or SkeletonGraphTracer()

    def extract_contours(self, image: np.ndarray) -> ExtractedContours:
        binary_contours = self.extract_binary_contours(image)
        skeleton_contours = self.extract_skeleton_contours(image)
        return ExtractedContours(binary_contours=binary_contours, skeleton_contours=skeleton_contours)

    def extract_binary_contours(self, image: np.ndarray) -> tuple[BinaryContour, ...]:
        closed = self._preprocess_binary_mask(image)
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return ()

        hierarchy_data = hierarchy[0]
        contour_ids = tuple(f"binary_contour_{index}" for index in range(len(contours)))
        depths = self._compute_depths(hierarchy_data)
        children_lookup = self._build_children_lookup(hierarchy_data, contour_ids)

        extracted: list[BinaryContour] = []
        for index, contour in enumerate(contours):
            points = self._to_vector_points(tuple((int(point[0][0]), int(point[0][1])) for point in contour))
            parent_index = int(hierarchy_data[index][3])
            parent_contour = contour_ids[parent_index] if parent_index >= 0 else None
            extracted.append(
                BinaryContour(
                    contour_id=contour_ids[index],
                    source="binary_contour",
                    points=points,
                    coordinate_space="vector",
                    closed=len(points) >= 3,
                    area=float(cv2.contourArea(contour)),
                    depth=depths[index],
                    parent_contour=parent_contour,
                    children=children_lookup[index],
                )
            )

        return tuple(extracted)

    def extract_skeleton_contours(self, image: np.ndarray) -> tuple[BinaryContour, ...]:
        skeleton_mask = self._skeletonize(self._preprocess_skeleton_mask(image))
        traced_paths = self.skeleton_graph_tracer.trace_mask(skeleton_mask)
        extracted: list[BinaryContour] = []

        for index, traced_path in enumerate(traced_paths):
            if len(traced_path.pixels) < 2:
                continue

            points = self._to_vector_points(traced_path.pixels)
            extracted.append(
                BinaryContour(
                    contour_id=f"skeleton_contour_{index}",
                    source="skeleton_contour",
                    points=points,
                    coordinate_space="vector",
                    closed=traced_path.closed,
                    area=float(len(points)),
                    depth=0,
                    parent_contour=None,
                    children=(),
                )
            )

        return tuple(extracted)

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError("image must be a 2D grayscale or 3D BGR array")

    def _preprocess_binary_mask(self, image: np.ndarray) -> np.ndarray:
        grayscale = self._to_grayscale(image)
        blurred = cv2.GaussianBlur(grayscale, (self.blur_kernel_size, self.blur_kernel_size), 0)
        _, binary = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY)
        denoised = cv2.medianBlur(binary, 3)
        kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    def _preprocess_skeleton_mask(self, image: np.ndarray) -> np.ndarray:
        grayscale = self._to_grayscale(image)
        _, binary = cv2.threshold(grayscale, self.threshold, 255, cv2.THRESH_BINARY)
        return binary

    def _to_vector_points(self, points: tuple[tuple[int, int], ...]) -> tuple[Point, ...]:
        return tuple(self.coordinate_transformer.pixel_to_vector((float(point[0]), float(point[1]))) for point in points)

    @staticmethod
    def _skeletonize(binary_mask: np.ndarray) -> np.ndarray:
        image = (binary_mask > 0).astype(np.uint8)
        changed = True

        while changed:
            changed = False
            for first_sub_iteration in (True, False):
                removable: list[tuple[int, int]] = []
                rows, cols = image.shape
                for y in range(1, rows - 1):
                    for x in range(1, cols - 1):
                        if image[y, x] != 1:
                            continue

                        p2 = image[y - 1, x]
                        p3 = image[y - 1, x + 1]
                        p4 = image[y, x + 1]
                        p5 = image[y + 1, x + 1]
                        p6 = image[y + 1, x]
                        p7 = image[y + 1, x - 1]
                        p8 = image[y, x - 1]
                        p9 = image[y - 1, x - 1]
                        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                        neighbor_count = sum(neighbors)
                        if neighbor_count < 2 or neighbor_count > 6:
                            continue

                        transitions = sum(
                            neighbors[index] == 0 and neighbors[(index + 1) % 8] == 1
                            for index in range(8)
                        )
                        if transitions != 1:
                            continue

                        if first_sub_iteration:
                            if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
                                continue
                        else:
                            if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
                                continue

                        removable.append((y, x))

                if removable:
                    changed = True
                    for y, x in removable:
                        image[y, x] = 0

        return (image * 255).astype(np.uint8)

    @staticmethod
    def _compute_depths(hierarchy: np.ndarray) -> tuple[int, ...]:
        depths: list[int] = []
        for index in range(len(hierarchy)):
            depth = 0
            parent = int(hierarchy[index][3])
            while parent >= 0:
                depth += 1
                parent = int(hierarchy[parent][3])
            depths.append(depth)
        return tuple(depths)

    @staticmethod
    def _build_children_lookup(hierarchy: np.ndarray, contour_ids: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        children: list[list[str]] = [[] for _ in range(len(hierarchy))]
        for index in range(len(hierarchy)):
            parent = int(hierarchy[index][3])
            if parent >= 0:
                children[parent].append(contour_ids[index])
        return tuple(tuple(items) for items in children)


__all__ = ["BinaryContour", "ContourExtractor", "ExtractedContours", "Point"]
