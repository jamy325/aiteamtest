from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


Point = tuple[int, int]


@dataclass(frozen=True, slots=True)
class BinaryContour:
    contour_id: str
    source: str
    points: tuple[Point, ...]
    closed: bool
    area: float
    depth: int
    parent_contour: str | None
    children: tuple[str, ...]


class ContourExtractor:
    def __init__(
        self,
        threshold: int = 127,
        blur_kernel_size: int = 5,
        morphology_kernel_size: int = 3,
    ) -> None:
        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size
        self.morphology_kernel_size = morphology_kernel_size

    def extract_binary_contours(self, image: np.ndarray) -> tuple[BinaryContour, ...]:
        grayscale = self._to_grayscale(image)
        blurred = cv2.GaussianBlur(grayscale, (self.blur_kernel_size, self.blur_kernel_size), 0)
        _, binary = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY)
        denoised = cv2.medianBlur(binary, 3)
        kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), dtype=np.uint8)
        closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return ()

        hierarchy_data = hierarchy[0]
        contour_ids = tuple(f"binary_contour_{index}" for index in range(len(contours)))
        depths = self._compute_depths(hierarchy_data)
        children_lookup = self._build_children_lookup(hierarchy_data, contour_ids)

        extracted: list[BinaryContour] = []
        for index, contour in enumerate(contours):
            points = tuple((int(point[0][0]), int(point[0][1])) for point in contour)
            parent_index = int(hierarchy_data[index][3])
            parent_contour = contour_ids[parent_index] if parent_index >= 0 else None
            extracted.append(
                BinaryContour(
                    contour_id=contour_ids[index],
                    source="binary_contour",
                    points=points,
                    closed=len(points) >= 3,
                    area=float(cv2.contourArea(contour)),
                    depth=depths[index],
                    parent_contour=parent_contour,
                    children=children_lookup[index],
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


__all__ = ["BinaryContour", "ContourExtractor", "Point"]
