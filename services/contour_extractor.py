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
    ) -> None:
        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size
        self.morphology_kernel_size = morphology_kernel_size

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

    def extract_skeleton_contours(self, image: np.ndarray) -> tuple[BinaryContour, ...]:
        skeleton_mask = self._skeletonize(self._preprocess_skeleton_mask(image))
        component_count, labels = cv2.connectedComponents(skeleton_mask)
        extracted: list[BinaryContour] = []

        for component in range(1, component_count):
            ys, xs = np.where(labels == component)
            if len(xs) == 0:
                continue

            # TODO: replace row-major point ordering with path/topology ordering
            # before fitting/resampling tasks consume skeleton contours.
            points = tuple((int(x), int(y)) for y, x in sorted(zip(ys.tolist(), xs.tolist()), key=lambda item: (item[0], item[1])))
            extracted.append(
                BinaryContour(
                    contour_id=f"skeleton_contour_{component - 1}",
                    source="skeleton_contour",
                    points=points,
                    closed=False,
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

    @staticmethod
    def _skeletonize(binary_mask: np.ndarray) -> np.ndarray:
        skeleton = np.zeros_like(binary_mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        current = binary_mask.copy()

        while True:
            eroded = cv2.erode(current, element)
            opened = cv2.dilate(eroded, element)
            residue = cv2.subtract(current, opened)
            skeleton = cv2.bitwise_or(skeleton, residue)
            current = eroded
            if cv2.countNonZero(current) == 0:
                break

        return skeleton

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
