from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from core.coordinate import CoordinateTransformer
from core.types import CoordinateSystem
from services.skeleton_graph import SkeletonGraphTraceResult, SkeletonGraphTracer, SkeletonJunction


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
    skeleton_junctions: tuple[SkeletonJunction, ...] = ()


@dataclass(frozen=True, slots=True)
class ContourExtractionDebugArtifacts:
    grayscale: np.ndarray
    alpha: np.ndarray | None
    alpha_mask: np.ndarray | None
    threshold_binary: np.ndarray
    denoised: np.ndarray
    morphology_closed: np.ndarray
    binary_contours_overlay: np.ndarray
    binary_contours_hierarchy: tuple[dict[str, object], ...]
    skeleton_mask: np.ndarray
    skeleton_contours_overlay: np.ndarray
    binary_contours: tuple[BinaryContour, ...]
    skeleton_contours: tuple[BinaryContour, ...]
    timings_ms: dict[str, float]
    threshold_polarity: str = "foreground_white"


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
        extracted, _ = self.extract_contours_with_debug(image)
        return extracted

    def extract_contours_with_debug(self, image: np.ndarray) -> tuple[ExtractedContours, ContourExtractionDebugArtifacts]:
        timings_ms: dict[str, float] = {}
        start = perf_counter()
        grayscale = self._to_grayscale(image)
        timings_ms["grayscale"] = self._elapsed_ms(start)
        alpha, alpha_mask = self._extract_alpha_debug(image)

        start = perf_counter()
        threshold_binary, denoised, closed = self._preprocess_binary_stages(grayscale)
        timings_ms["binary_preprocess"] = self._elapsed_ms(start)

        start = perf_counter()
        binary_contours, binary_hierarchy, contours_pixels = self._extract_binary_from_closed_mask(closed, image.shape[:2])
        timings_ms["binary_contours"] = self._elapsed_ms(start)

        start = perf_counter()
        skeleton_binary = self._preprocess_skeleton_mask(image)
        skeleton_mask = self._skeletonize(skeleton_binary)
        trace_result = self.skeleton_graph_tracer.trace_graph(skeleton_mask)
        skeleton_contours = self._extract_skeleton_contours_from_trace(trace_result)
        timings_ms["skeleton"] = self._elapsed_ms(start)

        binary_overlay = self._draw_contours_overlay(
            image=image,
            contours_pixels=contours_pixels,
            contour_ids=tuple(contour.contour_id for contour in binary_contours),
            closed_lookup={contour.contour_id: contour.closed for contour in binary_contours},
        )
        skeleton_pixels = {
            contour.contour_id: tuple(
                (int(round(pixel[0])), int(round(pixel[1])))
                for pixel in (
                    self.coordinate_transformer.vector_to_pixel(point)
                    for point in contour.points
                )
            )
            for contour in skeleton_contours
        }
        skeleton_overlay = self._draw_contours_overlay(
            image=image,
            contours_pixels=skeleton_pixels,
            contour_ids=tuple(contour.contour_id for contour in skeleton_contours),
            closed_lookup={contour.contour_id: contour.closed for contour in skeleton_contours},
        )

        extracted = ExtractedContours(
            binary_contours=binary_contours,
            skeleton_contours=skeleton_contours,
            skeleton_junctions=trace_result.junctions,
        )
        debug = ContourExtractionDebugArtifacts(
            grayscale=grayscale,
            alpha=alpha,
            alpha_mask=alpha_mask,
            threshold_binary=threshold_binary,
            denoised=denoised,
            morphology_closed=closed,
            binary_contours_overlay=binary_overlay,
            binary_contours_hierarchy=binary_hierarchy,
            skeleton_mask=skeleton_mask,
            skeleton_contours_overlay=skeleton_overlay,
            binary_contours=binary_contours,
            skeleton_contours=skeleton_contours,
            timings_ms=timings_ms,
        )
        return extracted, debug

    def extract_binary_contours(self, image: np.ndarray) -> tuple[BinaryContour, ...]:
        grayscale = self._to_grayscale(image)
        _, _, closed = self._preprocess_binary_stages(grayscale)
        extracted, _, _ = self._extract_binary_from_closed_mask(closed, image.shape[:2])
        return extracted

    def extract_skeleton_contours(self, image: np.ndarray) -> tuple[BinaryContour, ...]:
        trace_result = self._trace_skeleton_graph(image)
        return self._extract_skeleton_contours_from_trace(trace_result)

    def _trace_skeleton_graph(self, image: np.ndarray) -> SkeletonGraphTraceResult:
        skeleton_mask = self._skeletonize(self._preprocess_skeleton_mask(image))
        return self.skeleton_graph_tracer.trace_graph(skeleton_mask)

    def _extract_skeleton_contours_from_trace(self, trace_result: SkeletonGraphTraceResult) -> tuple[BinaryContour, ...]:
        extracted: list[BinaryContour] = []
        for index, traced_path in enumerate(trace_result.paths):
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
                    # Skeleton contours are centerline traces, so this field currently
                    # represents traced point count rather than enclosed geometric area.
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
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError("image must be a 2D grayscale or 3D BGR array")

    def _preprocess_binary_mask(self, image: np.ndarray) -> np.ndarray:
        grayscale = self._to_grayscale(image)
        _, _, closed = self._preprocess_binary_stages(grayscale)
        return closed

    def _preprocess_skeleton_mask(self, image: np.ndarray) -> np.ndarray:
        grayscale = self._to_grayscale(image)
        _, binary = cv2.threshold(grayscale, self.threshold, 255, cv2.THRESH_BINARY)
        return binary

    def _preprocess_binary_stages(self, grayscale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        blurred = cv2.GaussianBlur(grayscale, (self.blur_kernel_size, self.blur_kernel_size), 0)
        _, binary = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY)
        denoised = cv2.medianBlur(binary, 3)
        kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), dtype=np.uint8)
        closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        return binary, denoised, closed

    def _extract_binary_from_closed_mask(
        self,
        closed: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[tuple[BinaryContour, ...], tuple[dict[str, object], ...], dict[str, tuple[tuple[int, int], ...]]]:
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return (), (), {}

        height, width = image_shape
        hierarchy_data = hierarchy[0]
        contour_ids = tuple(f"binary_contour_{index}" for index in range(len(contours)))
        depths = self._compute_depths(hierarchy_data)
        children_lookup = self._build_children_lookup(hierarchy_data, contour_ids)

        extracted: list[BinaryContour] = []
        debug_hierarchy: list[dict[str, object]] = []
        contour_pixels: dict[str, tuple[tuple[int, int], ...]] = {}
        total_area = float(width * height) if width > 0 and height > 0 else 1.0
        for index, contour in enumerate(contours):
            pixel_points = tuple((int(point[0][0]), int(point[0][1])) for point in contour)
            contour_pixels[contour_ids[index]] = pixel_points
            points = self._to_vector_points(pixel_points)
            parent_index = int(hierarchy_data[index][3])
            parent_contour = contour_ids[parent_index] if parent_index >= 0 else None
            area_px = float(cv2.contourArea(contour))
            x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
            touches_border = x <= 0 or y <= 0 or (x + bbox_width) >= width or (y + bbox_height) >= height
            extracted.append(
                BinaryContour(
                    contour_id=contour_ids[index],
                    source="binary_contour",
                    points=points,
                    coordinate_space="vector",
                    closed=len(points) >= 3,
                    area=self._pixel_area_to_vector_area(area_px),
                    depth=depths[index],
                    parent_contour=parent_contour,
                    children=children_lookup[index],
                )
            )
            debug_hierarchy.append(
                {
                    "contour_id": contour_ids[index],
                    "area": area_px,
                    "bbox": [int(x), int(y), int(bbox_width), int(bbox_height)],
                    "depth": depths[index],
                    "parent": parent_contour,
                    "children": list(children_lookup[index]),
                    "touches_border": touches_border,
                    "bbox_coverage": (float(bbox_width) * float(bbox_height)) / total_area,
                }
            )

        return tuple(extracted), tuple(debug_hierarchy), contour_pixels

    def _extract_alpha_debug(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3].copy()
            _, alpha_mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
            return alpha, alpha_mask
        return None, None

    def _to_vector_points(self, points: tuple[tuple[int, int], ...]) -> tuple[Point, ...]:
        return tuple(self.coordinate_transformer.pixel_to_vector((float(point[0]), float(point[1]))) for point in points)

    def _pixel_area_to_vector_area(self, area_px: float) -> float:
        coordinate_system = self.coordinate_transformer.coordinate_system
        if coordinate_system.unit == "mm":
            scale = float(coordinate_system.scale.get("px_to_mm", 1.0))
            return float(area_px) * scale * scale
        return float(area_px)

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

    def _draw_contours_overlay(
        self,
        *,
        image: np.ndarray,
        contours_pixels: dict[str, tuple[tuple[int, int], ...]],
        contour_ids: tuple[str, ...],
        closed_lookup: dict[str, bool],
    ) -> np.ndarray:
        canvas = self._to_bgr_canvas(image)
        for index, contour_id in enumerate(contour_ids):
            pixels = contours_pixels.get(contour_id, ())
            if len(pixels) < 2:
                continue
            color = (
                int((53 * (index + 1)) % 255),
                int((97 * (index + 2)) % 255),
                int((193 * (index + 3)) % 255),
            )
            polyline = np.array(pixels, dtype=np.int32)
            cv2.polylines(canvas, [polyline], bool(closed_lookup.get(contour_id, False)), color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                contour_id,
                tuple(int(value) for value in polyline[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas

    @staticmethod
    def _to_bgr_canvas(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy()
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError("image must be grayscale, BGR, or BGRA")

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (perf_counter() - start) * 1000.0


__all__ = [
    "BinaryContour",
    "ContourExtractionDebugArtifacts",
    "ContourExtractor",
    "ExtractedContours",
    "Point",
]
