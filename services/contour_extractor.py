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
    foreground_mode: str = "light_on_dark"
    foreground_reason: str = ""
    filtered_binary_contours: tuple[dict[str, object], ...] = ()
    filtered_skeleton_contours: tuple[dict[str, object], ...] = ()


class ContourExtractor:
    def __init__(
        self,
        threshold: int = 127,
        blur_kernel_size: int = 5,
        morphology_kernel_size: int = 3,
        foreground_mode: str = "auto",
        border_margin: int = 0,
        max_bbox_coverage: float = 0.98,
        enable_page_border_filter: bool = True,
        coordinate_transformer: CoordinateTransformer | None = None,
        skeleton_graph_tracer: SkeletonGraphTracer | None = None,
    ) -> None:
        self.threshold = threshold
        self.blur_kernel_size = blur_kernel_size
        self.morphology_kernel_size = morphology_kernel_size
        self.foreground_mode = foreground_mode
        self.border_margin = int(border_margin)
        self.max_bbox_coverage = float(max_bbox_coverage)
        self.enable_page_border_filter = bool(enable_page_border_filter)
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
        threshold_binary, denoised, closed, selected_mode, selected_reason = self._preprocess_binary_stages(
            grayscale,
            alpha_mask=alpha_mask,
        )
        timings_ms["binary_preprocess"] = self._elapsed_ms(start)

        start = perf_counter()
        (
            binary_contours,
            binary_hierarchy,
            contours_pixels,
            binary_overlay_ids,
            binary_closed_lookup,
            filtered_binary_contours,
        ) = self._extract_binary_from_closed_mask(closed, image.shape[:2])
        timings_ms["binary_contours"] = self._elapsed_ms(start)

        start = perf_counter()
        skeleton_binary = self._preprocess_skeleton_mask(
            image,
            foreground_mode=selected_mode,
            alpha_mask=alpha_mask,
        )
        skeleton_mask = self._skeletonize(skeleton_binary)
        trace_result = self.skeleton_graph_tracer.trace_graph(skeleton_mask)
        skeleton_contours, skeleton_overlay_pixels, skeleton_closed_lookup, filtered_skeleton_contours = (
            self._extract_skeleton_contours_from_trace(trace_result, image.shape[:2])
        )
        timings_ms["skeleton"] = self._elapsed_ms(start)

        binary_overlay = self._draw_contours_overlay(
            image=image,
            contours_pixels=contours_pixels,
            contour_ids=binary_overlay_ids,
            closed_lookup=binary_closed_lookup,
        )
        skeleton_overlay = self._draw_contours_overlay(
            image=image,
            contours_pixels=skeleton_overlay_pixels,
            contour_ids=tuple(skeleton_overlay_pixels),
            closed_lookup=skeleton_closed_lookup,
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
            threshold_polarity=selected_mode,
            foreground_mode=selected_mode,
            foreground_reason=selected_reason,
            filtered_binary_contours=filtered_binary_contours,
            filtered_skeleton_contours=filtered_skeleton_contours,
        )
        return extracted, debug

    def extract_binary_contours(self, image: np.ndarray) -> tuple[BinaryContour, ...]:
        grayscale = self._to_grayscale(image)
        _, _, closed, _, _ = self._preprocess_binary_stages(grayscale)
        extracted, _, _, _, _, _ = self._extract_binary_from_closed_mask(closed, image.shape[:2])
        return extracted

    def extract_skeleton_contours(self, image: np.ndarray) -> tuple[BinaryContour, ...]:
        trace_result = self._trace_skeleton_graph(image)
        contours, _, _, _ = self._extract_skeleton_contours_from_trace(trace_result, image.shape[:2])
        return contours

    def _trace_skeleton_graph(self, image: np.ndarray) -> SkeletonGraphTraceResult:
        grayscale = self._to_grayscale(image)
        alpha, alpha_mask = self._extract_alpha_debug(image)
        selected_mode = self._select_foreground_mode(grayscale, alpha_mask=alpha_mask)[0]
        skeleton_binary = self._preprocess_skeleton_mask(
            image,
            foreground_mode=selected_mode,
            alpha_mask=alpha_mask,
        )
        skeleton_mask = self._skeletonize(skeleton_binary)
        return self.skeleton_graph_tracer.trace_graph(skeleton_mask)

    def _extract_skeleton_contours_from_trace(
        self,
        trace_result: SkeletonGraphTraceResult,
        image_shape: tuple[int, int],
    ) -> tuple[
        tuple[BinaryContour, ...],
        dict[str, tuple[tuple[int, int], ...]],
        dict[str, bool],
        tuple[dict[str, object], ...],
    ]:
        extracted: list[BinaryContour] = []
        overlay_pixels: dict[str, tuple[tuple[int, int], ...]] = {}
        closed_lookup: dict[str, bool] = {}
        filtered: list[dict[str, object]] = []
        height, width = image_shape
        total_area = float(width * height) if width > 0 and height > 0 else 1.0
        for index, traced_path in enumerate(trace_result.paths):
            if len(traced_path.pixels) < 2:
                continue
            contour_id = f"skeleton_contour_{index}"
            x_values = [int(pixel[0]) for pixel in traced_path.pixels]
            y_values = [int(pixel[1]) for pixel in traced_path.pixels]
            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)
            bbox_width = (max_x - min_x) + 1
            bbox_height = (max_y - min_y) + 1
            bbox_coverage = (float(bbox_width) * float(bbox_height)) / total_area
            touches_border = min_x <= 0 or min_y <= 0 or max_x >= (width - 1) or max_y >= (height - 1)
            filter_reason = self._page_border_filter_reason(
                touches_border=touches_border,
                bbox_coverage=bbox_coverage,
            )
            overlay_pixels[contour_id] = tuple((int(pixel[0]), int(pixel[1])) for pixel in traced_path.pixels)
            closed_lookup[contour_id] = traced_path.closed
            if filter_reason is not None:
                filtered.append(
                    {
                        "contour_id": contour_id,
                        "reason": filter_reason,
                        "touches_border": touches_border,
                        "bbox_coverage": bbox_coverage,
                    }
                )
                continue

            points = self._to_vector_points(traced_path.pixels)
            extracted.append(
                BinaryContour(
                    contour_id=contour_id,
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

        return tuple(extracted), overlay_pixels, closed_lookup, tuple(filtered)

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
        _, _, closed, _, _ = self._preprocess_binary_stages(grayscale)
        return closed

    def _preprocess_skeleton_mask(
        self,
        image: np.ndarray,
        *,
        foreground_mode: str | None = None,
        alpha_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        grayscale = self._to_grayscale(image)
        selected_mode = foreground_mode or self._select_foreground_mode(grayscale, alpha_mask=alpha_mask)[0]
        if selected_mode == "alpha_foreground":
            return alpha_mask.copy() if alpha_mask is not None else np.zeros_like(grayscale)
        threshold_type = cv2.THRESH_BINARY_INV if selected_mode == "dark_on_light" else cv2.THRESH_BINARY
        _, binary = cv2.threshold(grayscale, self.threshold, 255, threshold_type)
        return binary

    def _preprocess_binary_stages(
        self,
        grayscale: np.ndarray,
        *,
        alpha_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
        blurred = cv2.GaussianBlur(grayscale, (self.blur_kernel_size, self.blur_kernel_size), 0)
        binary_mode, reason = self._select_foreground_mode(blurred, alpha_mask=alpha_mask)
        if binary_mode == "alpha_foreground":
            binary = alpha_mask.copy() if alpha_mask is not None else np.zeros_like(grayscale)
        elif binary_mode == "dark_on_light":
            _, binary = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY)
        denoised = cv2.medianBlur(binary, 3)
        kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), dtype=np.uint8)
        closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        return binary, denoised, closed, binary_mode, reason

    def _extract_binary_from_closed_mask(
        self,
        closed: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[
        tuple[BinaryContour, ...],
        tuple[dict[str, object], ...],
        dict[str, tuple[tuple[int, int], ...]],
        tuple[str, ...],
        dict[str, bool],
        tuple[dict[str, object], ...],
    ]:
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return (), (), {}, (), {}, ()

        height, width = image_shape
        hierarchy_data = hierarchy[0]
        contour_ids = tuple(f"binary_contour_{index}" for index in range(len(contours)))
        depths = self._compute_depths(hierarchy_data)
        children_lookup = self._build_children_lookup(hierarchy_data, contour_ids)

        extracted: list[BinaryContour] = []
        debug_hierarchy: list[dict[str, object]] = []
        contour_pixels: dict[str, tuple[tuple[int, int], ...]] = {}
        closed_lookup: dict[str, bool] = {}
        filtered: list[dict[str, object]] = []
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
            bbox_coverage = (float(bbox_width) * float(bbox_height)) / total_area
            filter_reason = self._page_border_filter_reason(
                touches_border=touches_border,
                bbox_coverage=bbox_coverage,
            )
            closed_lookup[contour_ids[index]] = len(points) >= 3
            if filter_reason is None:
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
            else:
                filtered.append(
                    {
                        "contour_id": contour_ids[index],
                        "reason": filter_reason,
                        "touches_border": touches_border,
                        "bbox_coverage": bbox_coverage,
                    }
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
                    "bbox_coverage": bbox_coverage,
                    "filtered": filter_reason is not None,
                    "filter_reason": filter_reason,
                }
            )

        return (
            tuple(extracted),
            tuple(debug_hierarchy),
            contour_pixels,
            contour_ids,
            closed_lookup,
            tuple(filtered),
        )

    def _extract_alpha_debug(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3].copy()
            _, alpha_mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
            return alpha, alpha_mask
        return None, None

    def _select_foreground_mode(
        self,
        grayscale: np.ndarray,
        *,
        alpha_mask: np.ndarray | None = None,
    ) -> tuple[str, str]:
        configured = str(self.foreground_mode).strip().lower()
        if configured == "alpha_foreground" and alpha_mask is not None:
            return "alpha_foreground", "used explicit alpha foreground mode"
        if configured in {"dark_on_light", "light_on_dark"}:
            return configured, f"used explicit foreground_mode={configured}"

        _, light_mask = cv2.threshold(grayscale, self.threshold, 255, cv2.THRESH_BINARY)
        _, dark_mask = cv2.threshold(grayscale, self.threshold, 255, cv2.THRESH_BINARY_INV)
        light_border_ratio, light_foreground_ratio = self._mask_statistics(light_mask)
        dark_border_ratio, dark_foreground_ratio = self._mask_statistics(dark_mask)

        if dark_border_ratio < light_border_ratio:
            return "dark_on_light", (
                f"auto selected dark_on_light because border_foreground_ratio={dark_border_ratio:.4f} "
                f"< light_on_dark={light_border_ratio:.4f}"
            )
        if light_border_ratio < dark_border_ratio:
            return "light_on_dark", (
                f"auto selected light_on_dark because border_foreground_ratio={light_border_ratio:.4f} "
                f"< dark_on_light={dark_border_ratio:.4f}"
            )
        if dark_foreground_ratio < light_foreground_ratio:
            return "dark_on_light", (
                f"auto selected dark_on_light on foreground_ratio tie-break "
                f"({dark_foreground_ratio:.4f} < {light_foreground_ratio:.4f})"
            )
        return "light_on_dark", (
            f"auto selected light_on_dark on foreground_ratio tie-break "
            f"({light_foreground_ratio:.4f} <= {dark_foreground_ratio:.4f})"
        )

    def _mask_statistics(self, mask: np.ndarray) -> tuple[float, float]:
        foreground = mask > 0
        border = self._border_mask(mask.shape[:2])
        border_pixels = int(border.sum())
        border_foreground_ratio = float((foreground & border).sum()) / float(border_pixels) if border_pixels else 0.0
        foreground_ratio = float(foreground.sum()) / float(mask.shape[0] * mask.shape[1]) if mask.size else 0.0
        return border_foreground_ratio, foreground_ratio

    def _border_mask(self, image_shape: tuple[int, int]) -> np.ndarray:
        height, width = image_shape
        margin = max(0, self.border_margin)
        border = np.zeros((height, width), dtype=bool)
        if height == 0 or width == 0:
            return border
        border[: margin + 1, :] = True
        border[max(0, height - margin - 1) :, :] = True
        border[:, : margin + 1] = True
        border[:, max(0, width - margin - 1) :] = True
        return border

    def _page_border_filter_reason(self, *, touches_border: bool, bbox_coverage: float) -> str | None:
        if not self.enable_page_border_filter:
            return None
        if touches_border and bbox_coverage >= self.max_bbox_coverage:
            return "page_border_filter"
        return None

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
