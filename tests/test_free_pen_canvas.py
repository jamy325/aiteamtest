from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from services.free_pen_canvas import FreePenCanvasError, FreePenCanvasState


def _write_source_image(image_path: Path) -> np.ndarray:
    image = np.full((72, 96, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (48, 36), (24, 14), 0, 0, 360, (0, 0, 0), thickness=2)
    assert cv2.imwrite(str(image_path), image)
    return image


def test_free_pen_canvas_start_path_and_line_to_render_overlay(tmp_path: Path) -> None:
    canvas = FreePenCanvasState(width=96, height=72)
    canvas.apply_tool_call({"tool": "start_path", "x": 10, "y": 10})
    canvas.apply_tool_call({"tool": "line_to", "x": 80, "y": 10})
    overlay = canvas.render_overlay()

    assert overlay.shape == (72, 96, 4)
    assert int(np.count_nonzero(overlay[:, :, 3])) > 0
    payload = canvas.paths_payload()
    assert payload["paths"][0]["segments"][0]["type"] == "move"
    assert payload["paths"][0]["segments"][1]["type"] == "line"


def test_free_pen_canvas_curve_to_render_overlay_and_paths_payload(tmp_path: Path) -> None:
    canvas = FreePenCanvasState(width=96, height=72)
    canvas.apply_tool_call({"tool": "start_path", "x": 12, "y": 52})
    canvas.apply_tool_call({"tool": "curve_to", "c1": [24, 40], "c2": [60, 26], "p": [84, 18]})
    overlay = canvas.render_overlay()

    assert int(np.count_nonzero(overlay[:, :, 3])) > 0
    payload = canvas.paths_payload()
    assert payload["paths"][0]["segments"][1]["type"] == "cubic"


def test_free_pen_canvas_close_path_then_start_second_path() -> None:
    canvas = FreePenCanvasState(width=96, height=72)
    canvas.apply_tool_call({"tool": "start_path", "x": 10, "y": 10})
    canvas.apply_tool_call({"tool": "line_to", "x": 40, "y": 10})
    canvas.apply_tool_call({"tool": "close_path"})
    assert canvas.path_open is False
    canvas.apply_tool_call({"tool": "start_path", "x": 20, "y": 20})
    assert len(canvas.paths) == 2


def test_free_pen_canvas_rejects_non_finite_and_out_of_bounds_coordinates() -> None:
    canvas = FreePenCanvasState(width=96, height=72)
    with pytest.raises(FreePenCanvasError):
        canvas.apply_tool_call({"tool": "start_path", "x": float("nan"), "y": 10})

    canvas.apply_tool_call({"tool": "start_path", "x": 10, "y": 10})
    with pytest.raises(FreePenCanvasError):
        canvas.apply_tool_call({"tool": "line_to", "x": 1000, "y": 10})


def test_free_pen_canvas_render_composite_matches_source_size(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    source = _write_source_image(image_path)
    canvas = FreePenCanvasState(width=96, height=72)
    canvas.apply_tool_call({"tool": "start_path", "x": 24, "y": 36})
    canvas.apply_tool_call({"tool": "curve_to", "c1": [24, 20], "c2": [72, 20], "p": [72, 36]})
    composite = canvas.render_composite(source)

    assert composite.shape == (72, 96, 4)
    assert int(np.count_nonzero(composite[:, :, :3])) > 0
