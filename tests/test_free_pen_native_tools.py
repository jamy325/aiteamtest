from __future__ import annotations

import json

import pytest

from services.ai_adapters import SiliconFlowVisionAdapter
from services.free_pen_native_tools import (
    ToolArgumentsParseError,
    build_free_pen_native_tools_schema,
    parse_native_tool_call,
)


class DummyObj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _CompletionsStub:
    def __init__(self, response: object) -> None:
        self.response = response
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.last_kwargs = dict(kwargs)
        return self.response


class _ClientStub:
    def __init__(self, response: object) -> None:
        self.chat = DummyObj(completions=_CompletionsStub(response))


class _ToolReviewInput:
    original_image = None
    overlay_image = None
    distance_field_diff_image = None
    tool_mode = "native_tools"
    tools = [{"type": "function", "function": {"name": "finish_trace"}}]
    tool_choice = "none"
    messages = (
        {"role": "system", "content": "system"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    )


def test_siliconflow_tools_request_requires_tool_calls() -> None:
    response = DummyObj(
        choices=[
            DummyObj(
                message=DummyObj(content='{"decision":"stalled","reason":"fallback"}', tool_calls=None),
                finish_reason="stop",
            )
        ]
    )
    adapter = SiliconFlowVisionAdapter(client=_ClientStub(response))

    with pytest.raises(ValueError, match="missing_native_tool_calls"):
        adapter.review("prompt", _ToolReviewInput())


def test_siliconflow_tools_request_does_not_parse_reasoning_content_json() -> None:
    response = DummyObj(
        choices=[
            DummyObj(
                message=DummyObj(reasoning_content='{"decision":"stalled","reason":"fallback"}', tool_calls=None),
                finish_reason="stop",
            )
        ]
    )
    adapter = SiliconFlowVisionAdapter(client=_ClientStub(response))

    with pytest.raises(ValueError, match="missing_native_tool_calls"):
        adapter.review("prompt", _ToolReviewInput())


def test_parse_native_close_path_reason_top_level() -> None:
    result = parse_native_tool_call("close_path", {"reason": "near start"})
    assert result == {
        "decision": "tool_call",
        "tool_call": {"tool": "close_path"},
        "reason": "near start",
    }


def test_parse_native_curve_to_requires_reason() -> None:
    with pytest.raises(ToolArgumentsParseError, match="reason"):
        parse_native_tool_call("curve_to", {"c1": [1, 2], "c2": [3, 4], "p": [5, 6]})


def test_parse_native_tool_call_accepts_json_string_arguments() -> None:
    result = parse_native_tool_call(
        "start_path",
        json.dumps({"x": 12, "y": 52, "reason": "start at contour"}, ensure_ascii=False),
    )
    assert result == {
        "decision": "tool_call",
        "tool_call": {"tool": "start_path", "x": 12, "y": 52},
        "reason": "start at contour",
    }


def test_native_tools_schema_contains_handle_edit_tools() -> None:
    tool_names = {
        entry["function"]["name"]
        for entry in build_free_pen_native_tools_schema()
        if entry.get("type") == "function"
    }
    assert {"move_anchor", "move_handle", "set_segment_handles"} <= tool_names


def test_native_tools_schema_contains_convert_line_to_curve() -> None:
    tool_names = {
        entry["function"]["name"]
        for entry in build_free_pen_native_tools_schema()
        if entry.get("type") == "function"
    }
    assert "convert_line_to_curve" in tool_names


def test_native_tools_schema_contains_restore_best_segment() -> None:
    tool_names = {
        entry["function"]["name"]
        for entry in build_free_pen_native_tools_schema()
        if entry.get("type") == "function"
    }
    assert "restore_best_segment" in tool_names


def test_native_tools_schema_contains_request_zoom_tools() -> None:
    tool_names = {
        entry["function"]["name"]
        for entry in build_free_pen_native_tools_schema()
        if entry.get("type") == "function"
    }
    assert {"request_segment_zoom", "request_zoom_window"} <= tool_names


def test_parse_native_move_anchor() -> None:
    result = parse_native_tool_call(
        "move_anchor",
        {"anchor_id": "A2", "x": 88, "y": 60, "reason": "move anchor onto contour"},
    )
    assert result == {
        "decision": "tool_call",
        "tool_call": {"tool": "move_anchor", "anchor_id": "A2", "x": 88.0, "y": 60.0},
        "reason": "move anchor onto contour",
    }


def test_parse_native_move_handle() -> None:
    result = parse_native_tool_call(
        "move_handle",
        {"segment_id": "S1", "handle": "c1", "x": 70, "y": 44, "reason": "adjust tangent"},
    )
    assert result == {
        "decision": "tool_call",
        "tool_call": {"tool": "move_handle", "segment_id": "S1", "handle": "c1", "x": 70.0, "y": 44.0},
        "reason": "adjust tangent",
    }


def test_parse_native_set_segment_handles() -> None:
    result = parse_native_tool_call(
        "set_segment_handles",
        {"segment_id": "S2", "c1": [120, 44], "c2": [155, 66], "reason": "refine both handles"},
    )
    assert result == {
        "decision": "tool_call",
        "tool_call": {
            "tool": "set_segment_handles",
            "segment_id": "S2",
            "c1": [120.0, 44.0],
            "c2": [155.0, 66.0],
        },
        "reason": "refine both handles",
    }


def test_parse_native_convert_line_to_curve() -> None:
    result = parse_native_tool_call(
        "convert_line_to_curve",
        {"segment_id": "S3", "c1": [800, 258], "c2": [1000, 258], "reason": "make editable"},
    )
    assert result == {
        "decision": "tool_call",
        "tool_call": {
            "tool": "convert_line_to_curve",
            "segment_id": "S3",
            "c1": [800, 258],
            "c2": [1000, 258],
        },
        "reason": "make editable",
    }


def test_parse_native_restore_best_segment() -> None:
    result = parse_native_tool_call(
        "restore_best_segment",
        {"segment_id": "S3", "reason": "restore best"},
    )
    assert result == {
        "decision": "tool_call",
        "tool_call": {
            "tool": "restore_best_segment",
            "segment_id": "S3",
        },
        "reason": "restore best",
    }


def test_parse_native_request_segment_zoom() -> None:
    result = parse_native_tool_call(
        "request_segment_zoom",
        {"segment_id": "S3", "zoom_scale": 4, "padding_px": 100, "reason": "Need a closer view."},
    )
    assert result == {
        "decision": "tool_call",
        "tool_call": {
            "tool": "request_segment_zoom",
            "segment_id": "S3",
            "zoom_scale": 4,
            "padding_px": 100,
        },
        "reason": "Need a closer view.",
    }


def test_parse_native_request_zoom_window() -> None:
    result = parse_native_tool_call(
        "request_zoom_window",
        {"x": 1080, "y": 80, "width": 360, "height": 260, "zoom_scale": 4, "reason": "Need this local region."},
    )
    assert result == {
        "decision": "tool_call",
        "tool_call": {
            "tool": "request_zoom_window",
            "x": 1080,
            "y": 80,
            "width": 360,
            "height": 260,
            "zoom_scale": 4,
        },
        "reason": "Need this local region.",
    }
