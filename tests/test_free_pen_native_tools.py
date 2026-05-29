from __future__ import annotations

import json

import pytest

from services.ai_adapters import SiliconFlowVisionAdapter
from services.free_pen_native_tools import ToolArgumentsParseError, parse_native_tool_call


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
