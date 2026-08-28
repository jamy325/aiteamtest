from __future__ import annotations

import json
from typing import Any


class ToolArgumentsParseError(ValueError):
    """Raised when a native tool call cannot be parsed into canonical FreePen response data."""


_ALLOWED_TOOL_NAMES = (
    "start_path",
    "line_to",
    "curve_to",
    "convert_line_to_curve",
    "restore_best_segment",
    "request_segment_zoom",
    "request_zoom_window",
    "move_anchor",
    "move_handle",
    "set_segment_handles",
    "close_path",
    "undo_last",
    "rollback_to_step",
    "inspect_history",
    "restart_path",
    "finish_trace",
    "stalled",
)


def build_free_pen_native_tools_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "start_path",
                "description": "Start a new open path at a target contour point. Use only when path_open=false.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["x", "y", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "line_to",
                "description": "Draw one straight line segment from current_point to [x,y]. Use only for clearly straight target segments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["x", "y", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "curve_to",
                "description": "Draw one cubic Bezier curve from current_point to endpoint p. Required for smooth, oval, ellipse, circle, arc, rounded, or curved target segments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "c1": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "c2": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "p": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "reason": {"type": "string"},
                    },
                    "required": ["c1", "c2", "p", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_line_to_curve",
                "description": "Convert an existing line segment into a cubic Bezier segment so it can be refined with control handles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "string"},
                        "c1": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "c2": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "reason": {"type": "string"},
                    },
                    "required": ["segment_id", "c1", "c2", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "restore_best_segment",
                "description": "Restore an existing segment to the best-known geometry recorded by runtime quality tracking.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["segment_id", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "request_segment_zoom",
                "description": "Request a magnified visual crop around an existing segment. This is an inspection-only tool and does not modify the path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "string"},
                        "zoom_scale": {"type": "number", "minimum": 2, "maximum": 6},
                        "padding_px": {"type": "number", "minimum": 20, "maximum": 200},
                        "reason": {"type": "string"},
                    },
                    "required": ["segment_id", "zoom_scale", "padding_px", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "request_zoom_window",
                "description": "Request a magnified crop for a custom original-image coordinate window. This is an inspection-only tool and does not modify the path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "width": {"type": "number", "minimum": 80, "maximum": 600},
                        "height": {"type": "number", "minimum": 80, "maximum": 600},
                        "zoom_scale": {"type": "number", "minimum": 2, "maximum": 6},
                        "reason": {"type": "string"},
                    },
                    "required": ["x", "y", "width", "height", "zoom_scale", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move_anchor",
                "description": "Move an existing anchor point. Use when an anchor is not on the black source contour.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anchor_id": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["anchor_id", "x", "y", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move_handle",
                "description": "Move one control handle of an existing cubic segment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "string"},
                        "handle": {"type": "string", "enum": ["c1", "c2"]},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["segment_id", "handle", "x", "y", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_segment_handles",
                "description": "Set both control handles of an existing cubic segment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "string"},
                        "c1": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "c2": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "reason": {"type": "string"},
                    },
                    "required": ["segment_id", "c1", "c2", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "close_path",
                "description": "Close the current open path back to its start point only when current_point is already near the start point.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "undo_last",
                "description": "Undo the most recent successful drawing action.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rollback_to_step",
                "description": "Restore the canvas to a previous successful drawing step.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer", "minimum": 0},
                        "reason": {"type": "string"},
                    },
                    "required": ["step", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_history",
                "description": "Inspect recent drawing history without changing the canvas.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "last_n": {"type": "integer", "minimum": 1, "maximum": 20},
                        "reason": {"type": "string"},
                    },
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "restart_path",
                "description": "Restart the current path from [x,y] after discarding the wrong path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["x", "y", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish_trace",
                "description": "Finish tracing when the contour is complete enough and path_open=false.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stalled",
                "description": "Stop when you cannot continue reliably.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def parse_native_tool_call(name: str, arguments: str | dict[str, Any]) -> dict[str, Any]:
    normalized_name = str(name or "").strip()
    if normalized_name not in _ALLOWED_TOOL_NAMES:
        raise ValueError(f"Unknown native tool name: {normalized_name}")

    parsed_args = _parse_arguments(arguments)
    reason = _require_reason(parsed_args)

    if normalized_name == "start_path":
        return _tool_call_response("start_path", {"x": _require_key(parsed_args, "x"), "y": _require_key(parsed_args, "y")}, reason)
    if normalized_name == "line_to":
        return _tool_call_response("line_to", {"x": _require_key(parsed_args, "x"), "y": _require_key(parsed_args, "y")}, reason)
    if normalized_name == "curve_to":
        return _tool_call_response(
            "curve_to",
            {
                "c1": _require_key(parsed_args, "c1"),
                "c2": _require_key(parsed_args, "c2"),
                "p": _require_key(parsed_args, "p"),
            },
            reason,
        )
    if normalized_name == "convert_line_to_curve":
        return _tool_call_response(
            "convert_line_to_curve",
            {
                "segment_id": _require_key(parsed_args, "segment_id"),
                "c1": _require_key(parsed_args, "c1"),
                "c2": _require_key(parsed_args, "c2"),
            },
            reason,
        )
    if normalized_name == "restore_best_segment":
        return _tool_call_response(
            "restore_best_segment",
            {"segment_id": _require_key(parsed_args, "segment_id")},
            reason,
        )
    if normalized_name == "request_segment_zoom":
        return _tool_call_response(
            "request_segment_zoom",
            {
                "segment_id": _require_key(parsed_args, "segment_id"),
                "zoom_scale": _require_key(parsed_args, "zoom_scale"),
                "padding_px": _require_key(parsed_args, "padding_px"),
            },
            reason,
        )
    if normalized_name == "request_zoom_window":
        return _tool_call_response(
            "request_zoom_window",
            {
                "x": _require_key(parsed_args, "x"),
                "y": _require_key(parsed_args, "y"),
                "width": _require_key(parsed_args, "width"),
                "height": _require_key(parsed_args, "height"),
                "zoom_scale": _require_key(parsed_args, "zoom_scale"),
            },
            reason,
        )
    if normalized_name == "move_anchor":
        return _tool_call_response(
            "move_anchor",
            {
                "anchor_id": _require_key(parsed_args, "anchor_id"),
                "x": _require_key(parsed_args, "x"),
                "y": _require_key(parsed_args, "y"),
            },
            reason,
        )
    if normalized_name == "move_handle":
        return _tool_call_response(
            "move_handle",
            {
                "segment_id": _require_key(parsed_args, "segment_id"),
                "handle": _require_key(parsed_args, "handle"),
                "x": _require_key(parsed_args, "x"),
                "y": _require_key(parsed_args, "y"),
            },
            reason,
        )
    if normalized_name == "set_segment_handles":
        return _tool_call_response(
            "set_segment_handles",
            {
                "segment_id": _require_key(parsed_args, "segment_id"),
                "c1": _require_key(parsed_args, "c1"),
                "c2": _require_key(parsed_args, "c2"),
            },
            reason,
        )
    if normalized_name == "close_path":
        return _tool_call_response("close_path", {}, reason)
    if normalized_name == "undo_last":
        return _tool_call_response("undo_last", {}, reason)
    if normalized_name == "rollback_to_step":
        return _tool_call_response("rollback_to_step", {"step": _require_key(parsed_args, "step")}, reason)
    if normalized_name == "inspect_history":
        payload = {}
        if "last_n" in parsed_args:
            payload["last_n"] = parsed_args["last_n"]
        return _tool_call_response("inspect_history", payload, reason)
    if normalized_name == "restart_path":
        return _tool_call_response("restart_path", {"x": _require_key(parsed_args, "x"), "y": _require_key(parsed_args, "y")}, reason)
    if normalized_name == "finish_trace":
        return {"decision": "finish", "reason": reason}
    if normalized_name == "stalled":
        return {"decision": "stalled", "reason": reason}
    raise ValueError(f"Unknown native tool name: {normalized_name}")


def _parse_arguments(arguments: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolArgumentsParseError(f"Failed to parse tool call arguments as JSON: {arguments}") from exc
    elif isinstance(arguments, dict):
        parsed = dict(arguments)
    else:
        raise ToolArgumentsParseError(f"Unsupported tool call arguments type: {type(arguments)}")
    if not isinstance(parsed, dict):
        raise ToolArgumentsParseError(f"Tool call arguments must be a JSON object, got: {type(parsed)}")
    return parsed


def _require_reason(parsed_args: dict[str, Any]) -> str:
    reason = parsed_args.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ToolArgumentsParseError("Tool call 'reason' must be a non-empty string.")
    return reason.strip()


def _require_key(parsed_args: dict[str, Any], key: str) -> Any:
    if key not in parsed_args:
        raise ToolArgumentsParseError(f"Tool call arguments missing required field: {key}")
    return parsed_args[key]


def _tool_call_response(tool: str, payload: dict[str, Any], reason: str) -> dict[str, Any]:
    tool_call = {"tool": tool}
    tool_call.update(payload)
    return {
        "decision": "tool_call",
        "tool_call": tool_call,
        "reason": reason,
    }


__all__ = [
    "ToolArgumentsParseError",
    "build_free_pen_native_tools_schema",
    "parse_native_tool_call",
]
