from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from services.free_pen_runtime import (
    load_free_pen_schema,
    load_free_pen_tool_schema,
    normalize_free_pen_response,
    normalize_free_pen_tool_response,
    validate_free_pen_response,
    validate_free_pen_tool_response,
)


def test_ai_free_pen_schema_accepts_draw_accept_and_stalled() -> None:
    draw = {
        "decision": "draw",
        "coordinate_space": "image_px",
        "segments": [
            {
                "p0": [8, 8],
                "c1": [16, 8],
                "c2": [24, 24],
                "p1": [32, 32],
            }
        ],
        "reason": "Trace the visible stroke.",
    }
    accept = {
        "decision": "accept",
        "reason": "The current tracing is acceptable.",
    }
    stalled = {
        "decision": "stalled",
        "reason": "The source image is too ambiguous.",
    }

    validate_free_pen_response(draw)
    validate_free_pen_response(accept)
    validate_free_pen_response(stalled)


def test_ai_free_pen_schema_rejects_missing_segments_and_unknown_decision() -> None:
    with pytest.raises(ValidationError):
        validate_free_pen_response(
            {
                "decision": "draw",
                "coordinate_space": "image_px",
                "reason": "Missing segments should fail.",
            }
        )

    with pytest.raises(ValidationError):
        validate_free_pen_response(
            {
                "decision": "preview",
                "reason": "Unsupported decision.",
            }
        )


def test_ai_free_pen_schema_prompt_side_helpers_stay_json_schema_compatible() -> None:
    schema = load_free_pen_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator(schema)
    json.dumps(schema)

    normalized = normalize_free_pen_response(
        {
            "decision": "draw",
            "coordinate_space": "image_px",
            "segments": [
                {
                    "p0": [1, 2],
                    "c1": [3, 4],
                    "c2": [5, 6],
                    "p1": [7, 8],
                }
            ],
            "reason": "normalized",
        }
    )
    assert normalized["decision"] == "draw"
    assert normalized["segments"][0]["p0"] == [1.0, 2.0]


def test_ai_free_pen_tool_schema_accepts_valid_tools_and_terminal_decisions() -> None:
    validate_free_pen_tool_response(
        {
            "decision": "tool_call",
            "tool_call": {
                "tool": "start_path",
                "x": 12,
                "y": 18,
            },
            "reason": "start path",
        }
    )
    validate_free_pen_tool_response(
        {
            "decision": "tool_call",
            "tool_call": {
                "tool": "line_to",
                "x": 40,
                "y": 18,
            },
            "reason": "draw line",
        }
    )
    validate_free_pen_tool_response(
        {
            "decision": "tool_call",
            "tool_call": {
                "tool": "curve_to",
                "c1": [24, 10],
                "c2": [36, 10],
                "p": [48, 18],
            },
            "reason": "draw curve",
        }
    )
    validate_free_pen_tool_response(
        {
            "decision": "tool_call",
            "tool_call": {
                "tool": "close_path",
            },
            "reason": "close",
        }
    )
    validate_free_pen_tool_response({"decision": "finish", "reason": "done"})
    validate_free_pen_tool_response({"decision": "stalled", "reason": "ambiguous"})


def test_ai_free_pen_tool_schema_rejects_missing_curve_fields_and_unknown_values() -> None:
    with pytest.raises((ValidationError, ValueError)):
        validate_free_pen_tool_response(
            {
                "decision": "tool_call",
                "tool_call": {
                    "tool": "curve_to",
                    "c1": [1, 2],
                    "p": [3, 4],
                },
                "reason": "missing c2",
            }
        )

    with pytest.raises(ValidationError):
        validate_free_pen_tool_response(
            {
                "decision": "preview",
                "reason": "unsupported decision",
            }
        )

    with pytest.raises(ValueError):
        normalize_free_pen_tool_response(
            {
                "decision": "tool_call",
                "tool_call": {
                    "tool": "arc_to",
                    "x": 1,
                    "y": 2,
                },
                "reason": "unsupported tool",
            }
        )


def test_ai_free_pen_tool_schema_prompt_helpers_stay_json_schema_compatible() -> None:
    schema = load_free_pen_tool_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator(schema)
    json.dumps(schema)

    normalized = normalize_free_pen_tool_response(
        {
            "decision": "tool_call",
            "tool_call": {
                "tool": "curve_to",
                "c1": [1, 2],
                "c2": [3, 4],
                "p": [5, 6],
            },
            "reason": "curve",
        }
    )
    assert normalized["decision"] == "tool_call"
    assert normalized["tool_call"]["c1"] == [1.0, 2.0]
