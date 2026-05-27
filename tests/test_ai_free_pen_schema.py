from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from services.free_pen_runtime import load_free_pen_schema, normalize_free_pen_response, validate_free_pen_response


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
