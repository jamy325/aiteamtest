from __future__ import annotations

import json
from dataclasses import asdict, dataclass


FREE_PEN_PROMPT = """You are tracing the visible strokes in the provided source image.

Return JSON only.

Your task:
- Look at the attached source image.
- Produce a tracing attempt, not a review.
- Trace the visible drawing by directly returning cubic Bezier control points.
- Use image pixel coordinates for every point.
- Prefer a simple coarse tracing over refusing to draw.

Before you decide:
- Confirm to yourself that an image is attached to this request.
- If an image is attached, do not say that the source image is missing.
- Your default behavior should be to return `decision="draw"` with one or more cubic Bezier segments.
- On round 1, if any visible stroke or outline exists, return `decision="draw"`.
- On round 1, return at least one Bezier segment when you can see a simple ellipse, circle, arc, line, or outline.
- If the image is complex, still return your best partial `draw` result instead of `stalled`.
- Only return `stalled` when the image is genuinely unreadable or no visible tracing target exists.
- Mention a short visual summary in the `reason` text.

Coordinate system:
- coordinate_space: image_px
- origin: top-left
- x increases to the right
- y increases downward
- canvas size equals the source image size

Hard rules:
- Do not output SVG.
- Do not output VectorDocument.
- Do not output candidate, command, or review-planner objects.
- Do not describe a reconstruction workflow.
- Only return one JSON object with a `decision` field.
- Do not output explanation outside JSON.

If you can draw the tracing, return:
{
  "decision": "draw",
  "coordinate_space": "image_px",
  "segments": [
    {
      "p0": [start_x, start_y],
      "c1": [control1_x, control1_y],
      "c2": [control2_x, control2_y],
      "p1": [end_x, end_y]
    },
    {
      "p0": [x, y],
      "c1": [x, y],
      "c2": [x, y],
      "p1": [x, y]
    }
  ],
  "reason": "short visual reason"
}

If you cannot continue reliably, return:
{
  "decision": "stalled",
  "reason": "short visual reason"
}
"""


@dataclass(frozen=True, slots=True)
class FreePenPromptInput:
    canvas_width: int
    canvas_height: int
    round_index: int
    max_rounds: int
    coordinate_space: str = "image_px"
    previous_overlay_available: bool = False

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def build_free_pen_prompt(prompt_input: FreePenPromptInput) -> str:
    payload = json.dumps(prompt_input.to_payload(), ensure_ascii=True, sort_keys=True, indent=2)
    return f"{FREE_PEN_PROMPT}\n\nRuntime input:\n{payload}"

FREE_PEN_TOOL_PROMPT = """
You are tracing the visible strokes in the provided source image.

Return JSON only.

You are not reviewing. You are drawing by calling exactly one tool per round.

Coordinate system:
- coordinate_space: image_px
- origin: top-left
- x increases to the right
- y increases downward
- canvas size equals the source image size

Allowed decisions:
- tool_call
- finish
- stalled

Allowed tools:
- start_path(x, y)
- line_to(x, y)
- curve_to(c1, c2, p)
- close_path()

Rules:
- Call exactly one tool per round when you can continue drawing.
- Use finish only when the tracing is complete enough.
- Use stalled only when you cannot continue reliably.
- Do not output SVG.
- Do not output VectorDocument.
- Do not output AI review commands, candidates, or planner objects.
- Keep reason short and visual.

If you can continue drawing, return:
{
  "decision": "tool_call",
  "tool_call": {
    "tool": "curve_to",
    "c1": [x, y],
    "c2": [x, y],
    "p": [x, y]
  },
  "reason": "short visual reason"
}

If tracing is complete enough, return:
{
  "decision": "finish",
  "reason": "short visual reason"
}

If you cannot continue reliably, return:
{
  "decision": "stalled",
  "reason": "short visual reason"
}
"""


@dataclass(frozen=True, slots=True)
class FreePenToolPromptInput:
    canvas_width: int
    canvas_height: int
    step_index: int
    max_steps: int
    coordinate_space: str = "image_px"
    previous_overlay_available: bool = False
    path_open: bool = False
    current_point: list[float] | None = None
    current_subpath_start: list[float] | None = None
    path_count: int = 0
    successful_step_count: int = 0
    invalid_step_count: int = 0

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def build_free_pen_tool_prompt(prompt_input: FreePenToolPromptInput) -> str:
    payload = json.dumps(prompt_input.to_payload(), ensure_ascii=True, sort_keys=True, indent=2)
    return f"{FREE_PEN_TOOL_PROMPT}\n\nRuntime input:\n{payload}"


__all__ = [
    "FREE_PEN_PROMPT",
    "FREE_PEN_TOOL_PROMPT",
    "FreePenPromptInput",
    "FreePenToolPromptInput",
    "build_free_pen_prompt",
    "build_free_pen_tool_prompt",
]
