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

Image semantics:
- The source image is the target.
- The overlay image is your previous drawing only.
- Do not trace the overlay.
- If the overlay conflicts with the source image, the source image is always correct.
- Use the overlay only to understand what you have already drawn.

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
- undo_last()
- rollback_to_step(step)
- inspect_history(last_n)
- restart_path(x, y)

Rules:
- Call exactly one tool per round when you can continue drawing.
- The source image contains the real target stroke. The overlay is only your draft.
- For smooth curved strokes, prefer curve_to.
- Use line_to only for visibly straight segments.
- Do not approximate an oval, circle, or arc using many line_to calls.
- Do not call close_path unless the current point is already near the starting point.
- Do not finish while path_open=true.
- If an action is rejected, revise the action instead of repeating it.
- If the path became wrong, use undo_last or rollback_to_step.
- Use inspect_history if you need to understand recent mistakes.
- Use finish only when the tracing is complete enough and the current path is not open.
- Use stalled only when you cannot continue reliably.
- Do not output SVG.
- Do not output VectorDocument.
- Do not output AI review commands, candidates, or planner objects.
- Keep reason short and visual.

Bad sequence for an oval:
- start_path(50, 100)
- line_to(90, 152)
- close_path()

Why it is bad:
- line_to creates a straight chord instead of following a smooth curved boundary.
- close_path too early creates a triangle or long straight closure instead of a smooth oval.

Good sequence style for a smooth oval:
- start_path(50, 100)
- curve_to(c1=[55,70], c2=[120,55], p=[170,95])
- curve_to(c1=[190,120], c2=[150,175], p=[90,160])
- curve_to(c1=[45,145], c2=[35,115], p=[50,100])
- close_path()

The numeric points above are only illustrative. Do not copy them blindly. They show that smooth boundaries should usually use curve_to.

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
    recent_history: tuple[str, ...] = ()
    current_feedback: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def build_free_pen_tool_prompt(prompt_input: FreePenToolPromptInput) -> str:
    payload = json.dumps(prompt_input.to_payload(), ensure_ascii=True, sort_keys=True, indent=2)
    history_text = "\n".join(f"- {entry}" for entry in prompt_input.recent_history) or "- none"
    feedback_text = "\n".join(f"- {entry}" for entry in prompt_input.current_feedback) or "- none"
    return (
        f"{FREE_PEN_TOOL_PROMPT}\n\n"
        f"Recent history:\n{history_text}\n\n"
        f"Current feedback:\n{feedback_text}\n\n"
        f"Runtime input:\n{payload}"
    )


__all__ = [
    "FREE_PEN_PROMPT",
    "FREE_PEN_TOOL_PROMPT",
    "FreePenPromptInput",
    "FreePenToolPromptInput",
    "build_free_pen_prompt",
    "build_free_pen_tool_prompt",
]
