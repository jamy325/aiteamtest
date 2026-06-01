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
    closed_path_count: int = 0
    successful_step_count: int = 0
    invalid_step_count: int = 0
    recent_history: tuple[str, ...] = ()
    current_feedback: tuple[str, ...] = ()
    current_goal: str = ""
    last_action: str = ""
    allowed_next_actions: tuple[str, ...] = ()
    forbidden_next_actions: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


FREE_PEN_TOOL_NATIVE_SYSTEM_PROMPT = """
You are tracing a single black contour with a constrained headless pen tool.

Use exactly one provided function tool per round.
Use function tool calls only.
Do not write JSON manually.
Do not answer in natural language outside the tool call.
Put the short visual reason into the tool argument field named "reason".

Coordinate system:
- coordinate_space = image_px
- origin = top-left
- x increases to the right
- y increases downward

Image semantics:
- The source image is the target.
- The overlay image is your previous drawing only.
- The composite image is an auxiliary preview.
- If overlay and source conflict, the source image is always correct.
- Do not trace the overlay.

Human pen-tool tracing workflow:
- Trace like a human using a pen tool.
- Blue anchor points should stay on the black contour or be very close to it.
- Green control handles adjust tangent direction and curvature and do not need to lie on the contour.
- Start by placing anchors on the black contour.
- Each start_path point and each curve_to endpoint p should be on the target contour or very close to it.
- If the orange curve deviates from the black contour, do not blindly continue to the next segment.
- Prefer move_handle or set_segment_handles to refine curvature.
- Use move_anchor only when the anchor itself is wrong.
- After a segment is aligned, continue to the next segment.
- If the path has already returned to the start point and matches the contour, close_path before finish_trace.

Tool rules:
- line_to draws straight segments only.
- curve_to is required for curved, smooth, oval, ellipse, circle, arc, or rounded contour segments.
- move_handle edits one handle of an existing cubic segment.
- set_segment_handles edits both handles of an existing cubic segment.
- close_path only when the current point is already near the path start and the contour has been sufficiently traced.
- Follow runtime allowed_next_actions and forbidden_next_actions.
- If a tool is rejected, do not repeat the same call.
"""

FREE_PEN_TOOL_SYSTEM_PROMPT = FREE_PEN_TOOL_NATIVE_SYSTEM_PROMPT


def build_free_pen_tool_system_prompt() -> str:
    return FREE_PEN_TOOL_NATIVE_SYSTEM_PROMPT.strip()


def build_free_pen_tool_state_text(prompt_input: FreePenToolPromptInput) -> str:
    history_text = "\n".join(f"- {entry}" for entry in prompt_input.recent_history) or "- none"
    feedback_text = "\n".join(f"- {entry}" for entry in prompt_input.current_feedback) or "- none"
    allowed_text = ", ".join(prompt_input.allowed_next_actions) if prompt_input.allowed_next_actions else "none"
    forbidden_text = ", ".join(prompt_input.forbidden_next_actions) if prompt_input.forbidden_next_actions else "none"
    payload = json.dumps(prompt_input.to_payload(), ensure_ascii=True, sort_keys=True, indent=2)
    return (
        "Session state:\n"
        f"- task: continue tracing the same single black target contour\n"
        f"- mode: single_contour_pen_tracing\n"
        f"- path_count: {prompt_input.path_count}\n"
        f"- closed_path_count: {prompt_input.closed_path_count}\n"
        f"- path_open: {str(prompt_input.path_open).lower()}\n"
        f"- current_point: {prompt_input.current_point}\n"
        f"- current_subpath_start: {prompt_input.current_subpath_start}\n"
        f"- last_action: {prompt_input.last_action or 'none'}\n"
        f"- current_goal: {prompt_input.current_goal or 'continue tracing the same target contour'}\n"
        f"- allowed_next_actions: {allowed_text}\n"
        f"- forbidden_next_actions: {forbidden_text}\n\n"
        f"Recent history:\n{history_text}\n\n"
        f"Current feedback:\n{feedback_text}\n\n"
        f"Runtime input:\n{payload}"
    )


def build_free_pen_tool_prompt(prompt_input: FreePenToolPromptInput) -> str:
    return f"{build_free_pen_tool_system_prompt()}\n\n{build_free_pen_tool_state_text(prompt_input)}"


__all__ = [
    "FREE_PEN_PROMPT",
    "FREE_PEN_TOOL_SYSTEM_PROMPT",
    "FREE_PEN_TOOL_NATIVE_SYSTEM_PROMPT",
    "FreePenPromptInput",
    "FreePenToolPromptInput",
    "build_free_pen_prompt",
    "build_free_pen_tool_state_text",
    "build_free_pen_tool_system_prompt",
    "build_free_pen_tool_prompt",
]
