from __future__ import annotations

import json
from dataclasses import asdict, dataclass


FREE_PEN_PROMPT = """You are tracing the visible strokes in the provided source image.

Return JSON only.

Your task:
- Look at the attached source image.
- First estimate how many distinct visible shapes or stroke groups you can see in the image.
- Trace the visible drawing using cubic Bezier segments.
- Use image pixel coordinates for every point.

Before you decide:
- Confirm to yourself that an image is attached to this request.
- If an image is attached, do not say that the source image is missing.
- Mention your best visible shape or stroke-group count in the `reason` text, even if it is only an estimate.

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

If you can draw the tracing, return:
{
  "decision": "draw",
  "coordinate_space": "image_px",
  "segments": [
    {
      "p0": [x, y],
      "c1": [x, y],
      "c2": [x, y],
      "p1": [x, y]
    }
  ],
  "reason": "short reason, including the estimated visible shape count"
}

If the current tracing is already good enough, return:
{
  "decision": "accept",
  "reason": "short reason, including the estimated visible shape count"
}

If you cannot continue reliably, return:
{
  "decision": "stalled",
  "reason": "short reason, including the estimated visible shape count"
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


__all__ = ["FREE_PEN_PROMPT", "FreePenPromptInput", "build_free_pen_prompt"]
