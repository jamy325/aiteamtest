from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from services.ai_adapters import create_vision_adapter
from services.free_pen_prompt import build_free_pen_prompt
from services.free_pen_runtime import FreePenReviewInput, FreePenRuntime
from vector_reconstruction.cli import main


def _write_source_image(image_path: Path) -> None:
    image = np.full((72, 96, 3), 255, dtype=np.uint8)
    cv2.line(image, (12, 52), (84, 18), (0, 0, 0), thickness=3)
    assert cv2.imwrite(str(image_path), image)


def _draw_response() -> dict[str, object]:
    return {
        "decision": "draw",
        "coordinate_space": "image_px",
        "segments": [
            {
                "p0": [12, 52],
                "c1": [24, 40],
                "c2": [60, 26],
                "p1": [84, 18],
            }
        ],
        "reason": "Single Bezier segment traces the line.",
    }


def test_free_pen_runtime_generates_transparent_png_for_single_and_multi_segment_draws(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "out"

    response = {
        "decision": "draw",
        "coordinate_space": "image_px",
        "segments": [
            {
                "p0": [12, 52],
                "c1": [24, 40],
                "c2": [42, 34],
                "p1": [56, 28],
            },
            {
                "p0": [56, 28],
                "c1": [64, 24],
                "c2": [74, 20],
                "p1": [84, 18],
            },
        ],
        "reason": "Two connected curves trace the source stroke.",
    }
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")

    runtime = FreePenRuntime(adapter=create_vision_adapter("file", response_path=response_path))
    result = runtime.run(input_path, output_dir)

    assert result.status == "drawn"
    overlay = cv2.imread(str(output_dir / "final_overlay.png"), cv2.IMREAD_UNCHANGED)
    assert overlay is not None
    assert overlay.shape == (72, 96, 4)
    assert int(np.count_nonzero(overlay[:, :, 3])) > 0
    assert int(np.count_nonzero(overlay[:, :, :3])) > 0
    assert int(np.count_nonzero(overlay[:, :, 0:3][overlay[:, :, 3] == 0])) == 0


def test_free_pen_runtime_accepts_and_stalls_without_crashing(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)

    accept_path = tmp_path / "accept.json"
    accept_path.write_text(json.dumps({"decision": "accept", "reason": "done"}), encoding="utf-8")
    accept_result = FreePenRuntime(adapter=create_vision_adapter("file", response_path=accept_path)).run(
        input_path,
        tmp_path / "accept_out",
    )
    accept_overlay = cv2.imread(str(accept_result.final_overlay_path), cv2.IMREAD_UNCHANGED)
    assert accept_result.status == "accepted"
    assert accept_overlay is not None
    assert int(np.count_nonzero(accept_overlay[:, :, 3])) == 0

    stalled_path = tmp_path / "stalled.json"
    stalled_path.write_text(json.dumps({"decision": "stalled", "reason": "ambiguous"}), encoding="utf-8")
    stalled_result = FreePenRuntime(adapter=create_vision_adapter("file", response_path=stalled_path)).run(
        input_path,
        tmp_path / "stalled_out",
    )
    stalled_overlay = cv2.imread(str(stalled_result.final_overlay_path), cv2.IMREAD_UNCHANGED)
    assert stalled_result.status == "stalled"
    assert stalled_overlay is not None
    assert int(np.count_nonzero(stalled_overlay[:, :, 3])) == 0


def test_free_pen_runtime_records_invalid_response_without_crashing(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "out"
    response_path = tmp_path / "invalid.json"
    response_path.write_text(json.dumps({"decision": "draw", "reason": "missing geometry"}), encoding="utf-8")

    runtime = FreePenRuntime(adapter=create_vision_adapter("file", response_path=response_path))
    result = runtime.run(input_path, output_dir)

    assert result.status == "invalid_response"
    assert result.error_message is not None
    recorded = json.loads((output_dir / "round_001_response.json").read_text(encoding="utf-8"))
    assert recorded["status"] == "invalid_response"
    assert recorded["error"]


def test_free_pen_prompt_stays_isolated_from_formal_ai_review_terms() -> None:
    review_input = FreePenReviewInput(
        original_image="source.png",
        overlay_image=None,
        distance_field_diff_image=None,
        canvas_width=96,
        canvas_height=72,
        round_index=1,
        max_rounds=2,
    )
    prompt = build_free_pen_prompt(review_input.prompt_input())
    assert "proposed_commands" not in prompt
    assert '"vector_document_json"' not in prompt
    assert '"candidates"' not in prompt
    assert '"available_tools"' not in prompt
    assert "image_px" in prompt


def test_free_pen_cli_uses_file_provider_and_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "out"
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(_draw_response()), encoding="utf-8")

    monkeypatch.setenv("AI_PROVIDER", "file")
    monkeypatch.setenv("AI_FILE_RESPONSE_PATH", str(response_path))
    exit_code = main(
        [
            "free-pen",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--max-rounds",
            "1",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "final_overlay.png").exists()
    assert (output_dir / "round_001_response.json").exists()


def test_free_pen_cli_writes_ai_review_interaction_log(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "out"
    log_path = tmp_path / "free_pen_ai_log.json"
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(_draw_response()), encoding="utf-8")

    monkeypatch.setenv("AI_PROVIDER", "file")
    monkeypatch.setenv("AI_FILE_RESPONSE_PATH", str(response_path))
    exit_code = main(
        [
            "free-pen",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--max-rounds",
            "1",
            "--ai-review-log-path",
            str(log_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["interaction_count"] == 1
    interaction = payload["interactions"][0]
    assert interaction["status"] == "completed"
    assert interaction["provider"] == "file"
    assert interaction["image_file_count"] == 1
    assert interaction["prompt_char_count"] > 0
    assert interaction["image_paths"] == [str(input_path)]
    assert interaction["image_upload_summary"]["mime_type"] == "image/png"
    assert interaction["image_upload_summary"]["data_url_header"] == "data:image/png;base64"
    assert interaction["image_upload_summary"]["data_url_char_count"] > interaction["image_upload_summary"]["file_size_bytes"]
    assert interaction["provider_request_content_summary"]["content"][1]["type"] == "image_file"
    assert interaction["normalized_response"]["decision"] == "draw"
