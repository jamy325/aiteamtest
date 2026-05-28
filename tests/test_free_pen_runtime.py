from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from services.ai_adapters import create_vision_adapter
from services.free_pen_prompt import build_free_pen_prompt
from services.free_pen_runtime import (
    FileSequenceFreePenAdapter,
    FreePenImageTransportConfig,
    FreePenReviewInput,
    FreePenRuntime,
    FreePenToolRuntime,
)
from vector_reconstruction.cli import main


def _write_source_image(image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
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
    assert '"decision": "draw"' in prompt
    assert '"decision": "stalled"' in prompt
    assert '"decision": "accept"' not in prompt


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
    assert interaction["provider_request_content_summary"]["content"][0]["type"] == "image_file"
    assert interaction["provider_request_content_summary"]["content"][-1]["type"] == "text"
    assert interaction["normalized_response"]["decision"] == "draw"


def test_free_pen_cli_prints_raw_provider_response_to_stderr(tmp_path: Path, monkeypatch, capsys) -> None:
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

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[free_pen_raw_response][round=1]" in captured.err
    assert '"decision": "draw"' in captured.err


def test_free_pen_tool_runtime_sequence_generates_overlay_paths_and_trace(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "tool_out"
    response_path = tmp_path / "tool_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {
                    "decision": "tool_call",
                    "tool_call": {"tool": "start_path", "x": 12, "y": 52},
                    "reason": "start",
                },
                {
                    "decision": "tool_call",
                    "tool_call": {"tool": "curve_to", "c1": [24, 40], "c2": [60, 26], "p": [84, 18]},
                    "reason": "curve",
                },
                {
                    "decision": "finish",
                    "reason": "done",
                },
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=4,
        closed_contour_mode=False,
    )
    result = runtime.run(input_path, output_dir)

    assert result.status == "finished"
    assert result.successful_step_count == 2
    assert result.rounds_executed == 3
    overlay = cv2.imread(str(result.final_overlay_path), cv2.IMREAD_UNCHANGED)
    assert overlay is not None
    assert int(np.count_nonzero(overlay[:, :, 3])) > 0
    assert result.final_composite_path.exists()
    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    assert trace_payload["successful_step_count"] == 2
    assert trace_payload["rounds"][0]["executed_tool_call"]["tool"] == "start_path"
    paths_payload = json.loads(result.paths_json_path.read_text(encoding="utf-8"))
    assert paths_payload["paths"][0]["segments"][0]["type"] == "move"
    assert paths_payload["paths"][0]["segments"][1]["type"] == "cubic"


def test_free_pen_tool_runtime_records_invalid_tool_call_without_crashing(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "tool_invalid"
    response_path = tmp_path / "tool_invalid.json"
    response_path.write_text(
        json.dumps(
            [
                {
                    "decision": "tool_call",
                    "tool_call": {"tool": "line_to", "x": 20, "y": 20},
                    "reason": "invalid without start_path",
                }
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=FileSequenceFreePenAdapter(response_path=response_path), max_steps=2)
    result = runtime.run(input_path, output_dir)

    assert result.status == "invalid_response"
    assert result.error_message is not None
    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    assert trace_payload["invalid_step_count"] >= 1
    assert trace_payload["rounds"][-1]["validation_result"]["success"] is False


def test_free_pen_tool_cli_uses_file_sequence_provider_and_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "tool_cli"
    response_path = tmp_path / "tool_cli_response.json"
    response_path.write_text(
        json.dumps(
            [
                {
                    "decision": "tool_call",
                    "tool_call": {"tool": "start_path", "x": 12, "y": 52},
                    "reason": "start",
                },
                {
                    "decision": "tool_call",
                    "tool_call": {"tool": "curve_to", "c1": [24, 40], "c2": [60, 26], "p": [84, 18]},
                    "reason": "curve",
                },
                {
                    "decision": "stalled",
                    "reason": "done",
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_PROVIDER", "file")
    monkeypatch.setenv("AI_FILE_RESPONSE_PATH", str(response_path))
    exit_code = main(
        [
            "free-pen-tool",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--max-steps",
            "4",
        ]
    )

    assert exit_code == 0
    for name in ("final_overlay.png", "final_composite.png", "free_pen_paths.json", "tool_trace.json"):
        assert (output_dir / name).exists()


def test_rollback_to_step_restores_canvas_state(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "rollback_out"
    response_path = tmp_path / "rollback_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "tool_call", "tool_call": {"tool": "line_to", "x": 24, "y": 48}, "reason": "straight setup"},
                {
                    "decision": "tool_call",
                    "tool_call": {"tool": "curve_to", "c1": [32, 44], "c2": [56, 28], "p": [84, 18]},
                    "reason": "smooth curve",
                },
                {"decision": "tool_call", "tool_call": {"tool": "rollback_to_step", "step": 1}, "reason": "retry from step 1"},
                {"decision": "finish", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=5,
        closed_contour_mode=False,
    )
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    history = trace_payload["history"]
    rollback_entry = history[3]
    assert result.status == "finished"
    assert rollback_entry["tool"] == "rollback_to_step"
    assert rollback_entry["runtime_description"].startswith("Rolled back the canvas to successful drawing step 1")
    assert rollback_entry["state_after"]["current_point"] == [12.0, 52.0]
    assert rollback_entry["state_after"]["path_count"] == 1
    paths_payload = json.loads(result.paths_json_path.read_text(encoding="utf-8"))
    assert paths_payload["paths"][0]["segments"] == [{"type": "move", "p": [12.0, 52.0]}]


def test_undo_last_restores_previous_step(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "undo_out"
    response_path = tmp_path / "undo_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "tool_call", "tool_call": {"tool": "line_to", "x": 24, "y": 48}, "reason": "straight step"},
                {"decision": "tool_call", "tool_call": {"tool": "undo_last"}, "reason": "remove the wrong line"},
                {"decision": "finish", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=4,
        closed_contour_mode=False,
    )
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    history = trace_payload["history"]
    undo_entry = history[2]
    assert result.status == "finished"
    assert undo_entry["tool"] == "undo_last"
    assert undo_entry["state_after"]["current_point"] == [12.0, 52.0]
    assert trace_payload["rollback_count"] == 1


def test_inspect_history_returns_recent_descriptions_without_mutating_canvas(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "inspect_out"
    response_path = tmp_path / "inspect_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "tool_call", "tool_call": {"tool": "inspect_history", "last_n": 8}, "reason": "review recent mistakes"},
                {"decision": "finish", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=3,
        closed_contour_mode=False,
    )
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    inspect_entry = trace_payload["history"][1]
    assert result.status == "finished"
    assert inspect_entry["tool"] == "inspect_history"
    assert inspect_entry["runtime_description"].startswith("Reviewed the most recent 8 history entries")
    assert inspect_entry["state_after"]["current_point"] == [12.0, 52.0]


def test_close_path_too_early_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "close_reject_out"
    response_path = tmp_path / "close_reject_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {
                    "decision": "tool_call",
                    "tool_call": {"tool": "curve_to", "c1": [18, 44], "c2": [22, 40], "p": [24, 48]},
                    "reason": "ellipse curve start",
                },
                {"decision": "tool_call", "tool_call": {"tool": "close_path"}, "reason": "close the oval"},
                {"decision": "finish", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=FileSequenceFreePenAdapter(response_path=response_path), max_steps=4)
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    rejected_round = trace_payload["rounds"][2]
    assert rejected_round["rejected_tool_call"]["tool"] == "close_path"
    warning_codes = {warning["code"] for warning in rejected_round["warnings"]}
    assert "close_path_used_too_early" in warning_codes
    assert "long_chord_if_closed" in warning_codes
    assert rejected_round["canvas_state_summary"]["path_open"] is True


def test_finish_with_open_path_is_rejected_in_closed_contour_mode(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "finish_reject_out"
    response_path = tmp_path / "finish_reject_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "finish", "reason": "the current path looks good enough"},
                {"decision": "stalled", "reason": "stop after rejection"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=FileSequenceFreePenAdapter(response_path=response_path), max_steps=3)
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    finish_round = trace_payload["rounds"][1]
    assert result.status != "finished"
    assert result.status == "stalled"
    assert finish_round["rejected_tool_call"] is None
    assert trace_payload["final_status"] != "finish"
    assert trace_payload["rejected_step_count"] == 1
    warning_codes = {warning["code"] for warning in trace_payload["history"][1]["warnings"]}
    assert "finish_with_open_path" in warning_codes


def test_line_to_on_smooth_curve_reason_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "line_warning_out"
    response_path = tmp_path / "line_warning_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "tool_call", "tool_call": {"tool": "line_to", "x": 24, "y": 48}, "reason": "Tracing the bottom ellipse curve"},
                {"decision": "finish", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=3,
        closed_contour_mode=False,
    )
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    request_payload = json.loads((output_dir / "round_003_request.json").read_text(encoding="utf-8"))
    line_round = trace_payload["rounds"][1]
    warning_codes = {warning["code"] for warning in trace_payload["history"][1]["warnings"]}
    assert line_round["rejected_tool_call"]["tool"] == "line_to"
    assert trace_payload["history"][1]["round_status"] == "rejected_action"
    assert result.rejected_step_count >= 1
    assert line_round["canvas_state_summary"]["current_point"] == [12.0, 52.0]
    assert trace_payload["history"][1]["runtime_description"] == "Rejected line_to because the model described a smooth curve but used a straight line tool."
    assert trace_payload["history"][1]["quality_summary"] == "line_to draws a straight segment and is not appropriate for the described smooth curve."
    assert "line_to_used_on_smooth_curve_hint" in warning_codes
    round_three_state_text = request_payload["messages"][-1]["content"]
    assert "You described a smooth curve but used line_to." in round_three_state_text
    assert "Use curve_to with c1, c2, and p." in round_three_state_text


def test_line_to_for_straight_segment_still_allowed(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "line_straight_out"
    response_path = tmp_path / "line_straight_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "tool_call", "tool_call": {"tool": "line_to", "x": 24, "y": 48}, "reason": "Drawing a visible straight edge."},
                {"decision": "finish", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=3,
        closed_contour_mode=False,
    )
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    assert result.status == "finished"
    assert trace_payload["rounds"][1]["executed_tool_call"]["tool"] == "line_to"
    assert trace_payload["history"][1]["round_status"] == "tool_applied"


def test_overlay_confusion_reason_generates_warning(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "overlay_warning_out"
    response_path = tmp_path / "overlay_warning_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "follow the orange overlay start"},
                {"decision": "finish", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=2,
        closed_contour_mode=False,
    )
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    warning_codes = {warning["code"] for warning in trace_payload["history"][0]["warnings"]}
    assert result.status == "finished"
    assert "overlay_target_confusion_hint" in warning_codes


class _TimeoutAdapter:
    provider_name = "fake"
    model = "timeout-model"

    def review(self, prompt: str, review_input: object) -> dict[str, object]:
        raise TimeoutError("timed out while waiting for provider response")


def test_provider_timeout_status_is_provider_timeout(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "timeout_out"

    runtime = FreePenToolRuntime(adapter=_TimeoutAdapter(), max_steps=2)
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    assert result.status == "provider_timeout"
    assert result.error_type == "ProviderTimeout"
    assert trace_payload["final_status"] == "provider_timeout"
    assert result.final_overlay_path.exists()
    assert result.paths_json_path.exists()


def test_rejected_action_does_not_mutate_canvas(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "reject_nomutate_out"
    response_path = tmp_path / "reject_nomutate_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "tool_call", "tool_call": {"tool": "line_to", "x": 24, "y": 48}, "reason": "ellipse curve"},
                {"decision": "tool_call", "tool_call": {"tool": "close_path"}, "reason": "close now"},
                {"decision": "finish", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=4,
        closed_contour_mode=False,
    )
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    state_before = trace_payload["history"][1]["state_after"]
    state_after = trace_payload["history"][2]["state_after"]
    assert result.status == "finished"
    assert state_before["current_point"] == state_after["current_point"]
    assert state_before["path_count"] == state_after["path_count"]
    assert state_before["line_to_streak"] == state_after["line_to_streak"]


def test_prompt_contains_source_overlay_distinction_and_history_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "prompt_out"
    response_path = tmp_path / "prompt_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "tool_call", "tool_call": {"tool": "inspect_history", "last_n": 8}, "reason": "review history"},
                {"decision": "stalled", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    captured_prompts: list[str] = []
    captured_messages: list[tuple[dict[str, object], ...]] = []

    class _PromptCaptureAdapter(FileSequenceFreePenAdapter):
        def review(self, prompt: str, review_input: object) -> dict[str, object]:
            captured_prompts.append(prompt)
            captured_messages.append(tuple(getattr(review_input, "messages", ())))
            return super().review(prompt, review_input)

    runtime = FreePenToolRuntime(adapter=_PromptCaptureAdapter(response_path=response_path), max_steps=3)
    runtime.run(input_path, output_dir)

    assert len(captured_prompts) >= 2
    system_prompt = captured_prompts[1]
    assert "The source image is the target." in system_prompt
    assert "The overlay image is your previous drawing only." in system_prompt
    assert "Do not trace the overlay." in system_prompt
    messages = captured_messages[1]
    assert messages[0]["role"] == "system"
    assert "Recent history:" in messages[-1]["content"]


def test_round_request_snapshot_contains_messages_and_image_urls(tmp_path: Path) -> None:
    input_path = tmp_path / "samples" / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "out" / "request_snapshot_out"
    response_path = tmp_path / "request_snapshot_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "stalled", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=2,
        image_transport_config=FreePenImageTransportConfig(
            mode="url",
            public_image_base_url="https://img.jinyao.qzz.io/",
            public_image_root=tmp_path,
            conversation_max_turns=30,
        ),
    )
    runtime.run(input_path, output_dir)

    request_payload = json.loads((output_dir / "round_001_request.json").read_text(encoding="utf-8"))
    assert request_payload["image_transport"] == "url"
    assert request_payload["messages"][0]["role"] == "system"
    first_user_content = request_payload["messages"][1]["content"]
    assert first_user_content[0]["type"] == "image_url"
    assert first_user_content[0]["image_url"]["url"] == "https://img.jinyao.qzz.io/samples/source.png"
    assert first_user_content[1]["type"] == "text"
    assert "session_state" in request_payload


def test_source_image_appended_once_in_conversation(tmp_path: Path) -> None:
    input_path = tmp_path / "samples" / "circle_quickstart.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "out" / "source_once_url_out"
    response_path = tmp_path / "source_once_url_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "tool_call", "tool_call": {"tool": "inspect_history", "last_n": 5}, "reason": "inspect"},
                {"decision": "stalled", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=3,
        image_transport_config=FreePenImageTransportConfig(
            mode="url",
            public_image_base_url="https://img.jinyao.qzz.io/",
            public_image_root=tmp_path,
        ),
    )
    runtime.run(input_path, output_dir)

    source_url = "https://img.jinyao.qzz.io/samples/circle_quickstart.png"
    conversation_payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    assert json.dumps(conversation_payload).count(source_url) == 1
    for round_name in ("round_001_request.json", "round_002_request.json", "round_003_request.json"):
        request_payload = json.loads((output_dir / round_name).read_text(encoding="utf-8"))
        request_text = json.dumps(request_payload)
        assert request_text.count(source_url) == 1
        assert request_payload["messages"][0]["role"] == "system"
        assert "image_url" not in str(request_payload["messages"][0]["content"])
    round_two_payload = json.loads((output_dir / "round_002_request.json").read_text(encoding="utf-8"))
    round_three_payload = json.loads((output_dir / "round_003_request.json").read_text(encoding="utf-8"))
    assert any(part.get("type") == "image_url" and "round_001_overlay.png" in part["image_url"]["url"] for part in round_two_payload["messages"][-2]["content"])
    assert any(
        part.get("type") == "image_url"
        and ("round_002_overlay.png" in part["image_url"]["url"] or "round_003_composite_context.png" in part["image_url"]["url"])
        for part in round_three_payload["messages"][-2]["content"]
    )


def test_session_state_sent_each_round(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "session_state_out"
    response_path = tmp_path / "session_state_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "stalled", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=FileSequenceFreePenAdapter(response_path=response_path), max_steps=2)
    runtime.run(input_path, output_dir)

    request_payload = json.loads((output_dir / "round_001_request.json").read_text(encoding="utf-8"))
    session_state = request_payload["session_state"]
    assert "path_count" in session_state
    assert "closed_path_count" in session_state
    assert "path_open" in session_state
    assert "allowed_next_actions" in session_state
    assert "forbidden_next_actions" in session_state
    assert "current_goal" in session_state


def test_base64_transport_still_available(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "base64_out"
    response_path = tmp_path / "base64_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "stalled", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=2,
        image_transport_config=FreePenImageTransportConfig(mode="base64"),
    )
    runtime.run(input_path, output_dir)

    request_payload = json.loads((output_dir / "round_001_request.json").read_text(encoding="utf-8"))
    first_user_content = request_payload["messages"][1]["content"]
    assert first_user_content[0]["type"] == "image_url"
    assert first_user_content[1]["type"] == "text"
    image_url = first_user_content[0]["image_url"]["url"]
    assert image_url.endswith("<base64 data omitted>")


def test_source_not_reappended_after_first_round_base64_mode(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "base64_source_once_out"
    response_path = tmp_path / "base64_source_once_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "tool_call", "tool_call": {"tool": "inspect_history", "last_n": 5}, "reason": "inspect"},
                {"decision": "stalled", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=3,
        image_transport_config=FreePenImageTransportConfig(mode="base64"),
    )
    runtime.run(input_path, output_dir)

    conversation_payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    conversation_text = json.dumps(conversation_payload)
    assert conversation_text.count("data:image/png;base64,<base64 data omitted>") >= 1
    round_one = json.loads((output_dir / "round_001_request.json").read_text(encoding="utf-8"))
    round_two = json.loads((output_dir / "round_002_request.json").read_text(encoding="utf-8"))
    round_three = json.loads((output_dir / "round_003_request.json").read_text(encoding="utf-8"))
    assert json.dumps(round_one).count("This is the target source image. Trace the single black contour in this image.") == 1
    assert json.dumps(round_two).count("This is the target source image. Trace the single black contour in this image.") == 1
    assert json.dumps(round_three).count("This is the target source image. Trace the single black contour in this image.") == 1


def test_free_pen_tool_url_mode_file_provider_smoke(tmp_path: Path) -> None:
    input_path = tmp_path / "samples" / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "out" / "url_mode_out"
    response_path = tmp_path / "url_mode_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                {"decision": "tool_call", "tool_call": {"tool": "start_path", "x": 12, "y": 52}, "reason": "start"},
                {"decision": "stalled", "reason": "done"},
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=FileSequenceFreePenAdapter(response_path=response_path),
        max_steps=2,
        image_transport_config=FreePenImageTransportConfig(
            mode="url",
            public_image_base_url="https://img.jinyao.qzz.io/",
            public_image_root=tmp_path,
        ),
    )
    result = runtime.run(input_path, output_dir)

    request_payload = json.loads((output_dir / "round_001_request.json").read_text(encoding="utf-8"))
    assert result.final_overlay_path.exists()
    assert (output_dir / "conversation_messages.json").exists()
    assert (output_dir / "round_001_request.json").exists()
    assert request_payload["messages"][1]["content"][0]["image_url"]["url"] == "https://img.jinyao.qzz.io/samples/source.png"
