from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from services.ai_adapters import create_vision_adapter
from services.free_pen_canvas import FreePenCanvasState
from services.free_pen_prompt import build_free_pen_prompt
from services.free_pen_runtime import (
    FileSequenceFreePenAdapter,
    FreePenImageTransportConfig,
    FreePenReviewInput,
    FreePenRuntime,
    NativeToolCallSequenceAdapter,
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


def _native_tool_call(name: str, arguments: dict[str, object], *, call_id: str) -> dict[str, object]:
    return {
        "tool_call": {
            "id": call_id,
            "name": name,
            "arguments": arguments,
        }
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


def test_editable_geometry_exports_anchor_and_segment_ids() -> None:
    canvas = FreePenCanvasState(width=220, height=220)
    canvas.apply_tool_call({"tool": "start_path", "x": 33, "y": 91})
    canvas.apply_tool_call({"tool": "curve_to", "c1": [33, 73], "c2": [55, 58], "p": [85, 58]})
    canvas.apply_tool_call({"tool": "curve_to", "c1": [145, 58], "c2": [187, 90], "p": [187, 129]})

    geometry = canvas.editable_geometry()

    assert geometry["paths"][0]["anchors"] == [
        {"id": "A1", "p": [33.0, 91.0]},
        {"id": "A2", "p": [85.0, 58.0]},
        {"id": "A3", "p": [187.0, 129.0]},
    ]
    assert geometry["paths"][0]["segments"] == [
        {
            "id": "S1",
            "type": "cubic",
            "from_anchor": "A1",
            "to_anchor": "A2",
            "c1": [33.0, 73.0],
            "c2": [55.0, 58.0],
        },
        {
            "id": "S2",
            "type": "cubic",
            "from_anchor": "A2",
            "to_anchor": "A3",
            "c1": [145.0, 58.0],
            "c2": [187.0, 90.0],
        },
    ]


def test_execute_move_anchor_updates_path_endpoint() -> None:
    canvas = FreePenCanvasState(width=220, height=220)
    canvas.apply_tool_call({"tool": "start_path", "x": 33, "y": 91})
    canvas.apply_tool_call({"tool": "curve_to", "c1": [33, 73], "c2": [55, 58], "p": [85, 58]})

    canvas.apply_tool_call({"tool": "move_anchor", "anchor_id": "A2", "x": 88, "y": 60})

    assert canvas.paths[0].segments[1]["p"] == [88.0, 60.0]
    assert canvas.current_point == (88.0, 60.0)


def test_execute_move_handle_updates_cubic_control_point() -> None:
    canvas = FreePenCanvasState(width=220, height=220)
    canvas.apply_tool_call({"tool": "start_path", "x": 33, "y": 91})
    canvas.apply_tool_call({"tool": "curve_to", "c1": [33, 73], "c2": [55, 58], "p": [85, 58]})

    canvas.apply_tool_call({"tool": "move_handle", "segment_id": "S1", "handle": "c1", "x": 40, "y": 70})

    assert canvas.paths[0].segments[1]["c1"] == [40.0, 70.0]


def test_execute_set_segment_handles_updates_both_handles() -> None:
    canvas = FreePenCanvasState(width=220, height=220)
    canvas.apply_tool_call({"tool": "start_path", "x": 33, "y": 91})
    canvas.apply_tool_call({"tool": "curve_to", "c1": [33, 73], "c2": [55, 58], "p": [85, 58]})

    canvas.apply_tool_call(
        {"tool": "set_segment_handles", "segment_id": "S1", "c1": [44, 72], "c2": [66, 60]}
    )

    assert canvas.paths[0].segments[1]["c1"] == [44.0, 72.0]
    assert canvas.paths[0].segments[1]["c2"] == [66.0, 60.0]


def test_editable_geometry_marks_line_as_not_handle_editable() -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=Path("dummy.json")))
    canvas = FreePenCanvasState(width=220, height=220)
    canvas.apply_tool_call({"tool": "start_path", "x": 33, "y": 91})
    canvas.apply_tool_call({"tool": "line_to", "x": 85, "y": 58})

    geometry = runtime._build_editable_geometry(canvas)

    assert geometry["paths"][0]["segments"] == [
        {
            "id": "S1",
            "type": "line",
            "from_anchor": "A1",
            "to_anchor": "A2",
            "p": [85.0, 58.0],
            "editable_with_handles": False,
            "conversion_tool": "convert_line_to_curve",
        }
    ]


def test_convert_line_to_curve_updates_path_segment() -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=Path("dummy.json")))
    canvas = FreePenCanvasState(width=220, height=220)
    canvas.apply_tool_call({"tool": "start_path", "x": 33, "y": 91})
    canvas.apply_tool_call({"tool": "line_to", "x": 85, "y": 58})

    executed = runtime._convert_line_to_curve(
        canvas=canvas,
        tool_call={"tool": "convert_line_to_curve", "segment_id": "S1", "c1": [45, 80], "c2": [70, 60]},
    )

    assert executed["segment_id"] == "S1"
    assert canvas.paths[0].segments[1]["type"] == "cubic"
    assert canvas.paths[0].segments[1]["p"] == [85.0, 58.0]
    assert canvas.paths[0].segments[1]["c1"] == [45.0, 80.0]
    assert canvas.paths[0].segments[1]["c2"] == [70.0, 60.0]


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
    input_path = tmp_path / "samples" / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "out" / "tool_cli"
    response_path = tmp_path / "tool_cli_response.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "curve"},
                    call_id="call_002",
                ),
                _native_tool_call("stalled", {"reason": "done"}, call_id="call_003"),
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_PROVIDER", "file")
    monkeypatch.setenv("AI_FILE_RESPONSE_PATH", str(response_path))
    monkeypatch.setenv("AI_IMAGE_TRANSPORT", "base64")
    monkeypatch.delenv("AI_PUBLIC_IMAGE_BASE_URL", raising=False)
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


def test_current_segment_status_blocks_advancing_when_quality_bad(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "segment_gate_reject_out"
    response_path = tmp_path / "segment_gate_reject_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "initial curve"},
                    call_id="call_002",
                ),
                _native_tool_call(
                    "curve_to",
                    {"c1": [90, 20], "c2": [120, 20], "p": [140, 24], "reason": "should be blocked"},
                    call_id="call_003",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_004"),
            ]
        ),
        encoding="utf-8",
    )

    original = FreePenToolRuntime._build_current_segment_context

    def fake_context(self, *, canvas, source_distance_map, focus_tool_call, segment_refinement):
        context = original(
            self,
            canvas=canvas,
            source_distance_map=source_distance_map,
            focus_tool_call=focus_tool_call,
            segment_refinement=segment_refinement,
        )
        if canvas.path_open and canvas.current_path_drawable_segment_count() >= 1:
            context["focus"] = {
                "segment_id": "S1",
                "type": "cubic",
                "from_anchor": "A1",
                "to_anchor": "A2",
                "recommended_action": "inspect_or_refine",
                "hint": "Refine S1 first.",
            }
            context["status"] = {
                "segment_id": "S1",
                "status": "needs_refinement",
                "reason": "Newest segment needs refinement.",
                "recommended_next_tools": ["set_segment_handles", "move_handle", "move_anchor", "undo_last", "rollback_to_step", "inspect_history", "stalled"],
                "may_advance_to_next_segment": False,
            }
        return context

    monkeypatch.setattr(FreePenToolRuntime, "_build_current_segment_context", fake_context)
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=4)
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    blocked_round = trace_payload["rounds"][2]
    warning_codes = {warning["code"] for warning in blocked_round["warnings"]}
    assert blocked_round["rejected_tool_call"]["tool"] == "curve_to"
    assert "current_segment_needs_refinement" in warning_codes


def test_current_segment_status_allows_handle_edit_when_quality_bad(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "segment_gate_edit_out"
    response_path = tmp_path / "segment_gate_edit_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "initial curve"},
                    call_id="call_002",
                ),
                _native_tool_call(
                    "set_segment_handles",
                    {"segment_id": "S1", "c1": [22, 42], "c2": [58, 24], "reason": "refine"},
                    call_id="call_003",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_004"),
            ]
        ),
        encoding="utf-8",
    )

    original = FreePenToolRuntime._build_current_segment_context

    def fake_context(self, *, canvas, source_distance_map, focus_tool_call, segment_refinement):
        context = original(
            self,
            canvas=canvas,
            source_distance_map=source_distance_map,
            focus_tool_call=focus_tool_call,
            segment_refinement=segment_refinement,
        )
        if canvas.path_open and canvas.current_path_drawable_segment_count() >= 1:
            context["focus"] = {
                "segment_id": "S1",
                "type": "cubic",
                "from_anchor": "A1",
                "to_anchor": "A2",
                "recommended_action": "inspect_or_refine",
                "hint": "Refine S1 first.",
            }
            context["status"] = {
                "segment_id": "S1",
                "status": "needs_refinement",
                "reason": "Newest segment needs refinement.",
                "recommended_next_tools": ["set_segment_handles", "move_handle", "move_anchor", "undo_last", "rollback_to_step", "inspect_history", "stalled"],
                "may_advance_to_next_segment": False,
            }
        return context

    monkeypatch.setattr(FreePenToolRuntime, "_build_current_segment_context", fake_context)
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=4)
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    edit_round = trace_payload["rounds"][2]
    assert edit_round["executed_tool_call"]["tool"] == "set_segment_handles"


def test_current_segment_status_allows_advancing_when_quality_acceptable(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "segment_gate_allow_out"
    response_path = tmp_path / "segment_gate_allow_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "advance"},
                    call_id="call_002",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_003"),
            ]
        ),
        encoding="utf-8",
    )

    original = FreePenToolRuntime._build_current_segment_context

    def fake_context(self, *, canvas, source_distance_map, focus_tool_call, segment_refinement):
        context = original(
            self,
            canvas=canvas,
            source_distance_map=source_distance_map,
            focus_tool_call=focus_tool_call,
            segment_refinement=segment_refinement,
        )
        if canvas.path_open and canvas.current_path_drawable_segment_count() >= 1:
            context["focus"] = {
                "segment_id": "S1",
                "type": "cubic",
                "from_anchor": "A1",
                "to_anchor": "A2",
                "recommended_action": "inspect_or_refine",
                "hint": "S1 acceptable.",
            }
            context["status"] = {
                "segment_id": "S1",
                "status": "acceptable",
                "reason": "acceptable",
                "recommended_next_tools": ["curve_to", "line_to", "close_path"],
                "may_advance_to_next_segment": True,
            }
        return context

    monkeypatch.setattr(FreePenToolRuntime, "_build_current_segment_context", fake_context)
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=3)
    result = runtime.run(input_path, output_dir)

    trace_payload = json.loads(result.tool_trace_path.read_text(encoding="utf-8"))
    allowed_round = trace_payload["rounds"][1]
    assert allowed_round["executed_tool_call"]["tool"] == "curve_to"


def test_quality_delta_reports_improvement(tmp_path: Path) -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    quality_delta = runtime._build_quality_delta(
        focus={"segment_id": "S2"},
        metrics={"current_segment": {"segment_id": "S2", "path_to_source_p90_px": 6.0, "path_to_source_mean_px": 2.0, "path_to_source_max_px": 7.0}},
        refinement_state={
            "previous_quality": {"path_to_source_p90_px": 10.0},
            "last_quality": {"path_to_source_p90_px": 10.0},
            "best_quality": {"path_to_source_p90_px": 10.0},
            "worse_streak": 0,
        },
    )
    assert quality_delta["improved_vs_previous"] is True
    assert quality_delta["improved_vs_best"] is True


def test_quality_delta_reports_worse_edit(tmp_path: Path) -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    quality_delta = runtime._build_quality_delta(
        focus={"segment_id": "S2"},
        metrics={"current_segment": {"segment_id": "S2", "path_to_source_p90_px": 16.0, "path_to_source_mean_px": 8.0, "path_to_source_max_px": 20.0}},
        refinement_state={
            "previous_quality": {"path_to_source_p90_px": 10.0},
            "last_quality": {"path_to_source_p90_px": 10.0},
            "best_quality": {"path_to_source_p90_px": 8.0},
            "worse_streak": 2,
        },
    )
    assert quality_delta["improved_vs_previous"] is False
    assert quality_delta["worse_streak"] == 2
    assert "worse" in quality_delta["message"].lower()


def test_segment_refine_count_increments_on_set_segment_handles(tmp_path: Path) -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    context = {
        "focus": {"segment_id": "S2"},
        "quality_metrics": {
            "current_segment": {
                "segment_id": "S2",
                "path_to_source_mean_px": 4.0,
                "path_to_source_max_px": 8.0,
                "path_to_source_p90_px": 7.0,
            }
        },
    }
    state = runtime._update_segment_refinement_state(
        segment_refinement={},
        tool_call={"tool": "set_segment_handles", "segment_id": "S2", "c1": [10, 10], "c2": [20, 20]},
        current_segment_context=context,
    )
    state = runtime._update_segment_refinement_state(
        segment_refinement=state,
        tool_call={"tool": "set_segment_handles", "segment_id": "S2", "c1": [11, 10], "c2": [21, 20]},
        current_segment_context=context,
    )
    assert state["S2"]["refine_count"] == 2


def test_best_quality_updates_only_when_improved(tmp_path: Path) -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    state: dict[str, dict[str, object]] = {}
    for p90 in (12.0, 8.0, 20.0):
        state = runtime._update_segment_refinement_state(
            segment_refinement=state,
            tool_call={"tool": "set_segment_handles", "segment_id": "S2", "c1": [10, 10], "c2": [20, 20]},
            current_segment_context={
                "focus": {"segment_id": "S2"},
                "quality_metrics": {
                    "current_segment": {
                        "segment_id": "S2",
                        "path_to_source_mean_px": p90 / 4.0,
                        "path_to_source_max_px": p90 + 2.0,
                        "path_to_source_p90_px": p90,
                    }
                },
            },
        )
    assert state["S2"]["best_quality"]["path_to_source_p90_px"] == 8.0


def test_refinement_limit_blocks_more_handle_edits(tmp_path: Path) -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    canvas = FreePenCanvasState(width=96, height=72)
    canvas.apply_tool_call({"tool": "start_path", "x": 12, "y": 52})
    canvas.apply_tool_call({"tool": "curve_to", "c1": [24, 40], "c2": [60, 26], "p": [84, 18]})
    preflight = runtime._preflight_tool_call(
        tool_call={"tool": "set_segment_handles", "segment_id": "S1", "c1": [22, 42], "c2": [58, 24]},
        ai_reason="keep tuning handles",
        canvas=canvas,
        successful_drawing_step_count=2,
        rollback_count=0,
        current_segment_context={
            "editable_geometry": canvas.editable_geometry(),
            "focus": {"segment_id": "S1", "type": "cubic"},
            "status": {
                "segment_id": "S1",
                "status": "needs_refinement",
                "reason": "bad",
                "recommended_next_tools": ["move_anchor", "rollback_to_step", "restart_path"],
                "may_advance_to_next_segment": False,
                "refinement_limit_reached": True,
                "may_continue_handle_refinement": False,
            },
            "quality_metrics": {"current_segment": {"segment_id": "S1", "path_to_source_p90_px": 12.0}},
            "quality_delta": {},
            "refinement_summary": {},
        },
        source_distance_map=None,
    )
    warning_codes = {warning["code"] for warning in preflight["warnings"]}
    assert preflight["success"] is False
    assert "refinement_limit_reached" in warning_codes


def test_refinement_limit_allows_move_anchor(tmp_path: Path) -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    canvas = FreePenCanvasState(width=96, height=72)
    canvas.apply_tool_call({"tool": "start_path", "x": 12, "y": 52})
    canvas.apply_tool_call({"tool": "curve_to", "c1": [24, 40], "c2": [60, 26], "p": [84, 18]})
    preflight = runtime._preflight_tool_call(
        tool_call={"tool": "move_anchor", "anchor_id": "A2", "x": 82, "y": 20},
        ai_reason="anchor wrong",
        canvas=canvas,
        successful_drawing_step_count=2,
        rollback_count=0,
        current_segment_context={
            "editable_geometry": canvas.editable_geometry(),
            "focus": {"segment_id": "S1", "type": "cubic"},
            "status": {
                "segment_id": "S1",
                "status": "needs_refinement",
                "reason": "bad",
                "recommended_next_tools": ["move_anchor", "rollback_to_step", "restart_path"],
                "may_advance_to_next_segment": False,
                "refinement_limit_reached": True,
                "may_continue_handle_refinement": False,
            },
            "quality_metrics": {"current_segment": {"segment_id": "S1", "path_to_source_p90_px": 12.0}},
            "quality_delta": {},
            "refinement_summary": {},
        },
        source_distance_map=None,
    )
    assert preflight["success"] is True


def test_refinement_limit_allows_restart_path(tmp_path: Path) -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    canvas = FreePenCanvasState(width=96, height=72)
    canvas.apply_tool_call({"tool": "start_path", "x": 12, "y": 52})
    preflight = runtime._preflight_tool_call(
        tool_call={"tool": "restart_path", "x": 10, "y": 50},
        ai_reason="wrong contour",
        canvas=canvas,
        successful_drawing_step_count=1,
        rollback_count=0,
        current_segment_context={
            "editable_geometry": canvas.editable_geometry(),
            "focus": {"segment_id": "S1", "type": "cubic"},
            "status": {
                "segment_id": "S1",
                "status": "needs_refinement",
                "reason": "bad",
                "recommended_next_tools": ["move_anchor", "rollback_to_step", "restart_path"],
                "may_advance_to_next_segment": False,
                "refinement_limit_reached": True,
                "may_continue_handle_refinement": False,
            },
            "quality_metrics": {"current_segment": {"segment_id": "S1", "path_to_source_p90_px": 12.0}},
            "quality_delta": {},
            "refinement_summary": {},
        },
        source_distance_map=None,
    )
    assert preflight["success"] is True


def test_current_segment_needs_refinement_allows_restart_path(tmp_path: Path) -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    canvas = FreePenCanvasState(width=96, height=72)
    canvas.apply_tool_call({"tool": "start_path", "x": 12, "y": 52})
    preflight = runtime._preflight_tool_call(
        tool_call={"tool": "restart_path", "x": 10, "y": 50},
        ai_reason="wrong path",
        canvas=canvas,
        successful_drawing_step_count=1,
        rollback_count=0,
        current_segment_context={
            "editable_geometry": canvas.editable_geometry(),
            "focus": {"segment_id": "S1", "type": "cubic"},
            "status": {
                "segment_id": "S1",
                "status": "needs_refinement",
                "reason": "bad",
                "recommended_next_tools": ["set_segment_handles", "move_handle", "move_anchor", "restart_path"],
                "may_advance_to_next_segment": False,
                "refinement_limit_reached": False,
                "may_continue_handle_refinement": True,
            },
            "quality_metrics": {"current_segment": {"segment_id": "S1", "path_to_source_p90_px": 12.0}},
            "quality_delta": {},
            "refinement_summary": {},
        },
        source_distance_map=None,
    )
    assert preflight["success"] is True


def test_start_path_rejects_point_far_from_source_contour(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    source_image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    source_distance_map = runtime._build_source_distance_map(runtime._build_source_mask(source_image))
    canvas = FreePenCanvasState(width=96, height=72)
    preflight = runtime._preflight_tool_call(
        tool_call={"tool": "start_path", "x": 90, "y": 70},
        ai_reason="start",
        canvas=canvas,
        successful_drawing_step_count=0,
        rollback_count=0,
        current_segment_context={
            "editable_geometry": {"paths": []},
            "focus": {"segment_id": None},
            "status": {"status": "unknown", "may_advance_to_next_segment": True},
            "quality_metrics": {},
            "anchor_quality": {},
            "quality_delta": {},
            "refinement_summary": {},
        },
        source_distance_map=source_distance_map,
    )
    warning_codes = {warning["code"] for warning in preflight["warnings"]}
    assert preflight["success"] is False
    assert "anchor_not_on_source_contour" in warning_codes


def test_restart_path_rejects_point_far_from_source_contour(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    source_image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    source_distance_map = runtime._build_source_distance_map(runtime._build_source_mask(source_image))
    canvas = FreePenCanvasState(width=96, height=72)
    preflight = runtime._preflight_tool_call(
        tool_call={"tool": "restart_path", "x": 90, "y": 70},
        ai_reason="restart",
        canvas=canvas,
        successful_drawing_step_count=0,
        rollback_count=0,
        current_segment_context={
            "editable_geometry": {"paths": []},
            "focus": {"segment_id": None},
            "status": {"status": "unknown", "may_advance_to_next_segment": True},
            "quality_metrics": {},
            "anchor_quality": {},
            "quality_delta": {},
            "refinement_summary": {},
        },
        source_distance_map=source_distance_map,
    )
    warning_codes = {warning["code"] for warning in preflight["warnings"]}
    assert preflight["success"] is False
    assert "anchor_not_on_source_contour" in warning_codes


def test_curve_to_rejects_endpoint_far_from_source_contour(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    source_image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    source_distance_map = runtime._build_source_distance_map(runtime._build_source_mask(source_image))
    canvas = FreePenCanvasState(width=96, height=72)
    canvas.apply_tool_call({"tool": "start_path", "x": 12, "y": 52})
    preflight = runtime._preflight_tool_call(
        tool_call={"tool": "curve_to", "c1": [40, 10], "c2": [60, 10], "p": [90, 70]},
        ai_reason="curve",
        canvas=canvas,
        successful_drawing_step_count=1,
        rollback_count=0,
        current_segment_context={
            "editable_geometry": canvas.editable_geometry(),
            "focus": {"segment_id": None},
            "status": {"status": "unknown", "may_advance_to_next_segment": True},
            "quality_metrics": {},
            "anchor_quality": {},
            "quality_delta": {},
            "refinement_summary": {},
        },
        source_distance_map=source_distance_map,
    )
    warning_codes = {warning["code"] for warning in preflight["warnings"]}
    assert preflight["success"] is False
    assert "endpoint_not_on_source_contour" in warning_codes


def test_curve_to_allows_control_points_far_from_contour(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    source_image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    source_distance_map = runtime._build_source_distance_map(runtime._build_source_mask(source_image))
    canvas = FreePenCanvasState(width=96, height=72)
    canvas.apply_tool_call({"tool": "start_path", "x": 12, "y": 52})
    preflight = runtime._preflight_tool_call(
        tool_call={"tool": "curve_to", "c1": [0, 0], "c2": [95, 0], "p": [84, 18]},
        ai_reason="curve",
        canvas=canvas,
        successful_drawing_step_count=1,
        rollback_count=0,
        current_segment_context={
            "editable_geometry": canvas.editable_geometry(),
            "focus": {"segment_id": None},
            "status": {"status": "unknown", "may_advance_to_next_segment": True},
            "quality_metrics": {},
            "anchor_quality": {},
            "quality_delta": {},
            "refinement_summary": {},
        },
        source_distance_map=source_distance_map,
    )
    warning_codes = {warning["code"] for warning in preflight["warnings"]}
    assert preflight["success"] is True
    assert "endpoint_not_on_source_contour" not in warning_codes


def test_move_anchor_rejects_target_far_from_source_contour(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    source_image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    source_distance_map = runtime._build_source_distance_map(runtime._build_source_mask(source_image))
    canvas = FreePenCanvasState(width=96, height=72)
    canvas.apply_tool_call({"tool": "start_path", "x": 12, "y": 52})
    canvas.apply_tool_call({"tool": "curve_to", "c1": [24, 40], "c2": [60, 26], "p": [84, 18]})
    preflight = runtime._preflight_tool_call(
        tool_call={"tool": "move_anchor", "anchor_id": "A2", "x": 90, "y": 70},
        ai_reason="anchor move",
        canvas=canvas,
        successful_drawing_step_count=2,
        rollback_count=0,
        current_segment_context={
            "editable_geometry": canvas.editable_geometry(),
            "focus": {"segment_id": "S1", "type": "cubic"},
            "status": {"status": "acceptable", "may_advance_to_next_segment": True},
            "quality_metrics": {},
            "anchor_quality": {},
            "quality_delta": {},
            "refinement_summary": {},
        },
        source_distance_map=source_distance_map,
    )
    warning_codes = {warning["code"] for warning in preflight["warnings"]}
    assert preflight["success"] is False
    assert "anchor_not_on_source_contour" in warning_codes


def test_reject_next_hint_mentions_retry_same_segment(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "reject_hint_out"
    response_path = tmp_path / "reject_hint_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [-10, 40], "c2": [60, 26], "p": [84, 18], "reason": "out of bounds"},
                    call_id="call_002",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_003"),
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=3)
    runtime.run(input_path, output_dir)

    payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
    parsed_tool_content = json.loads(tool_messages[1]["content"])
    next_hint = parsed_tool_content["next_hint"].lower()
    assert "rejected" in next_hint
    assert "do not continue" in next_hint
    assert "retry the same segment" in next_hint


def test_next_hint_says_do_not_continue_when_segment_bad(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "segment_bad_hint_out"
    response_path = tmp_path / "segment_bad_hint_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "curve"},
                    call_id="call_002",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_003"),
            ]
        ),
        encoding="utf-8",
    )

    original = FreePenToolRuntime._build_current_segment_context

    def fake_context(self, *, canvas, source_distance_map, focus_tool_call, segment_refinement):
        context = original(
            self,
            canvas=canvas,
            source_distance_map=source_distance_map,
            focus_tool_call=focus_tool_call,
            segment_refinement=segment_refinement,
        )
        if canvas.path_open and canvas.current_path_drawable_segment_count() >= 1:
            context["focus"] = {
                "segment_id": "S1",
                "type": "cubic",
                "from_anchor": "A1",
                "to_anchor": "A2",
                "recommended_action": "inspect_or_refine",
                "hint": "Refine S1 first.",
            }
            context["status"] = {
                "segment_id": "S1",
                "status": "needs_refinement",
                "reason": "Newest segment needs refinement.",
                "recommended_next_tools": ["set_segment_handles", "move_handle", "move_anchor", "undo_last", "rollback_to_step", "inspect_history", "stalled"],
                "may_advance_to_next_segment": False,
            }
        return context

    monkeypatch.setattr(FreePenToolRuntime, "_build_current_segment_context", fake_context)
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=3)
    runtime.run(input_path, output_dir)

    payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
    parsed_tool_content = json.loads(tool_messages[1]["content"])
    next_hint = parsed_tool_content["next_hint"].lower()
    assert "do not draw the next segment" in next_hint
    assert "refine the current segment first" in next_hint


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
    assert "Use function tool calls only" in system_prompt
    assert "Do not write JSON manually" in system_prompt
    assert '"decision"' not in system_prompt
    assert '"tool_call"' not in system_prompt
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
    assert request_payload["tool_mode"] == "native_tools"
    assert request_payload["tools"]
    assert request_payload["image_urls"] == ["https://img.jinyao.qzz.io/samples/source.png"]
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
        assert json.dumps(request_payload["messages"]).count(source_url) == 1
        assert request_payload["image_urls"].count(source_url) == 1
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
    contour_summary = session_state["source_contour_summary"]
    assert contour_summary["canvas_width"] == 96
    assert contour_summary["canvas_height"] == 72
    assert "bbox" in contour_summary
    assert "anchors" in contour_summary
    assert "leftmost" in contour_summary["anchors"]
    assert "topmost" in contour_summary["anchors"]
    assert "rightmost" in contour_summary["anchors"]
    assert "bottommost" in contour_summary["anchors"]


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


def test_free_pen_runtime_always_uses_native_tools(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "native_tools_out"
    response_path = tmp_path / "native_tools_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_002"),
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=2)
    runtime.run(input_path, output_dir)

    request_payload = json.loads((output_dir / "round_001_request.json").read_text(encoding="utf-8"))
    assert request_payload["tool_mode"] == "native_tools"
    tool_names = {tool["function"]["name"] for tool in request_payload["tools"]}
    assert {"curve_to", "finish_trace", "stalled"} <= tool_names


def test_free_pen_ignores_ai_free_pen_tool_mode_env(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "ignore_tool_mode_env_out"
    response_path = tmp_path / "ignore_tool_mode_env_sequence.json"
    response_path.write_text(
        json.dumps([_native_tool_call("stalled", {"reason": "stop"}, call_id="call_001")]),
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_FREE_PEN_TOOL_MODE", "prompt_json")
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=1)
    runtime.run(input_path, output_dir)

    request_payload = json.loads((output_dir / "round_001_request.json").read_text(encoding="utf-8"))
    assert request_payload["tool_mode"] == "native_tools"
    assert request_payload["tools"]


def test_conversation_uses_assistant_tool_calls_not_content_json(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "assistant_tool_calls_out"
    response_path = tmp_path / "assistant_tool_calls_sequence.json"
    response_path.write_text(
        json.dumps([_native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001")]),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=1)
    runtime.run(input_path, output_dir)

    payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    assistant_messages = [message for message in payload["messages"] if message["role"] == "assistant"]
    assert assistant_messages
    assert assistant_messages[0]["content"] is None
    assert assistant_messages[0]["tool_calls"][0]["function"]["name"] == "start_path"


def test_conversation_appends_tool_result_after_assistant_tool_call(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "tool_result_message_out"
    response_path = tmp_path / "tool_result_message_sequence.json"
    response_path.write_text(
        json.dumps([_native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001")]),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=1)
    runtime.run(input_path, output_dir)

    payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    assistant_index = next(index for index, message in enumerate(payload["messages"]) if message["role"] == "assistant")
    tool_message = payload["messages"][assistant_index + 1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_001"
    parsed_tool_content = json.loads(tool_message["content"])
    assert "runtime_description" in parsed_tool_content
    assert "state_after" in parsed_tool_content
    assert "allowed_next_actions" in parsed_tool_content


def test_tool_result_contains_editable_geometry(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "editable_geometry_out"
    response_path = tmp_path / "editable_geometry_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "curve"},
                    call_id="call_002",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_003"),
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=3)
    runtime.run(input_path, output_dir)

    payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
    parsed_tool_content = json.loads(tool_messages[1]["content"])
    assert "editable_geometry" in parsed_tool_content
    assert parsed_tool_content["geometry_hint"].startswith("Blue points are anchors.")
    assert parsed_tool_content["editable_geometry"]["paths"][0]["anchors"][0]["id"] == "A1"
    assert parsed_tool_content["editable_geometry"]["paths"][0]["segments"][0]["id"] == "S1"


def test_tool_result_contains_current_segment_focus_after_curve_to(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "segment_focus_out"
    response_path = tmp_path / "segment_focus_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "curve"},
                    call_id="call_002",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_003"),
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=3)
    runtime.run(input_path, output_dir)

    payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
    parsed_tool_content = json.loads(tool_messages[1]["content"])
    assert parsed_tool_content["current_segment_focus"]["segment_id"] == "S1"
    assert parsed_tool_content["current_segment_focus"]["type"] == "cubic"


def test_tool_result_contains_current_segment_status(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "segment_status_out"
    response_path = tmp_path / "segment_status_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "curve"},
                    call_id="call_002",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_003"),
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=3)
    runtime.run(input_path, output_dir)

    payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
    parsed_tool_content = json.loads(tool_messages[1]["content"])
    assert "current_segment_status" in parsed_tool_content
    assert parsed_tool_content["current_segment_status"]["segment_id"] == "S1"


def test_tool_result_contains_quality_metrics_for_current_segment(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "segment_metrics_out"
    response_path = tmp_path / "segment_metrics_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "curve"},
                    call_id="call_002",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_003"),
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=3)
    runtime.run(input_path, output_dir)

    payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
    parsed_tool_content = json.loads(tool_messages[1]["content"])
    current_segment = parsed_tool_content["quality_metrics"]["current_segment"]
    assert current_segment["segment_id"] == "S1"
    assert "path_to_source_mean_px" in current_segment or "unavailable_reason" in current_segment


def test_tool_result_contains_anchor_quality(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "anchor_quality_out"
    response_path = tmp_path / "anchor_quality_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "curve"},
                    call_id="call_002",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_003"),
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=3)
    runtime.run(input_path, output_dir)

    payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
    parsed_tool_content = json.loads(tool_messages[1]["content"])
    anchor_quality = parsed_tool_content["anchor_quality"]["current_segment"]
    assert anchor_quality["segment_id"] == "S1"
    assert "from_anchor" in anchor_quality
    assert "to_anchor" in anchor_quality


def test_tool_result_contains_refinement_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "refinement_summary_out"
    response_path = tmp_path / "refinement_summary_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "curve"},
                    call_id="call_002",
                ),
                _native_tool_call(
                    "set_segment_handles",
                    {"segment_id": "S1", "c1": [22, 42], "c2": [58, 24], "reason": "refine"},
                    call_id="call_003",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_004"),
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=response_path), max_steps=4)
    runtime.run(input_path, output_dir)

    payload = json.loads((output_dir / "conversation_messages.json").read_text(encoding="utf-8"))
    tool_messages = [message for message in payload["messages"] if message["role"] == "tool"]
    parsed_tool_content = json.loads(tool_messages[2]["content"])
    assert "refinement_summary" in parsed_tool_content
    assert parsed_tool_content["refinement_summary"]["segment_id"] == "S1"


def test_next_hint_changes_strategy_after_repeated_bad_edits(tmp_path: Path) -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    next_hint = runtime._next_hint(
        final_decision="tool_call",
        round_status="tool_applied",
        warnings=[],
        session_state={"allowed_next_actions": ["move_anchor", "restart_path"]},
        current_segment_context={
            "focus": {"segment_id": "S2"},
            "status": {
                "segment_id": "S2",
                "status": "needs_refinement",
                "reason": "bad",
                "may_advance_to_next_segment": False,
                "refinement_limit_reached": True,
            },
            "quality_delta": {
                "segment_id": "S2",
                "previous_p90_px": 15.0,
                "current_p90_px": 20.0,
                "best_p90_px": 12.7,
                "improved_vs_previous": False,
                "message": "This edit made the current segment worse. Do not keep moving handles in the same direction.",
            },
            "refinement_summary": {
                "segment_id": "S2",
                "best_quality": {"path_to_source_p90_px": 12.7},
            },
        },
    ).lower()
    assert "do not keep adjusting the same handles" in next_hint
    assert "move_anchor" in next_hint
    assert "restart_path" in next_hint


def test_visual_feedback_contains_handle_composite_or_handle_annotations(tmp_path: Path) -> None:
    input_path = tmp_path / "samples" / "source.png"
    _write_source_image(input_path)
    output_dir = tmp_path / "out" / "handles_feedback_out"
    response_path = tmp_path / "handles_feedback_sequence.json"
    response_path.write_text(
        json.dumps(
            [
                _native_tool_call("start_path", {"x": 12, "y": 52, "reason": "start"}, call_id="call_001"),
                _native_tool_call(
                    "curve_to",
                    {"c1": [24, 40], "c2": [60, 26], "p": [84, 18], "reason": "curve"},
                    call_id="call_002",
                ),
                _native_tool_call("stalled", {"reason": "stop"}, call_id="call_003"),
            ]
        ),
        encoding="utf-8",
    )

    runtime = FreePenToolRuntime(
        adapter=NativeToolCallSequenceAdapter(response_path=response_path),
        max_steps=3,
        image_transport_config=FreePenImageTransportConfig(
            mode="url",
            public_image_base_url="https://img.jinyao.qzz.io/",
            public_image_root=tmp_path,
        ),
    )
    runtime.run(input_path, output_dir)

    assert (output_dir / "round_002_post_tool_composite.png").exists()
    request_payload = json.loads((output_dir / "round_003_request.json").read_text(encoding="utf-8"))
    visual_feedback_content = request_payload["messages"][-2]["content"]
    assert any(
        part.get("type") == "image_url"
        and "round_002_post_tool_composite.png" in part["image_url"]["url"]
        for part in visual_feedback_content
    )
    assert any(
        part.get("type") == "text"
        and "BLACK = source target." in part["text"]
        and "BLUE = anchors." in part["text"]
        and "GREEN = control handles." in part["text"]
        for part in visual_feedback_content
    )
    feedback_text = next(part["text"] for part in visual_feedback_content if part.get("type") == "text")
    assert len(feedback_text) < 1200


def test_bad_anchor_status_disables_handle_editing(tmp_path: Path) -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    status = runtime._build_current_segment_status(
        focus={"segment_id": "S1", "type": "cubic"},
        metrics={"current_segment": {"segment_id": "S1", "path_to_source_mean_px": 1.0, "path_to_source_max_px": 2.0, "path_to_source_p90_px": 2.0}},
        refinement_state=None,
        anchor_quality={
            "current_segment": {
                "segment_id": "S1",
                "from_anchor": {"id": "A1", "p": [24, 436], "distance_to_source_px": 125.13, "status": "bad"},
                "to_anchor": {"id": "A2", "p": [45, 203], "distance_to_source_px": 2.2, "status": "ok"},
            }
        },
    )
    assert status["status"] == "needs_anchor_correction"
    assert "set_segment_handles" not in status["recommended_next_tools"]
    assert "move_handle" not in status["recommended_next_tools"]
    assert "move_anchor" in status["recommended_next_tools"]
    assert "restart_path" in status["recommended_next_tools"]


def test_next_hint_mentions_move_anchor_when_anchor_bad(tmp_path: Path) -> None:
    runtime = FreePenToolRuntime(adapter=NativeToolCallSequenceAdapter(response_path=tmp_path / "unused.json"))
    next_hint = runtime._next_hint(
        final_decision="tool_call",
        round_status="tool_applied",
        warnings=[],
        session_state={"allowed_next_actions": ["move_anchor", "restart_path"]},
        current_segment_context={
            "focus": {"segment_id": "S1"},
            "status": {
                "segment_id": "S1",
                "status": "needs_anchor_correction",
                "may_advance_to_next_segment": False,
                "refinement_limit_reached": False,
            },
            "anchor_quality": {
                "current_segment": {
                    "segment_id": "S1",
                    "from_anchor": {"status": "bad"},
                    "to_anchor": {"status": "ok"},
                }
            },
            "quality_delta": {},
            "refinement_summary": {},
        },
    ).lower()
    assert "anchor far from the black contour" in next_hint
    assert "do not adjust handles" in next_hint
    assert "move_anchor" in next_hint
    assert "restart_path" in next_hint


def test_system_prompt_has_no_decision_tool_call_json_protocol() -> None:
    from services.free_pen_prompt import build_free_pen_tool_system_prompt

    system_prompt = build_free_pen_tool_system_prompt()
    assert "Return JSON only" not in system_prompt
    assert '"decision"' not in system_prompt
    assert '"tool_call"' not in system_prompt
    assert "Response format for tool_call" not in system_prompt
    assert "Use function tool calls only" in system_prompt
    assert "Do not write JSON manually" in system_prompt
    assert "reason" in system_prompt


def test_system_prompt_mentions_human_pen_anchor_handle_workflow() -> None:
    from services.free_pen_prompt import build_free_pen_tool_system_prompt

    system_prompt = build_free_pen_tool_system_prompt()
    assert "anchor" in system_prompt.lower()
    assert "handle" in system_prompt.lower()
    assert "move_handle" in system_prompt
    assert "set_segment_handles" in system_prompt
    assert "exactly one provided function tool per round" in system_prompt.lower()


def test_system_prompt_enforces_current_segment_refinement() -> None:
    from services.free_pen_prompt import build_free_pen_tool_system_prompt

    system_prompt = build_free_pen_tool_system_prompt()
    assert "After each curve_to" in system_prompt
    assert "Only continue to the next segment after the current segment is visually acceptable." in system_prompt
    assert "prefer set_segment_handles or move_handle" in system_prompt.lower()
    assert "exactly one provided function tool" in system_prompt.lower()
    assert "convert_line_to_curve" in system_prompt


def test_system_prompt_mentions_sequential_segment_rule() -> None:
    from services.free_pen_prompt import build_free_pen_tool_system_prompt

    system_prompt = build_free_pen_tool_system_prompt()
    assert "Do not move on to the next segment" in system_prompt
    assert "newest/current segment" in system_prompt
    assert "Do not plan to come back later" in system_prompt


def test_system_prompt_mentions_local_refinement_failure_rule() -> None:
    from services.free_pen_prompt import build_free_pen_tool_system_prompt

    system_prompt = build_free_pen_tool_system_prompt()
    assert "Do not adjust the same segment forever" in system_prompt
    assert "change strategy" in system_prompt.lower()
    assert "rollback_to_step" in system_prompt
    assert "restart_path" in system_prompt


def test_system_prompt_mentions_anchor_validation_rule() -> None:
    from services.free_pen_prompt import build_free_pen_tool_system_prompt

    system_prompt = build_free_pen_tool_system_prompt()
    assert "start_path and restart_path" in system_prompt
    assert "curve_to endpoint p" in system_prompt
    assert "Control handles c1/c2 may leave the contour" in system_prompt
    assert "If an anchor is far from the BLACK contour, do not adjust handles" in system_prompt
