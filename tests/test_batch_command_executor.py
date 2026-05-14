from pathlib import Path

from core.document import add_path, add_segment, create_document
from core.types import CoordinateSystem, Path as VectorPath, Segment
from services.batch_command_executor import BatchCommandExecutor
from services.command_executor import CommandExecutionResult


def _build_document_with_paths() -> object:
    document = create_document(
        document_id="doc_batch",
        width=200.0,
        height=200.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )

    path_1_points = ((0.0, 0.0), (2.0, 0.1), (4.0, -0.1), (6.0, 0.0))
    path_2_points = ((10.0, 0.0), (12.0, -0.05), (14.0, 0.05), (16.0, 0.0))
    locked_points = ((20.0, 0.0), (22.0, 0.0), (24.0, 0.0))

    document = add_path(document, VectorPath(path_id="path_1", segments=("path_1_seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="path_1_seg_1",
            path_id="path_1",
            type="polyline",
            params={"points": [[float(x), float(y)] for x, y in path_1_points]},
        ),
    )

    document = add_path(document, VectorPath(path_id="path_2", segments=("path_2_seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="path_2_seg_1",
            path_id="path_2",
            type="polyline",
            params={"points": [[float(x), float(y)] for x, y in path_2_points]},
        ),
    )

    document = add_path(document, VectorPath(path_id="locked_path", locked=True, segments=("locked_path_seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="locked_path_seg_1",
            path_id="locked_path",
            type="polyline",
            params={"points": [[float(x), float(y)] for x, y in locked_points]},
        ),
    )
    return document


def _command(tool: str, *, path_id: str, command_id: str) -> dict[str, object]:
    return {
        "command_id": command_id,
        "tool": tool,
        "path_id": path_id,
        "segment_range": [0, 0],
        "reason": "intent only",
        "confidence": 0.8,
        "requires_user_confirmation": True,
    }


class _InPlaceMutatingExecutor:
    def execute(self, command: object, document: object) -> CommandExecutionResult:
        if isinstance(command, dict) and command.get("command_id") == "mutate_then_fail":
            return CommandExecutionResult(
                success=False,
                command_id="mutate_then_fail",
                document=document,
                affected_paths=("path_1",),
                affected_segments=(),
                old_score=1.0,
                new_score=None,
                topology_status="open",
                self_intersection_count=0,
                requires_rerender=False,
                reason="forced failure",
            )

        document.segments[0].params["points"][0][0] = 999.0
        return CommandExecutionResult(
            success=True,
            command_id="mutate_then_succeed",
            document=document,
            affected_paths=("path_1",),
            affected_segments=("path_1_seg_1",),
            old_score=1.0,
            new_score=0.5,
            topology_status="open",
            self_intersection_count=0,
            requires_rerender=True,
            reason=None,
        )


def test_batch_command_executor_executes_all_successful_commands() -> None:
    document = _build_document_with_paths()
    executor = BatchCommandExecutor()
    commands = [
        _command("propose_replace_segment_with_line", path_id="path_1", command_id="cmd_1"),
        _command("propose_replace_segment_with_line", path_id="path_2", command_id="cmd_2"),
    ]

    result = executor.execute_batch(commands, document)

    assert result.batch_id.startswith("batch_")
    assert result.success_count == 2
    assert result.failure_count == 0
    assert result.rolled_back is False
    assert tuple(item.success for item in result.results) == (True, True)
    assert tuple(segment.type for segment in result.document.segments if segment.path_id in {"path_1", "path_2"}) == (
        "line",
        "line",
    )


def test_batch_command_executor_continues_after_partial_failure() -> None:
    document = _build_document_with_paths()
    original_document = document
    executor = BatchCommandExecutor()
    commands = [
        _command("propose_replace_segment_with_line", path_id="path_1", command_id="cmd_success"),
        _command("propose_replace_segment_with_line", path_id="locked_path", command_id="cmd_fail"),
        _command("propose_replace_segment_with_line", path_id="path_2", command_id="cmd_after_fail"),
    ]

    result = executor.execute_batch(commands, document)

    assert result.success_count == 2
    assert result.failure_count == 1
    assert tuple(item.success for item in result.results) == (True, False, True)
    assert "locked path" in (result.results[1].reason or "")
    path_1_segment = next(segment for segment in result.document.segments if segment.path_id == "path_1")
    path_2_segment = next(segment for segment in result.document.segments if segment.path_id == "path_2")
    locked_segment = next(segment for segment in result.document.segments if segment.path_id == "locked_path")
    assert path_1_segment.type == "line"
    assert path_2_segment.type == "line"
    assert locked_segment.type == "polyline"
    assert document == original_document


def test_batch_command_executor_continues_when_first_command_fails() -> None:
    document = _build_document_with_paths()
    executor = BatchCommandExecutor()
    commands = [
        _command("propose_replace_segment_with_line", path_id="locked_path", command_id="cmd_fail_first"),
        _command("propose_replace_segment_with_line", path_id="path_1", command_id="cmd_success_second"),
    ]

    result = executor.execute_batch(commands, document)

    assert result.success_count == 1
    assert result.failure_count == 1
    assert tuple(item.success for item in result.results) == (False, True)
    assert next(segment for segment in result.document.segments if segment.path_id == "path_1").type == "line"


def test_batch_command_executor_rolls_back_after_failure_when_enabled() -> None:
    document = _build_document_with_paths()
    original_document = document
    executor = BatchCommandExecutor()
    commands = [
        _command("propose_replace_segment_with_line", path_id="path_1", command_id="cmd_success_before_rollback"),
        _command("propose_replace_segment_with_line", path_id="locked_path", command_id="cmd_failure"),
        _command("propose_replace_segment_with_line", path_id="path_2", command_id="cmd_not_executed"),
    ]

    result = executor.execute_batch(commands, document, rollback_batch_on_failure=True)

    assert result.success_count == 1
    assert result.failure_count == 1
    assert result.rolled_back is True
    assert len(result.results) == 2
    assert tuple(item.command_id for item in result.results) == (
        "cmd_success_before_rollback",
        "cmd_failure",
    )
    assert result.document == original_document


def test_batch_command_executor_rolls_back_to_deep_snapshot_after_in_place_mutation() -> None:
    document = create_document(
        document_id="doc_mutation",
        width=100.0,
        height=100.0,
        coordinate_system=CoordinateSystem(internal_space="vector"),
    )
    document = add_path(document, VectorPath(path_id="path_1", segments=("path_1_seg_1",)))
    document = add_segment(
        document,
        Segment(
            segment_id="path_1_seg_1",
            path_id="path_1",
            type="polyline",
            params={"points": [[0.0, 0.0], [1.0, 0.0]]},
        ),
    )
    executor = BatchCommandExecutor(command_executor=_InPlaceMutatingExecutor())
    commands = [
        {"command_id": "mutate_then_succeed"},
        {"command_id": "mutate_then_fail"},
    ]

    result = executor.execute_batch(commands, document, rollback_batch_on_failure=True)

    assert result.rolled_back is True
    assert result.document.segments[0].params["points"] == [[0.0, 0.0], [1.0, 0.0]]
    assert result.document is not document
    assert result.success_count == 1
    assert result.failure_count == 1
    assert len(result.results) == 2


def test_batch_command_executor_continues_after_non_dict_command_failure() -> None:
    document = _build_document_with_paths()
    executor = BatchCommandExecutor()
    commands = [
        None,
        _command("propose_replace_segment_with_line", path_id="path_1", command_id="cmd_valid_after_none"),
    ]

    result = executor.execute_batch(commands, document, continue_on_failure=True)

    assert result.success_count == 1
    assert result.failure_count == 1
    assert tuple(item.success for item in result.results) == (False, True)
    assert "command must be a dictionary" in (result.results[0].reason or "")
    assert next(segment for segment in result.document.segments if segment.path_id == "path_1").type == "line"


def test_batch_command_executor_has_no_forbidden_dependencies() -> None:
    source = Path("services/batch_command_executor.py").read_text(encoding="utf-8")
    assert "cv2" not in source
    assert "PyQt" not in source
    assert "openai" not in source
