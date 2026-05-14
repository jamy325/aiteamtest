from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import uuid

from core.types import VectorDocument
from services.command_executor import CommandExecutionResult, CommandExecutor


@dataclass(frozen=True, slots=True)
class BatchCommandItemResult:
    command_index: int
    command_id: str
    success: bool
    reason: str | None
    old_score: float | None
    new_score: float | None
    affected_paths: tuple[str, ...]
    affected_segments: tuple[str, ...]
    requires_rerender: bool


@dataclass(frozen=True, slots=True)
class BatchCommandExecutionResult:
    batch_id: str
    success_count: int
    failure_count: int
    results: tuple[BatchCommandItemResult, ...]
    document: VectorDocument
    rolled_back: bool


class BatchCommandExecutor:
    def __init__(self, command_executor: CommandExecutor | None = None) -> None:
        self.command_executor = command_executor or CommandExecutor()

    def execute_batch(
        self,
        commands: list[object] | tuple[object, ...],
        document: VectorDocument,
        *,
        continue_on_failure: bool = True,
        rollback_batch_on_failure: bool = False,
    ) -> BatchCommandExecutionResult:
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        current_document = document
        rollback_document = deepcopy(document) if rollback_batch_on_failure else document
        item_results: list[BatchCommandItemResult] = []
        failure_seen = False

        for index, command in enumerate(commands):
            execution_result = self.command_executor.execute(command, current_document)
            item_results.append(self._item_result(index, execution_result))

            if execution_result.success:
                current_document = execution_result.document
                continue

            failure_seen = True
            if rollback_batch_on_failure:
                current_document = rollback_document
                break
            if not continue_on_failure:
                break

        if rollback_batch_on_failure and failure_seen:
            current_document = rollback_document

        success_count = sum(1 for result in item_results if result.success)
        failure_count = len(item_results) - success_count
        return BatchCommandExecutionResult(
            batch_id=batch_id,
            success_count=success_count,
            failure_count=failure_count,
            results=tuple(item_results),
            document=current_document,
            rolled_back=rollback_batch_on_failure and failure_seen,
        )

    def _item_result(self, index: int, execution_result: CommandExecutionResult) -> BatchCommandItemResult:
        return BatchCommandItemResult(
            command_index=index,
            command_id=execution_result.command_id,
            success=execution_result.success,
            reason=execution_result.reason,
            old_score=execution_result.old_score,
            new_score=execution_result.new_score,
            affected_paths=execution_result.affected_paths,
            affected_segments=execution_result.affected_segments,
            requires_rerender=execution_result.requires_rerender,
        )


__all__ = [
    "BatchCommandExecutionResult",
    "BatchCommandExecutor",
    "BatchCommandItemResult",
]
