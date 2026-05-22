# UI P2 Auto-Refinement Review

`MainWindow` now supports a lightweight auto-refinement review flow for tests and non-PyQt orchestration:

- run `trigger_auto_refinement_review_for_current_document()` after auto-fit
- inspect `review_display_state.candidates`, `preview_decisions`, and `diff_summary`
- render candidate / command / decision overlays through `CanvasWidget`
- apply a single pending command with `apply_user_confirm_command(decision_id)`
- apply all pending commands with `apply_all_user_confirm_commands()`
- ignore a pending command with `ignore_command(decision_id)`

The UI review layer stays on top of `AutoRefinementPipeline` and `CommandExecutor`:

- candidate detection and preview policy remain in services
- UI does not perform fitting directly
- user-confirmed commands are executed through `CommandExecutor`
- locked targets remain visible in overlay metadata and are not auto-applied

`diff_summary` reports:

- segment type counts before / after / delta
- `score_before`, `score_after`, `score_delta`
- topology status count changes
- total self-intersection changes
