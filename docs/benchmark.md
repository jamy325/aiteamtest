# Benchmark

`BenchmarkRunner` supports an explicit `auto_refine` mode for quality-gate runs.
The repository-level `benchmark_manifest.json` is intended to stay green by default; failure-only gates should live in unit tests or a separate negative manifest.

Manifest fields:

- `auto_refine`: enable the end-to-end auto refinement pipeline for the case.
- `auto_refine_target_types`: optional primitive filter such as `["circle", "ellipse"]`.
- `auto_refine_dry_run_only`: run preview policy and reporting without mutating the final document.
- `fail_thresholds`: numeric quality gates such as `max_total_score`, `min_circle_count`, or `max_segment_count`.

Report highlights:

- `candidate_counts`
- `proposed_command_counts`
- `preview_decisions`
- `accepted_count`
- `rejected_count`
- `user_confirm_count`
- `geometry_before`
- `geometry_after`
- `score_before`
- `score_after`
- `score_delta`
- `export_summary.svg.path_command_counts`
- `export_summary.dxf.entity_counts`
