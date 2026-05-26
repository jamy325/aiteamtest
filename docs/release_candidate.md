# V3.5 Release Candidate

## Scope

V3.5 packages the current autonomous vector reconstruction engine into a deliverable developer-facing release candidate.

Current release-candidate surface:

- offline-first CLI entrypoint
- artifact bundle generation
- acceptance benchmark suite
- real-world regression suite
- quality profile calibration outputs
- AI review adapters with mock, file, and recorded replay support
- external decision queue data model and persistence

## Primary Entry Points

Image-to-artifact bundle:

```bash
python -m vector_reconstruction run \
  --input samples/inputs/circle_quickstart.png \
  --output out/quickstart-circle \
  --dry-run-only \
  --max-iterations 1 \
  --target-types circle
```

Acceptance benchmark:

```bash
python scripts/run_acceptance_benchmark.py \
  --manifest benchmarks/acceptance_manifest.json \
  --output-dir out/acceptance
```

Real-world regression:

```bash
python scripts/run_real_world_regression.py \
  --manifest benchmarks/real_samples/manifest.json \
  --baseline benchmarks/baselines/real_world_regression_baseline.json \
  --output-dir out/real-world-regression
```

## Autonomy Levels

- `manual_only`
  - never auto-applies a preview result
  - useful when integrating an external decision consumer first
- `assisted`
  - allows deterministic preview and scoring, but escalates more often
- `autonomous_safe`
  - default CLI level
  - applies hard safety gates before auto-apply
- `autonomous_full`
  - highest automation level
  - still does not bypass hard rejects such as topology regression or locked-target violations

## Quality Gates

Current quality gating combines:

- algorithm fitting confidence
- inlier ratio
- fit error
- topology status and self-intersection counts
- complexity and edge-error deltas
- benchmark fail thresholds

Seed quality profiles live in:

- [configs/quality_profiles.json](../configs/quality_profiles.json)
- [configs/release_candidate_quality_profile.sample.json](../configs/release_candidate_quality_profile.sample.json)

These profiles are intentionally conservative templates. Manifest-level thresholds remain the source of truth for benchmark pass/fail overrides.

## AI Provider Configuration

The release candidate keeps quickstart offline by default.

The standard CLI currently does not expose provider flags directly. Provider setup is for integrators embedding `AIReviewService` or custom wrappers around the engine.

Recommended provider modes:

- `mock` for unit tests
- `file` for deterministic local replay
- `recorded_mode="replay"` for provider-shaped regression without live API traffic
- `recorded_mode="record"` only when explicitly enabled

Example provider template:

- [configs/release_candidate_ai_provider.sample.json](../configs/release_candidate_ai_provider.sample.json)

Live record mode must be explicitly enabled. Default quickstart and default pytest must not require network access or real API keys.

## Benchmark Packaging

Use the sample benchmark template as a release-candidate starting point:

- [configs/release_candidate_benchmark.sample.json](../configs/release_candidate_benchmark.sample.json)

The acceptance suite writes per-case artifacts plus a suite summary. The regression suite compares current outputs against a committed baseline.

## Sample Inputs

Bundled lightweight inputs:

- [samples/inputs/circle_quickstart.png](../samples/inputs/circle_quickstart.png)
- [samples/inputs/ellipse_quickstart.png](../samples/inputs/ellipse_quickstart.png)

Expected artifact notes:

- [samples/expected_artifacts.md](../samples/expected_artifacts.md)

## Known Limitations

Current release-candidate limitations:

- complex boolean decomposition is limited and may require external decision handling
- G2 or higher-order continuity is not solved; only current G1-oriented constraints are covered
- gradient, textured fill, and advanced style reconstruction are not modeled as first-class outputs
- higher-order constraint solving remains limited compared with a full CAD-grade solver
- bezier fallback is available, but still depends on conservative escalation and benchmark validation
- standard CLI does not yet expose provider config, external decision queue operations, or benchmark orchestration in one unified command

These are intentional boundaries for V3.5 and should not be presented as fully solved capabilities.

## Troubleshooting

### Provider configuration

- If you want deterministic AI review without a live API, use recorded replay or file/mock adapters.
- If a live provider complains about missing keys, do not add keys to docs examples; use replay for the release-candidate path.
- If replay fails with a missing fixture error, generate the fixture explicitly in record mode and commit only sanitized metadata plus response JSON.

### Output directory is empty or missing artifacts

- Verify `--input` points to a real image file.
- Check stderr for the CLI's structured JSON error payload.
- Use `--dry-run-only` first to confirm the engine can produce the artifact bundle without committing further refinement.

### Quality gate failures

- Inspect `metrics.json` and `decision_report.json` in the artifact bundle.
- Compare against the configured target types and quality-profile thresholds.
- Re-run with a narrower `--target-types` scope before widening geometry classes.

### External decision handling

- `requires_external_decision` is not a fatal crash.
- It means the engine intentionally stopped at a policy boundary.
- Persist and review the decision request through `ExternalDecisionQueue` if your integration uses queued approval.

## Architecture Reminder

This release candidate still follows the core design constraints:

- all core fitting and scoring run in vector space
- `VectorDocument` remains a pure data layer
- AI only proposes semantic modification intent
- deterministic modules own geometry solving, validation, topology, and export
