# Curve Fitting AI Agent

V3.5 release candidate for an autonomous vector reconstruction engine.

The repository converts a raster image into a vector-space reconstruction flow and writes an artifact bundle containing:

- `document.json`
- `output.svg`
- `output.dxf`
- `overlay.png`
- `diff.png`
- `decision_report.json`
- `metrics.json`

## Quickstart

Install the project in editable mode:

```bash
python -m pip install -e .[dev]
```

Run the offline quickstart on the bundled circle sample:

```bash
python -m vector_reconstruction run \
  --input samples/inputs/circle_quickstart.png \
  --output out/quickstart-circle \
  --dry-run-only \
  --max-iterations 1 \
  --target-types circle
```

Run the offline quickstart on the bundled ellipse sample:

```bash
python -m vector_reconstruction run \
  --input samples/inputs/ellipse_quickstart.png \
  --output out/quickstart-ellipse \
  --dry-run-only \
  --max-iterations 1 \
  --target-types ellipse
```

Both commands are offline by default. They do not require a real API key and do not enable live AI review.

## Output Artifacts

Each run writes a bundle to the `--output` directory:

- `document.json`: serialized `VectorDocument`
- `output.svg`: SVG export
- `output.dxf`: DXF export
- `overlay.png`: rendered overlay preview
- `diff.png`: distance-field diff preview
- `decision_report.json`: engine and policy report
- `metrics.json`: summary metrics for scoring and automation decisions

## CLI Notes

Current standard entrypoint:

```bash
python -m vector_reconstruction run --input <image> --output <artifact-dir>
```

Important flags:

- `--autonomy manual_only|assisted|autonomous_safe|autonomous_full`
- `--max-iterations <n>`
- `--target-types circle rectangle ellipse arc line`
- `--dry-run-only`
- `--enable-ai-review`
- `--document-id <id>`

The standard CLI is intentionally offline-first. It does not require network access unless you integrate an `AIReviewService` and explicitly enable live provider usage in your own wrapper.

## Sample Inputs And Configs

- Sample inputs: [samples/inputs/circle_quickstart.png](/d:/works/curve-fitting-ai-agent/samples/inputs/circle_quickstart.png), [samples/inputs/ellipse_quickstart.png](/d:/works/curve-fitting-ai-agent/samples/inputs/ellipse_quickstart.png)
- Expected artifact notes: [samples/expected_artifacts.md](/d:/works/curve-fitting-ai-agent/samples/expected_artifacts.md)
- Quality profile template: [configs/release_candidate_quality_profile.sample.json](/d:/works/curve-fitting-ai-agent/configs/release_candidate_quality_profile.sample.json)
- AI provider replay template: [configs/release_candidate_ai_provider.sample.json](/d:/works/curve-fitting-ai-agent/configs/release_candidate_ai_provider.sample.json)
- Benchmark template: [configs/release_candidate_benchmark.sample.json](/d:/works/curve-fitting-ai-agent/configs/release_candidate_benchmark.sample.json)

## Benchmarks

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

## AI Provider Modes

The repo supports provider-neutral AI review adapters, including recorded replay:

- `mock`: deterministic in-memory payload
- `file`: fixed JSON response from disk
- `recorded_mode="replay"`: replay a recorded provider response without API keys
- `recorded_mode="record"`: explicitly gated live record mode

See [docs/ai_review.md](/d:/works/curve-fitting-ai-agent/docs/ai_review.md) and [docs/release_candidate.md](/d:/works/curve-fitting-ai-agent/docs/release_candidate.md) for the release-candidate packaging details.
