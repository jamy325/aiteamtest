# Sample Inputs And Expected Artifacts

## Included Inputs

- `samples/inputs/circle_quickstart.png`
- `samples/inputs/ellipse_quickstart.png`

These are intentionally small fixtures so the release-candidate quickstart stays fast and versionable.

## Circle Quickstart

Recommended command:

```bash
python -m vector_reconstruction run \
  --input samples/inputs/circle_quickstart.png \
  --output out/quickstart-circle \
  --dry-run-only \
  --max-iterations 1 \
  --target-types circle
```

Expected artifact bundle:

- `document.json`
- `output.svg`
- `output.dxf`
- `overlay.png`
- `diff.png`
- `decision_report.json`
- `metrics.json`

Expected behavior:

- command completes offline
- stdout returns a JSON payload with `ok: true`
- `metrics.json` and `decision_report.json` exist even if the run escalates to `requires_external_decision`

## Ellipse Quickstart

Recommended command:

```bash
python -m vector_reconstruction run \
  --input samples/inputs/ellipse_quickstart.png \
  --output out/quickstart-ellipse \
  --dry-run-only \
  --max-iterations 1 \
  --target-types ellipse
```

Expected artifact bundle:

- `document.json`
- `output.svg`
- `output.dxf`
- `overlay.png`
- `diff.png`
- `decision_report.json`
- `metrics.json`

Expected behavior:

- command completes offline
- artifact directory is populated
- outputs are suitable for smoke validation, not golden-geometry guarantees
