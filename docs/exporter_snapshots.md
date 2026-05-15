# Exporter Snapshot Tests

This project keeps golden snapshot fixtures for SVG and DXF exporters to catch
accidental output regressions when geometry, style, coordinate, or exporter code
changes.

## Scope

Current snapshot coverage includes:

- line + bezier SVG path output
- arc / circle / ellipse export
- closed compound path SVG output with `fill-rule="evenodd"`
- style fields such as `fill_color`, `fill_alpha`, `stroke_color`,
  `stroke_width`, and `opacity`
- DXF `px_to_mm` scaling
- DXF `y_axis` flip behavior

## Update policy

Do not overwrite snapshot baselines casually.

Only update snapshot files when:

1. exporter behavior is intentionally changed, and
2. the new output has been reviewed as the desired canonical format.

If a snapshot test fails:

1. inspect the unified diff in the failing test output
2. confirm whether the change is intentional
3. update the golden file only after that review

## Running the checks

```bash
python -m pytest tests/test_svg_exporter.py tests/test_dxf_exporter.py -q
python -m pytest -q
```
