from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SUITE_REPORT_FILENAMES: tuple[str, ...] = (
    "acceptance_report.json",
    "real_world_regression_report.json",
)


@dataclass(frozen=True, slots=True)
class HtmlCaseReportResult:
    case_id: str
    report_path: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HtmlSuiteReportResult:
    index_path: str
    case_reports: tuple[HtmlCaseReportResult, ...]
    warnings: tuple[str, ...]


class HtmlQualityReportGenerator:
    def generate_case_report(
        self,
        case_dir: str | Path,
        *,
        output_path: str | Path | None = None,
        suite_report_path: str | Path | None = None,
    ) -> HtmlCaseReportResult:
        case_root = Path(case_dir)
        warnings: list[str] = []
        case_id = case_root.name
        suite_payload = self._load_suite_payload(case_root.parent, suite_report_path=suite_report_path, warnings=warnings)
        case_payload = self._suite_case_payload(suite_payload, case_id)

        metrics = self._load_json(case_root / "metrics.json", warnings=warnings)
        decision_report = self._load_json(case_root / "decision_report.json", warnings=warnings)
        output_document = self._load_json(case_root / "output.json", warnings=warnings, required=False)

        output_file = Path(output_path) if output_path is not None else case_root / "report.html"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            self._render_case_html(
                case_root=case_root,
                case_id=case_id,
                case_payload=case_payload,
                metrics=metrics if isinstance(metrics, Mapping) else {},
                decision_report=decision_report if isinstance(decision_report, Mapping) else {},
                output_document=output_document if isinstance(output_document, Mapping) else {},
                warnings=warnings,
            ),
            encoding="utf-8",
        )
        return HtmlCaseReportResult(case_id=case_id, report_path=str(output_file), warnings=tuple(warnings))

    def generate_suite_report(
        self,
        suite_dir: str | Path,
        *,
        output_path: str | Path | None = None,
    ) -> HtmlSuiteReportResult:
        root = Path(suite_dir)
        warnings: list[str] = []
        suite_payload = self._load_suite_payload(root, suite_report_path=None, warnings=warnings)
        case_ids = self._suite_case_ids(root, suite_payload)
        case_reports: list[HtmlCaseReportResult] = []
        for case_id in case_ids:
            case_result = self.generate_case_report(
                root / case_id,
                output_path=(root / case_id / "report.html"),
                suite_report_path=self._suite_report_path(root),
            )
            case_reports.append(case_result)
            for warning in case_result.warnings:
                warnings.append(f"{case_id}: {warning}")

        output_file = Path(output_path) if output_path is not None else root / "index.html"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            self._render_index_html(root=root, suite_payload=suite_payload, case_reports=case_reports, warnings=warnings),
            encoding="utf-8",
        )
        return HtmlSuiteReportResult(index_path=str(output_file), case_reports=tuple(case_reports), warnings=tuple(warnings))

    def _load_suite_payload(
        self,
        root: Path,
        *,
        suite_report_path: str | Path | None,
        warnings: list[str],
    ) -> Mapping[str, Any] | None:
        path = Path(suite_report_path) if suite_report_path is not None else self._suite_report_path(root)
        if path is None:
            warnings.append("missing_suite_report")
            return None
        payload = self._load_json(path, warnings=warnings)
        return payload if isinstance(payload, Mapping) else None

    def _suite_report_path(self, root: Path) -> Path | None:
        for filename in SUITE_REPORT_FILENAMES:
            candidate = root / filename
            if candidate.exists():
                return candidate
        return None

    def _suite_case_payload(self, suite_payload: Mapping[str, Any] | None, case_id: str) -> Mapping[str, Any] | None:
        if suite_payload is None:
            return None
        cases = suite_payload.get("cases")
        if not isinstance(cases, list):
            return None
        for item in cases:
            if isinstance(item, Mapping) and item.get("case_id") == case_id:
                return item
        return None

    def _suite_case_ids(self, root: Path, suite_payload: Mapping[str, Any] | None) -> tuple[str, ...]:
        ids = {path.name for path in root.iterdir() if path.is_dir()}
        if suite_payload is not None:
            cases = suite_payload.get("cases")
            if isinstance(cases, list):
                for item in cases:
                    if isinstance(item, Mapping) and isinstance(item.get("case_id"), str):
                        ids.add(str(item["case_id"]))
        return tuple(sorted(ids))

    def _render_case_html(
        self,
        *,
        case_root: Path,
        case_id: str,
        case_payload: Mapping[str, Any] | None,
        metrics: Mapping[str, Any],
        decision_report: Mapping[str, Any],
        output_document: Mapping[str, Any],
        warnings: Sequence[str],
    ) -> str:
        input_image = self._input_image_path(case_root, case_payload)
        overlay_image = self._relative_asset(case_root, case_root / "overlay.png")
        diff_image = self._relative_asset(case_root, case_root / "diff.png")
        svg_preview = self._inline_svg(case_root / "output.svg", warnings)
        failure_reason = decision_report.get("failure_reason") or (case_payload.get("failure_reason") if case_payload is not None else None)
        decision_counts = decision_report.get("decision_counts")
        timeline = decision_report.get("preview_decisions")

        sections = [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            f"<title>{escape(case_id)} Quality Report</title>",
            "<style>",
            self._css(),
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{escape(case_id)} Quality Report</h1>",
            f"<p class=\"meta\">Failure reason: <strong>{escape(str(failure_reason or 'None'))}</strong></p>",
            self._warning_block(warnings),
            "<section class=\"grid\">",
            self._image_card("Input Image", input_image, missing_label="Input image unavailable"),
            self._image_card("Overlay", overlay_image, missing_label="Overlay unavailable"),
            self._image_card("Diff", diff_image, missing_label="Diff unavailable"),
            self._svg_card("SVG Preview", svg_preview),
            "</section>",
            "<section>",
            "<h2>Metrics</h2>",
            self._metrics_table(metrics),
            "</section>",
            "<section>",
            "<h2>Stroke Summary</h2>",
            self._stroke_summary(metrics=metrics, output_document=output_document),
            "</section>",
            "<section>",
            "<h2>Decision Summary</h2>",
            self._decision_counts_block(decision_counts),
            "</section>",
            "<section>",
            "<h2>Decision Timeline</h2>",
            self._decision_timeline(timeline),
            "</section>",
            "<section>",
            "<h2>Document Summary</h2>",
            self._document_summary(output_document),
            "</section>",
            "</body>",
            "</html>",
        ]
        return "\n".join(sections)

    def _render_index_html(
        self,
        *,
        root: Path,
        suite_payload: Mapping[str, Any] | None,
        case_reports: Sequence[HtmlCaseReportResult],
        warnings: Sequence[str],
    ) -> str:
        summary = suite_payload.get("summary") if isinstance(suite_payload, Mapping) else {}
        cases = suite_payload.get("cases") if isinstance(suite_payload, Mapping) else []
        case_map = {
            str(item["case_id"]): item
            for item in cases
            if isinstance(item, Mapping) and isinstance(item.get("case_id"), str)
        } if isinstance(cases, list) else {}

        rows = []
        for report in case_reports:
            payload = case_map.get(report.case_id, {})
            success = payload.get("success")
            failure_reason = payload.get("failure_reason")
            metrics = payload.get("metrics")
            total_score = None
            if isinstance(metrics, Mapping):
                total_score = metrics.get("total_score")
            relative_report = Path(report.report_path).relative_to(root)
            rows.append(
                "<tr>"
                f"<td><a href=\"{escape(str(relative_report).replace('\\', '/'))}\">{escape(report.case_id)}</a></td>"
                f"<td>{escape(str(success))}</td>"
                f"<td>{escape(str(total_score))}</td>"
                f"<td>{escape(str(failure_reason or ''))}</td>"
                f"<td>{len(report.warnings)}</td>"
                "</tr>"
            )

        html_parts = [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<title>Benchmark Quality Report Index</title>",
            "<style>",
            self._css(),
            "</style>",
            "</head>",
            "<body>",
            "<h1>Benchmark Quality Report Index</h1>",
            f"<p class=\"meta\">Overall pass: <strong>{escape(str(summary.get('overall_pass')))}</strong></p>",
            f"<p class=\"meta\">Failed cases: <strong>{escape(str(summary.get('failed_case_count')))}</strong></p>",
            self._warning_block(warnings),
            "<table>",
            "<thead><tr><th>Case</th><th>Success</th><th>Total Score</th><th>Failure Reason</th><th>Warnings</th></tr></thead>",
            "<tbody>",
            *(rows if rows else ["<tr><td colspan=\"5\">No case reports found.</td></tr>"]),
            "</tbody>",
            "</table>",
            "</body>",
            "</html>",
        ]
        return "\n".join(html_parts)

    def _input_image_path(self, case_root: Path, case_payload: Mapping[str, Any] | None) -> str | None:
        if case_payload is None:
            return None
        image_path = case_payload.get("image_path")
        if not isinstance(image_path, str):
            return None
        resolved = Path(image_path)
        if not resolved.is_absolute():
            resolved = (case_root.parent / image_path).resolve()
        if not resolved.exists():
            return None
        return self._relative_asset(case_root, resolved)

    def _relative_asset(self, case_root: Path, asset_path: Path) -> str | None:
        if not asset_path.exists():
            return None
        try:
            relative = asset_path.relative_to(case_root)
            return str(relative).replace("\\", "/")
        except ValueError:
            try:
                relative = asset_path.relative_to(case_root.parent)
                return str(relative).replace("\\", "/")
            except ValueError:
                return asset_path.as_uri()

    def _inline_svg(self, svg_path: Path, warnings: list[str]) -> str | None:
        if not svg_path.exists():
            warnings.append("missing_output_svg")
            return None
        try:
            return svg_path.read_text(encoding="utf-8")
        except OSError as exc:
            warnings.append(f"invalid_output_svg: {exc}")
            return None

    def _warning_block(self, warnings: Sequence[str]) -> str:
        if not warnings:
            return "<section><h2>Warnings</h2><p>None</p></section>"
        items = "".join(f"<li>{escape(item)}</li>" for item in warnings)
        return f"<section><h2>Warnings</h2><ul>{items}</ul></section>"

    def _image_card(self, title: str, image_path: str | None, *, missing_label: str) -> str:
        if image_path is None:
            content = f"<div class=\"missing\">{escape(missing_label)}</div>"
        else:
            content = f"<img src=\"{escape(image_path)}\" alt=\"{escape(title)}\">"
        return f"<article class=\"card\"><h2>{escape(title)}</h2>{content}</article>"

    def _svg_card(self, title: str, svg_payload: str | None) -> str:
        if svg_payload is None:
            content = "<div class=\"missing\">SVG preview unavailable</div>"
        else:
            content = f"<div class=\"svg-preview\">{svg_payload}</div>"
        return f"<article class=\"card\"><h2>{escape(title)}</h2>{content}</article>"

    def _metrics_table(self, metrics: Mapping[str, Any]) -> str:
        if not metrics:
            return "<p>No metrics available.</p>"
        rows = []
        for key in sorted(metrics.keys()):
            rows.append(f"<tr><th>{escape(str(key))}</th><td>{escape(str(metrics[key]))}</td></tr>")
        return "<table><tbody>" + "".join(rows) + "</tbody></table>"

    def _decision_counts_block(self, decision_counts: object) -> str:
        if not isinstance(decision_counts, Mapping):
            return "<p>No decision summary available.</p>"
        items = []
        for key in ("auto_apply", "auto_reject", "requires_external_decision"):
            items.append(f"<li>{escape(key)}: <strong>{escape(str(decision_counts.get(key, 0)))}</strong></li>")
        return "<ul>" + "".join(items) + "</ul>"

    def _decision_timeline(self, preview_decisions: object) -> str:
        if not isinstance(preview_decisions, list) or not preview_decisions:
            return "<p>No preview decisions available.</p>"
        items = []
        for item in preview_decisions:
            if not isinstance(item, Mapping):
                continue
            decision = str(item.get("decision_kind") or item.get("decision") or "unknown")
            command = item.get("command")
            tool = command.get("tool") if isinstance(command, Mapping) else None
            feedback = item.get("policy_feedback")
            reason_code = feedback.get("reason_code") if isinstance(feedback, Mapping) else None
            failure_reason = item.get("reason")
            class_name = {
                "auto_apply": "status-apply",
                "auto_reject": "status-reject",
                "requires_external_decision": "status-review",
            }.get(decision, "status-other")
            items.append(
                "<li class=\"timeline-item\">"
                f"<span class=\"badge {class_name}\">{escape(decision)}</span> "
                f"<strong>{escape(str(tool or 'unknown_tool'))}</strong> "
                f"<span>{escape(str(reason_code or failure_reason or ''))}</span>"
                "</li>"
            )
        return "<ul class=\"timeline\">" + "".join(items) + "</ul>"

    def _document_summary(self, output_document: Mapping[str, Any]) -> str:
        if not output_document:
            return "<p>No output document metadata available.</p>"
        keys = []
        for key in ("document_id", "width", "height", "paths", "segments", "constraints"):
            if key in output_document:
                value = output_document[key]
                if isinstance(value, list):
                    value = len(value)
                keys.append(f"<li>{escape(str(key))}: <strong>{escape(str(value))}</strong></li>")
        return "<ul>" + "".join(keys or ["<li>No document summary fields available.</li>"]) + "</ul>"

    def _stroke_summary(self, *, metrics: Mapping[str, Any], output_document: Mapping[str, Any]) -> str:
        metric_items: list[str] = []
        for key in ("stroke_width", "stroke_width_confidence", "stroke_mask_error"):
            if key in metrics:
                metric_items.append(f"<li>{escape(key)}: <strong>{escape(str(metrics[key]))}</strong></li>")

        paths = output_document.get("paths")
        endpoint_count = 0
        junction_count = 0
        branch_count = 0
        if isinstance(paths, list):
            for path in paths:
                if not isinstance(path, Mapping):
                    continue
                metadata = path.get("metadata")
                if not isinstance(metadata, Mapping):
                    continue
                endpoint_count += int(metadata.get("endpoint_count", 0))
                junction_count += int(metadata.get("junction_count", 0))
                branch_count += int(metadata.get("branch_count", 0))

        topology_items = [
            f"<li>endpoint_count: <strong>{endpoint_count}</strong></li>",
            f"<li>junction_count: <strong>{junction_count}</strong></li>",
            f"<li>branch_count: <strong>{branch_count}</strong></li>",
        ]
        if not metric_items and endpoint_count == 0 and junction_count == 0 and branch_count == 0:
            return "<p>No stroke summary available.</p>"
        return "<ul>" + "".join(metric_items + topology_items) + "</ul>"

    def _load_json(self, path: Path, *, warnings: list[str], required: bool = True) -> Any:
        if not path.exists():
            if required:
                warnings.append(f"missing_file: {path.name}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"invalid_json: {path.name}: {exc}")
            return None

    def _css(self) -> str:
        return """
body { font-family: "Segoe UI", Arial, sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }
h1, h2 { margin: 0 0 12px; }
.meta { color: #475569; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
.card, section { background: white; border: 1px solid #dbe4ee; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
img { max-width: 100%; border-radius: 8px; border: 1px solid #cbd5e1; background: #fff; }
.svg-preview svg { width: 100%; height: auto; border: 1px solid #cbd5e1; border-radius: 8px; background: white; }
.missing { padding: 24px; border: 1px dashed #94a3b8; border-radius: 8px; color: #64748b; background: #f8fafc; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
.timeline { list-style: none; padding: 0; margin: 0; }
.timeline-item { padding: 10px 0; border-bottom: 1px solid #e2e8f0; }
.badge { display: inline-block; min-width: 150px; margin-right: 8px; padding: 2px 8px; border-radius: 999px; font-size: 12px; color: white; }
.status-apply { background: #15803d; }
.status-reject { background: #b91c1c; }
.status-review { background: #92400e; }
.status-other { background: #475569; }
a { color: #0f766e; text-decoration: none; }
a:hover { text-decoration: underline; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } .badge { display: inline-block; margin-bottom: 6px; } }
"""


__all__ = [
    "HtmlCaseReportResult",
    "HtmlQualityReportGenerator",
    "HtmlSuiteReportResult",
]
