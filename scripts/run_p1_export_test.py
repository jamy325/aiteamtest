import argparse
from pathlib import Path
import inspect
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.minimal_pipeline import MinimalPipeline
from services.svg_exporter import SvgExporter
from services.dxf_exporter import DxfExporter
from services.document_integrity import DocumentIntegrityValidator

parser = argparse.ArgumentParser()
parser.add_argument("--export-mode", choices=("outline", "centerline", "all_debug"), default="centerline")
args = parser.parse_args()

input_image = Path("test_input.png")
out = Path("out/p1_manual_test")
debug_out = out / "debug"
out.mkdir(parents=True, exist_ok=True)

def step(name, fn):
    t0 = time.time()
    print(f"[START] {name}")
    result = fn()
    print(f"[DONE]  {name}: {time.time() - t0:.2f}s")
    return result

pipeline = MinimalPipeline(segment_type="line")

run_from_file_signature = inspect.signature(pipeline.run_from_file)
supports_debug = "debug" in run_from_file_signature.parameters

if supports_debug:
    result = step(
        "run minimal pipeline",
        lambda: pipeline.run_from_file(
            input_image,
            document_id="manual_p1_test",
            debug=True,
            debug_output_dir=debug_out,
        ),
    )
else:
    print("[INFO] Current MinimalPipeline does not support debug artifacts on this branch.")
    print("[INFO] Checkout the branch/PR that contains Issue #133 to generate debug images.")
    result = step(
        "run minimal pipeline",
        lambda: pipeline.run_from_file(
            input_image,
            document_id="manual_p1_test",
        ),
    )
document = result.document

report = step("validate document", lambda: DocumentIntegrityValidator().validate(document))
print("integrity:", report.success)
if not report.success:
    for error in report.errors:
        print(error.code, error.affected_ids, error.message)

json_path = out / "vector_document.json"
overlay_path = out / "overlay.png"
svg_path = out / "vector_result.svg"
dxf_path = out / "vector_result.dxf"
summary_path = out / "summary.json"

step("export json", lambda: pipeline.export_json(document, json_path))
step("export overlay", lambda: pipeline.export_overlay(document, result.source_image, overlay_path))
svg_exporter = SvgExporter()
dxf_exporter = DxfExporter()
svg_report = svg_exporter.export_report(document, export_mode=args.export_mode)
dxf_report = dxf_exporter.export_report(document, export_mode=args.export_mode)
step("export svg", lambda: svg_path.write_text(svg_exporter.export_document(document, export_mode=args.export_mode), encoding="utf-8"))
step("export dxf", lambda: dxf_path.write_text(dxf_exporter.export_document(document, export_mode=args.export_mode), encoding="utf-8"))

summary = {
    "input": str(input_image),
    "document_id": document.document_id,
    "path_count": len(document.paths),
    "segment_count": len(document.segments),
    "integrity": report.success,
    "export_mode": args.export_mode,
    "files": {
        "json": str(json_path),
        "overlay": str(overlay_path),
        "svg": str(svg_path),
        "dxf": str(dxf_path),
    },
    "export": {
        "svg": svg_report,
        "dxf": dxf_report,
    },
    "debug": {
        "supported": supports_debug,
        "output_dir": str(result.debug_artifacts.output_dir) if getattr(result, "debug_artifacts", None) is not None else None,
        "files": list(result.debug_artifacts.exported_files) if getattr(result, "debug_artifacts", None) is not None else [],
    },
}
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
