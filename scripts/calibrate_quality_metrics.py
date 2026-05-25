from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.quality_metric_calibration import QualityMetricCalibrator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate quality metric profiles from existing benchmark outputs.")
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Benchmark output directory. Pass multiple times to combine acceptance and regression outputs.",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path("configs") / "quality_profiles.json"),
        help="Target JSON path for calibrated profiles.",
    )
    parser.add_argument(
        "--report-md",
        default=str(Path("configs") / "calibration_report.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate suggestions without overwriting the JSON profile file.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = QualityMetricCalibrator().calibrate(
        args.input_dir,
        output_json=args.output_json,
        report_path=args.report_md,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(Path(args.output_json))
    print(Path(args.report_md))
    return 0 if report.summary["warning_count"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
