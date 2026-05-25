from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.html_quality_report import HtmlQualityReportGenerator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate offline HTML quality reports from existing artifact bundles.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Case artifact directory or benchmark suite output directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output file path. Defaults to report.html for a case or index.html for a suite.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    generator = HtmlQualityReportGenerator()
    input_dir = Path(args.input_dir)
    if any((input_dir / name).exists() for name in ("acceptance_report.json", "real_world_regression_report.json")):
        result = generator.generate_suite_report(input_dir, output_path=args.output)
        print(result.index_path)
        return 0
    result = generator.generate_case_report(input_dir, output_path=args.output)
    print(result.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
