from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.benchmark_runner import BenchmarkRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run acceptance benchmark suite and write artifacts.")
    parser.add_argument(
        "--manifest",
        default=str(Path("benchmarks") / "acceptance_manifest.json"),
        help="Path to the acceptance benchmark manifest.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-case artifacts and the suite report will be written.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional explicit path for the suite summary JSON. Defaults to <output-dir>/acceptance_report.json.",
    )
    parser.add_argument(
        "--execute-proposed-commands",
        action="store_true",
        help="Execute manifest proposed_commands after the acceptance flow.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    runner = BenchmarkRunner()
    report = runner.run_acceptance_manifest(
        args.manifest,
        output_dir=output_dir,
        execute_proposed_commands=args.execute_proposed_commands,
    )
    report_path = Path(args.report_json) if args.report_json else output_dir / "acceptance_report.json"
    report_path.write_text(report.to_json(), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
