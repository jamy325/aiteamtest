from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.benchmark_runner import BenchmarkRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-world regression samples against a baseline.")
    parser.add_argument(
        "--manifest",
        default=str(Path("benchmarks") / "real_samples" / "manifest.json"),
        help="Path to the real-world regression manifest.",
    )
    parser.add_argument(
        "--baseline",
        default=str(Path("benchmarks") / "baselines" / "real_world_regression_baseline.json"),
        help="Path to the baseline metrics JSON.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-case artifacts and regression report will be written.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional explicit path for the regression report JSON. Defaults to <output-dir>/real_world_regression_report.json.",
    )
    parser.add_argument(
        "--execute-proposed-commands",
        action="store_true",
        help="Execute manifest proposed_commands after the regression flow.",
    )
    parser.add_argument("--case-timeout-seconds", type=float, default=None, help="Optional timeout override for each case.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    runner = BenchmarkRunner()
    output_dir = Path(args.output_dir)
    report = runner.run_real_world_regression(
        args.manifest,
        baseline_path=args.baseline,
        output_dir=output_dir,
        execute_proposed_commands=args.execute_proposed_commands,
        timeout_seconds=args.case_timeout_seconds,
    )
    report_path = Path(args.report_json) if args.report_json else output_dir / "real_world_regression_report.json"
    report_path.write_text(report.to_json(), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
