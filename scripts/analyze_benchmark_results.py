from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.benchmark_result_analyzer import BenchmarkResultAnalyzer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze benchmark output directories without rerunning benchmarks.")
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Benchmark output directory. Pass multiple times to merge acceptance and regression outputs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary.json, worst_cases.json, and benchmark_analysis.md will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of entries to keep for each worst-case ranking.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    analyzer = BenchmarkResultAnalyzer()
    report = analyzer.analyze(args.input_dir, output_dir=args.output_dir, top_k=args.top_k)
    print(Path(args.output_dir) / "summary.json")
    print(Path(args.output_dir) / "worst_cases.json")
    print(Path(args.output_dir) / "benchmark_analysis.md")
    return 0 if report.summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
