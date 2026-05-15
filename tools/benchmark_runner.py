from __future__ import annotations

import argparse
from pathlib import Path
import sys

from services.benchmark_runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark manifest against the vector reconstruction pipeline.")
    parser.add_argument("--manifest", required=True, help="Path to the benchmark manifest JSON file.")
    parser.add_argument("--output", help="Optional path for the JSON benchmark report.")
    parser.add_argument(
        "--execute-proposed-commands",
        action="store_true",
        help="Execute proposed commands after the minimal pipeline run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = BenchmarkRunner()
    report = runner.run_manifest(
        Path(args.manifest),
        execute_proposed_commands=bool(args.execute_proposed_commands),
    )
    payload = report.to_json()
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
        if not payload.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
