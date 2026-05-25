from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

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
        default=None,
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
    parser.add_argument("--case-timeout-seconds", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-case-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-json", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    runner = BenchmarkRunner()
    if args.worker_case_json is not None:
        case_payload = json.loads(Path(args.worker_case_json).read_text(encoding="utf-8"))
        case = runner.benchmark_case_from_dict(case_payload)
        result = runner._run_acceptance_case_inline(
            case,
            output_dir=Path(args.worker_output_dir),
            execute_proposed_commands=args.execute_proposed_commands,
        )
        Path(args.worker_result_json).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        print(args.worker_result_json)
        return 0

    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless worker mode is used")

    output_dir = Path(args.output_dir)
    report = runner.run_acceptance_manifest(
        args.manifest,
        output_dir=output_dir,
        execute_proposed_commands=args.execute_proposed_commands,
        timeout_seconds=args.case_timeout_seconds,
    )
    report_path = Path(args.report_json) if args.report_json else output_dir / "acceptance_report.json"
    report_path.write_text(report.to_json(), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
