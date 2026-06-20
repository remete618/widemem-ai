"""widemem command-line interface.

Currently exposes the benchmark reporting surface:

    widemem bench report <result.json>     # reproducible report from a run
    widemem bench locomo                    # how to produce a run

The `report` command is the credibility unlock: it turns a raw result file into
a verifiable benchmark report with a reproducibility hash.
"""
from __future__ import annotations

import argparse
import json
import sys

from widemem.bench import report_from_file


def _cmd_bench_report(args) -> int:
    judge_prompt = None
    if args.judge_prompt_file:
        with open(args.judge_prompt_file) as f:
            judge_prompt = f.read()
    md, machine = report_from_file(args.result, judge_prompt=judge_prompt)
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(machine, f, indent=2)
        print(f"\n[machine report written to {args.out}]", file=sys.stderr)
    return 0


def _cmd_bench_locomo(args) -> int:
    if args.compare:
        print(
            f"--compare {args.compare}: a shared evaluator that runs widemem AND "
            f"{args.compare} under one judge is not implemented yet (needs a "
            f"{args.compare} adapter). Tracked in the hardening backlog.",
            file=sys.stderr,
        )
        return 2
    print(
        "Run the LoCoMo harness to produce a result file, then `widemem bench "
        "report <file>`:\n"
        "  python benchmark/val.py ingest --store-dir <dir>\n"
        "  python benchmark/val.py eval --store-dir <dir> --out result.json\n"
        "  widemem bench report result.json"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="widemem", description="widemem CLI")
    sub = p.add_subparsers(dest="group", required=True)
    bench = sub.add_parser("bench", help="benchmarking")
    bsub = bench.add_subparsers(dest="bench_cmd", required=True)

    rep = bsub.add_parser("report", help="reproducible report from a result file")
    rep.add_argument("result", help="path to a LoCoMo result JSON")
    rep.add_argument("--judge-prompt-file", default=None, help="embed the judge prompt used")
    rep.add_argument("--out", default=None, help="write the machine-readable report JSON here")
    rep.set_defaults(func=_cmd_bench_report)

    loc = bsub.add_parser("locomo", help="how to run the LoCoMo benchmark")
    loc.add_argument("--compare", default=None, help="(planned) compare against another system, e.g. mem0")
    loc.set_defaults(func=_cmd_bench_locomo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
