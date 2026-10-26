"""Command line: genint run | check | resume | report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config as C
from . import experiment as E
from .backend import AnthropicBackend, BackendError, RequestConfigError
from .mock import MockBackend

CONFIRM_ABOVE_USD = 10.0


def _backend(args):
    if args.mock:
        return MockBackend(seed=args.mock_seed), "mock", "scripted mock backend"
    return AnthropicBackend(), "anthropic", "Claude API (api.anthropic.com)"


def _load(args) -> C.Config:
    try:
        cfg = C.load(args.config, args.set or [])
    except (C.ConfigError, FileNotFoundError) as e:
        sys.exit(f"config error: {e}")
    if args.runs:
        cfg.runs = args.runs
    if args.out_root:
        cfg.output_dir = args.out_root
    return cfg


def _print_checks(checks) -> bool:
    ok_all = True
    for role, model, ok, detail in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {role:<14} {model}: {detail}")
        ok_all &= ok
    return ok_all


def cmd_check(args) -> int:
    cfg = _load(args)
    backend, kind, _ = _backend(args)
    print(f"Config '{cfg.name}': {C.label(cfg, 'a')} vs {C.label(cfg, 'b')}, classifier {cfg.classifier.model} ({cfg.classifier.mode})")
    ok = _print_checks(E.preflight(cfg, backend))
    lo, hi = E.estimate_cost(cfg)
    print(f"Rough cost if every run reaches {cfg.stop.max_turns} turns x {cfg.runs} run(s): ${lo:.2f} to ${hi:.2f} (see estimate_cost for assumptions)")
    return 0 if ok else 1


def cmd_run(args) -> int:
    cfg = _load(args)
    backend, kind, note = _backend(args)
    if not args.no_check:
        print("Preflight:")
        if not _print_checks(E.preflight(cfg, backend)):
            print("\nOne or more models are unavailable. Edit the config (or pass --set participants.b.model=...) and retry; "
                  "use --no-check to skip this check.")
            return 1
    lo, hi = E.estimate_cost(cfg)
    if kind != "mock":
        print(f"Rough cost estimate: ${lo:.2f} to ${hi:.2f} (cap per run with stop.max_cost_usd)")
        if hi > CONFIRM_ABOVE_USD and not args.yes:
            print(f"Estimate exceeds ${CONFIRM_ABOVE_USD:.0f}; re-run with --yes to proceed.")
            return 2
    try:
        out = E.run_experiment(cfg, backend, Path(args.out) if args.out else None, kind, note)
    except (BackendError, RequestConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"done: {out}")
    return 0


def cmd_resume(args) -> int:
    backend, kind, _ = _backend(args)
    out = E.resume_experiment(Path(args.dir), backend, kind if args.mock else None)
    print(f"done: {out}")
    return 0


def cmd_report(args) -> int:
    from .report import build_report
    print(build_report(Path(args.dir)))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="genint", description="Staged conversations between two Claude generations.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, config=True):
        if config:
            p.add_argument("config", help="path to a YAML config")
            p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value, e.g. stop.max_turns=10")
            p.add_argument("--runs", type=int, help="override the number of runs")
            p.add_argument("--out-root", help="override output_dir")
        p.add_argument("--mock", action="store_true", help="use the scripted offline backend (synthetic text, no API calls)")
        p.add_argument("--mock-seed", type=int, default=0)

    p = sub.add_parser("check", help="verify models exist and estimate cost, without running")
    common(p); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("run", help="run an experiment and write the report")
    common(p)
    p.add_argument("--out", help="output directory (default: <output_dir>/<name>_<timestamp>)")
    p.add_argument("--yes", action="store_true", help="proceed even if the cost estimate is high")
    p.add_argument("--no-check", action="store_true", help="skip the model availability preflight")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("resume", help="finish an interrupted experiment directory")
    p.add_argument("dir"); common(p, config=False); p.set_defaults(fn=cmd_resume)
    p = sub.add_parser("report", help="rebuild report.md/report.html from saved runs (offline)")
    p.add_argument("dir"); p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
