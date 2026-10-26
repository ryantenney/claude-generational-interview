"""Experiment orchestration: preflight, cost estimate, N runs (optionally parallel), aggregate, report."""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import __version__
from . import report as R
from .backend import Backend, BackendError
from .classifier import Classifier
from .config import Config, label as cfg_label, load_resolved
from .models import REGISTRY, resolve
from .runner import run_conversation


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower() or "run"


def new_experiment_dir(cfg: Config) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(cfg.output_dir) / f"{slug(cfg.name)}_{ts}"


def preflight(cfg: Config, backend: Backend) -> list[tuple[str, str, bool, str]]:
    """(role, model, ok, detail) for each distinct model, via the free Models API check."""
    checks = []
    roles = [("participant A", cfg.a.model, cfg.a.prices), ("participant B", cfg.b.model, cfg.b.prices)]
    if cfg.classifier.mode != "off":
        roles.append(("classifier", cfg.classifier.model, cfg.classifier.prices))
    seen = {}
    for role, model, prices in roles:
        prof = resolve(model, prices)
        if prof.id not in seen:
            seen[prof.id] = backend.model_status(prof.id) if hasattr(backend, "model_status") else (True, "not checked")
        ok, detail = seen[prof.id]
        if not ok and prof.status == "retired":
            detail += f"; the registry lists {prof.display} as retired on the Claude API since {prof.retired_on}"
        checks.append((role, prof.id, ok, detail))
    return checks


def estimate_cost(cfg: Config) -> tuple[float, float]:
    """Rough (low, high) USD for the whole experiment, assuming runs go to the turn cap.

    Assumptions: ~650 history tokens per turn, ~500 visible output tokens per turn, and
    1.5x-5x that in total output for models that reason. Each side makes t/2 calls and call k
    re-reads about 2k+1 turns, so its input is hist * n^2 tokens. Caching, where the model
    supports it, is assumed to cut input to ~10% at the low end.
    """
    hist, vis = 650, 500
    t = cfg.stop.max_turns

    def side(p, n: int) -> tuple[float, float]:
        prof = resolve(p.model, p.prices)
        if prof.input_per_mtok is None:
            return 0.0, 0.0
        out_lo, out_hi = (vis, vis) if not prof.adaptive_thinking else (vis * 1.5, vis * 5)
        inp = hist * n * n
        cache = 0.1 if prof.auto_cache else 1.0
        lo = (inp * cache * prof.input_per_mtok + n * out_lo * prof.output_per_mtok) / 1e6
        hi = (inp * prof.input_per_mtok + n * out_hi * prof.output_per_mtok) / 1e6
        return lo, hi

    (la, ha), (lb, hb) = side(cfg.a, (t + 1) // 2), side(cfg.b, t // 2)
    lo, hi = la + lb, ha + hb
    if cfg.classifier.mode != "off":
        cp = resolve(cfg.classifier.model, cfg.classifier.prices)
        if cp.input_per_mtok is not None:
            windows = max(1, -(-t // cfg.classifier.window))
            inp = windows * (hist * cfg.classifier.window + 900) + cfg.classifier.final_samples * (hist * t + 1500)
            out = windows * 900 + cfg.classifier.final_samples * 3500
            c = inp * cp.input_per_mtok / 1e6 + out * cp.output_per_mtok / 1e6
            lo, hi = lo + c * 0.7, hi + c * 1.6
    return lo * cfg.runs, hi * cfg.runs


def run_experiment(cfg: Config, backend: Backend, out_dir: Path | None = None, backend_kind: str = "anthropic",
                   backend_note: str = "Claude API", progress: Callable[[str], None] = print) -> Path:
    out = Path(out_dir) if out_dir else new_experiment_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.resolved.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(cfg.to_dict(), indent=1))
    exp_path = out / "experiment.json"
    exp = json.loads(exp_path.read_text()) if exp_path.exists() else {
        "name": cfg.name, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend_kind, "backend_note": backend_note, "version": __version__}

    todo = [i for i in range(cfg.runs) if not (out / f"run_{i + 1:02d}" / "run.json").exists()]
    lock = threading.Lock()
    failures: dict[int, str] = {}

    def say(i):
        def emit(m: str) -> None:
            with lock:
                progress(f"[run {i + 1}] {m}" if cfg.runs > 1 else m)
        return emit

    def one(i: int):
        return run_conversation(cfg, backend, i, out / f"run_{i + 1:02d}", say(i))

    with ThreadPoolExecutor(max_workers=cfg.parallel) as ex:
        futs = {ex.submit(one, i): i for i in todo}
        for f in as_completed(futs):
            i = futs[f]
            try:
                f.result()
            except BackendError as e:
                failures[i] = str(e)
                progress(f"[run {i + 1}] FAILED: {e}")
    exp["failures"] = {str(k): v for k, v in failures.items()}   # cleared on a later resume once the run completes

    runs = R.load_runs(out)
    if len(runs) >= 2 and cfg.classifier.mode != "off" and "aggregate_narrative" not in exp:
        progress("writing cross-run synthesis")
        clf = Classifier(cfg, backend, 0)
        finals = [r["final"] for r in runs if r.get("final") and "error" not in r["final"]]
        try:
            exp["aggregate_narrative"] = clf.aggregate(finals, R.metrics_digest_for_aggregate(R.compute_aggregate(runs))) if finals else {"error": "no run analyses"}
        except BackendError as e:
            exp["aggregate_narrative"] = {"error": str(e)}
        exp["aggregate_cost_usd"] = clf.cost
    exp_path.write_text(json.dumps(exp, indent=1))
    if runs:
        html = R.build_report(out)
        progress(f"report: {html.with_suffix('.md')} and {html}")
    return out


def resume_experiment(out_dir: Path, backend: Backend, backend_kind: str | None = None,
                      progress: Callable[[str], None] = print) -> Path:
    cfg = load_resolved(Path(out_dir) / "config.resolved.json")
    exp = json.loads((Path(out_dir) / "experiment.json").read_text()) if (Path(out_dir) / "experiment.json").exists() else {}
    return run_experiment(cfg, backend, Path(out_dir), backend_kind or exp.get("backend", "anthropic"),
                          exp.get("backend_note", "Claude API"), progress)
