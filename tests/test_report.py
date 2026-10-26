import json

from genint import report as R
from genint.experiment import estimate_cost, preflight, run_experiment
from genint.mock import MockBackend
from conftest import make_cfg


def build(tmp_path, **over):
    cfg = make_cfg(**{"stop": {"max_turns": 10}, "classifier": {"mode": "steer", "window": 4}, **over})
    out = run_experiment(cfg, MockBackend(1), tmp_path / "exp", "mock", "mock", progress=lambda m: None)
    return cfg, out


def test_single_run_report_has_all_sections(tmp_path):
    _, out = build(tmp_path, runs=1)
    md = (out / "report.md").read_text()
    for h in ("## Summary", "### Measured differences", "## Headline numbers", "## Charts", "## Metrics by speaker",
              "## Arc of the conversation", "## Speaker profiles", "## Notable moments", "## Method and limitations",
              "## Appendix: transcript"):
        assert h in md, h
    assert "Synthetic run (mock backend)" in md and "A single conversation is an anecdote" in md
    html = (out / "report.html").read_text()
    assert "<svg" in html and 'src="charts/' not in html           # charts are inlined: the file is self-contained
    assert "prefers-color-scheme:dark" in html
    for f in ("config.resolved.json", "experiment.json", "run_01/events.jsonl", "run_01/run.json", "run_01/transcript.md"):
        assert (out / f).exists(), f


def test_multi_run_report_uses_intervals_from_three_runs(tmp_path):
    _, out = build(tmp_path, runs=3, first_speaker="alternate")
    md = (out / "report.md").read_text()
    assert "## Runs" in md and "### Run 3:" in md and "Mean ± 95% CI" in md and " ± " in md
    assert "Appendix: transcript" not in md                         # multi-run: transcripts live in run_NN/
    exp = json.loads((out / "experiment.json").read_text())
    assert "aggregate_narrative" in exp


def test_two_runs_omit_intervals(tmp_path):
    _, out = build(tmp_path, runs=2)
    md = (out / "report.md").read_text()
    assert "Intervals need at least 3 runs" in md and "directional" in md
    table = md.split("### Length and structure")[1].split("###")[0]
    assert "±" not in table


def test_report_rebuilds_offline_from_saved_files(tmp_path):
    _, out = build(tmp_path, runs=1)
    first = (out / "report.md").read_text()
    (out / "report.md").unlink(); (out / "report.html").unlink()
    R.build_report(out)
    assert (out / "report.md").read_text() == first


def test_resume_experiment_only_runs_missing_runs(tmp_path):
    from genint.experiment import resume_experiment
    cfg, out = build(tmp_path, runs=2)
    (out / "run_02" / "run.json").unlink()
    be = MockBackend(1)
    n = {"calls": 0}
    orig = be.complete
    be.complete = lambda req: (n.__setitem__("calls", n["calls"] + 1), orig(req))[1]
    resume_experiment(out, be, "mock", progress=lambda m: None)
    assert (out / "run_02" / "run.json").exists() and n["calls"] > 0
    assert n["calls"] < 30        # run 1 was not regenerated; run 2's turns already existed, so only analysis re-ran


def test_formatting_helpers():
    assert R._fmt(-0.024, "${:.3f}", signed=True) == "-$0.024"
    assert R._fmt(0.0004, "{:.2f}", signed=True) == "0.00"
    assert R._fmt(0.12, "{:+.0%}", signed=True) == "+12%" and R._fmt(None, "{:.1f}") == "–"


def test_preflight_flags_retired_model_with_registry_note():
    class Gone:
        def model_status(self, mid):
            return (False, "not found (retired or misspelled)") if "3-opus" in mid else (True, "ok")
    checks = preflight(make_cfg(), Gone())
    bad = [c for c in checks if not c[2]]
    assert len(bad) == 1 and bad[0][1] == "claude-3-opus-20240229" and "retired on the Claude API since 2026-01-05" in bad[0][3]


def test_estimate_cost_scales_with_runs_and_turns():
    one = estimate_cost(make_cfg(runs=1)); two = estimate_cost(make_cfg(runs=2))
    longer = estimate_cost(make_cfg(runs=1, stop={"max_turns": 24}))
    assert two[0] == one[0] * 2 and longer[1] > one[1] > 0 and one[0] < one[1]


def test_cli_reports_failed_runs_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    from genint import cli
    from genint.backend import BackendError

    class Down:
        def complete(self, req):
            raise BackendError("authentication failed")
    monkeypatch.setattr(cli, "_backend", lambda args: (Down(), "anthropic", "test"))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: x\nparticipants: {a: {model: claude-opus-5-5}, b: {model: claude-opus-4-5}}\nstop: {max_turns: 4}\n")
    code = cli.main(["run", str(cfg), "--no-check", "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 1 and "1 run(s) failed" in err and "genint resume" in err and "authentication failed" in err
    assert not (tmp_path / "out" / "report.md").exists()
