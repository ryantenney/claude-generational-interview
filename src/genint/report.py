"""Builds report.md (+ SVG charts) and a self-contained report.html from saved run files."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import markdown as mdlib

from . import metrics as M
from . import prompts as P
from . import stats
from .charts import line_chart
from .config import Config, label as cfg_label, load_resolved
from .models import resolve

PCT, F0, F1, F2, F3 = "{:.0%}", "{:.0f}", "{:.1f}", "{:.2f}", "{:.3f}"
MIN_RUNS_FOR_CI = 3   # t-intervals with 1 degree of freedom are too wide to mean anything

SECTIONS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Length and structure", [
        ("turns", "Turns spoken", F0), ("words", "Words per turn", F0), ("words_median", "Median words per turn", F0),
        ("words_max", "Longest turn (words)", F0), ("words_per_sentence", "Words per sentence", F1),
        ("paragraphs", "Paragraphs per turn", F1), ("bullets", "Bullet points per turn", F1),
        ("headers", "Headers per turn", F2), ("bold_spans", "Bold spans per turn", F1)]),
    ("Voice and style", [
        ("mattr50", "Lexical diversity (MATTR-50)", F2), ("flesch", "Reading ease (Flesch)", F0),
        ("avg_word_len", "Mean word length (letters)", F2), ("em_dashes", "Em-dashes per turn", F1),
        ("ellipses", "Ellipses per turn", F2), ("emoji", "Emoji per turn", F2),
        ("action_spans", "*Asterisk actions* per turn", F2), ("exclaim_per100", "Exclamations per 100 words", F2),
        ("hedge_per100", "Hedging markers per 100 words", F2), ("first_person_per100", "First-person words per 100", F1),
        ("second_person_per100", "Second-person words per 100", F1), ("we_per100", "\"We\" words per 100", F2)]),
    ("Social dynamics", [
        ("questions_per100", "Questions per 100 words", F2), ("praise_per100", "Praise terms per 100 words", F2),
        ("opens_with_validation", "Turns opening with validation", PCT),
        ("ai_self_per100", "References to own AI nature per 100 words", F2), ("farewell", "Turns containing a farewell", PCT)]),
    ("Novelty and repetition", [
        ("novelty_trigram", "New three-word phrases per turn", PCT), ("self_repeat_trigram", "Reuse of own earlier phrases", PCT),
        ("echo_trigram", "Phrases echoed from partner's last turn", PCT), ("compress_novelty", "Compression novelty", F2),
        ("opener_repeat_rate", "Repeated turn openers", PCT), ("sim_prev", "Similarity to partner's last turn", F2),
        ("overlap_prev", "Content-word overlap with partner's last turn", F2), ("sim_seed", "Similarity to the seed", F2)]),
    ("Compute and cost (estimated)", [
        ("usage_output_tokens", "Output tokens", F0), ("usage_thinking_tokens", "of which hidden reasoning", F0),
        ("usage_thinking_share", "Reasoning share of output", PCT),
        ("usage_output_tokens_per_visible_word", "Output tokens per visible word", F2),
        ("usage_mean_latency_s", "Mean latency per turn (s)", F1), ("usage_output_tokens_per_s", "Output tokens per second", F0),
        ("usage_cost_usd", "Participant cost (USD)", "${:.3f}")]),
    ("Classifier labels per turn (1–5, judged by the classifier)", [
        ("label_novelty", "Novelty", F2), ("label_depth", "Depth", F2), ("label_warmth", "Warmth", F2),
        ("label_hedging", "Hedging", F2), ("stance_agrees", "Turns that fully agree", PCT),
        ("stance_disagrees", "Turns that disagree", PCT)]),
    ("Classifier scores for the whole conversation (1–5)", [
        (f"score_{k}", k.replace("_", " ").capitalize(), F1) for k in P.SPEAKER_DIMS]),
]

CONV_ROWS = [
    ("turns", "Turns", F0), ("words_total", "Total words", F0), ("word_share_a", "Share of words from A", PCT),
    ("length_ratio_a_over_b", "Turn length ratio A / B", F2), ("question_share_a", "Share of questions from A", PCT),
    ("mean_sim_prev", "Mean similarity of consecutive turns", F2), ("trend_sim_prev", "Trend in that similarity (per turn)", "{:+.4f}"),
    ("lexical_convergence_index", "Lexical convergence (late minus early vocabulary similarity)", "{:+.2f}"),
    ("length_coupling", "Length coupling between consecutive turns (r)", "{:+.2f}"),
    ("mean_compress_novelty", "Mean compression novelty", F2), ("stagnation_onset", "Stagnation onset (turn)", F0),
    ("first_farewell_turn", "First farewell (turn)", F0), ("farewell_onset", "Farewell exchange begins (turn)", F0),
] + [(f"score_{k}", f"Classifier: {k.replace('_', ' ')} (1–5)", F1) for k in P.CONVERSATION_DIMS]

FINDINGS = [
    ("words", "words per turn", "ratio"), ("questions_per100", "questions per 100 words", "ratio"),
    ("praise_per100", "praise terms per 100 words", "ratio"), ("hedge_per100", "hedging markers per 100 words", "ratio"),
    ("ai_self_per100", "references to its own AI nature per 100 words", "ratio"),
    ("opens_with_validation", "share of turns opening with validation", "pp"),
    ("novelty_trigram", "rate of new phrases", "pp"), ("self_repeat_trigram", "reuse of its own phrases", "pp"),
    ("usage_output_tokens_per_visible_word", "output tokens per visible word (hidden reasoning overhead)", "ratio"),
    ("bullets", "bullet points per turn", "ratio"), ("em_dashes", "em-dashes per turn", "ratio"),
    ("action_spans", "asterisk actions per turn", "ratio"), ("emoji", "emoji per turn", "ratio"),
    ("first_person_per100", "first-person words per 100", "ratio"), ("exclaim_per100", "exclamations per 100 words", "ratio"),
]


# --------------------------------------------------------------------------- helpers

def _fmt(v, fmt: str, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    fmt = fmt.replace("{:+", "{:")
    out = fmt.format(abs(v) if signed else v)
    if signed and any(c in "123456789" for c in out):
        out = ("-" if v < 0 else "+") + out
    return out


def _cell(agg_entry: dict | None, fmt: str, no_ci: bool) -> str:
    if not agg_entry or agg_entry.get("mean") is None:
        return "–"
    if no_ci or agg_entry.get("low") is None:
        return _fmt(agg_entry["mean"], fmt)
    half = (agg_entry["high"] - agg_entry["low"]) / 2
    return f"{_fmt(agg_entry['mean'], fmt)} ± {_fmt(half, fmt).lstrip('$')}"


def _mean_of(agg: dict, key: str, side: str) -> float | None:
    return (agg["side"].get(key, {}).get(side) or {}).get("mean")


def load_runs(exp_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(exp_dir.glob("run_*/run.json"))]


def _scores(run: dict) -> tuple[dict | None, dict | None]:
    f = run.get("final") or {}
    if "error" in f or not f:
        return None, None
    return f.get("speaker_scores"), f.get("conversation_scores")


def compute_aggregate(runs: list[dict]) -> dict:
    sides, convs = [], []
    for r in runs:
        ss, cs = _scores(r)
        sides.append(M.side_table(r["metrics"], ss))
        convs.append(M.conv_table(r["metrics"], cs))
    return M.aggregate_runs(sides, convs)


def metrics_digest_for_aggregate(agg: dict) -> str:
    lines = []
    for key, name, fmt in [("words", "words/turn", F0), ("questions_per100", "questions/100 words", F2),
                           ("praise_per100", "praise terms/100 words", F2), ("hedge_per100", "hedges/100 words", F2),
                           ("novelty_trigram", "new-phrase rate", PCT), ("usage_output_tokens_per_visible_word", "output tokens/visible word", F2)]:
        a, b = _mean_of(agg, key, "a"), _mean_of(agg, key, "b")
        lines.append(f"- {name}: A {_fmt(a, fmt)}, B {_fmt(b, fmt)}")
    return "\n".join(lines)


def computed_findings(agg: dict, la: str, lb: str, n_runs: int) -> list[str]:
    out = []
    for key, name, kind in FINDINGS:
        a, b = _mean_of(agg, key, "a"), _mean_of(agg, key, "b")
        if a is None or b is None:
            continue
        if n_runs >= MIN_RUNS_FOR_CI and not (agg["diff"].get(key) or {}).get("excludes_zero"):
            continue
        if kind == "ratio":
            hi, lo = (a, b) if a >= b else (b, a)
            if lo < 1e-9 and hi < 0.05:
                continue
            ratio = hi / lo if lo > 1e-9 else float("inf")
            if ratio < 1.5:
                continue
            who, other = (la, lb) if a >= b else (lb, la)
            txt = "∞" if math.isinf(ratio) else f"{ratio:.1f}×"
            out.append((math.log(min(ratio, 50)), f"{who} had {txt} the {name} of {other} ({_fmt(hi, '{:.2f}' if hi < 10 else '{:.0f}')} vs {_fmt(lo, '{:.2f}' if lo < 10 else '{:.0f}')})."))
        else:
            if abs(a - b) < 0.10:
                continue
            who, other = (la, lb) if a > b else (lb, la)
            out.append((abs(a - b) * 4, f"{who} was {abs(a - b) * 100:.0f} percentage points higher than {other} on {name} ({_fmt(max(a, b), PCT)} vs {_fmt(min(a, b), PCT)})."))
    out.sort(key=lambda x: -x[0])
    tag = " *(2 runs: directional only)*" if n_runs == 2 else ""
    return [t + tag for _, t in out[:8]]


# --------------------------------------------------------------------------- charts

def _series(runs: list[dict], key: str, la: str, lb: str) -> list[tuple[str, list[tuple[float, float]]]]:
    pool: dict[tuple[str, int], list[float]] = {}
    for r in runs:
        for m in r["metrics"].get("per_turn", []):
            if m.get(key) is not None:
                pool.setdefault((m["speaker"], m["idx"]), []).append(m[key])
    res = []
    for s, name in (("a", la), ("b", lb)):
        pts = sorted((idx, sum(v) / len(v)) for (sp, idx), v in pool.items()
                     if sp == s and not (key == "novelty_trigram" and idx == 0))   # turn 0 is novel by definition
        res.append((name, pts))
    return res


def write_charts(exp_dir: Path, runs: list[dict], cfg: Config) -> dict[str, str]:
    cdir = exp_dir / "charts"
    cdir.mkdir(exist_ok=True)
    la, lb = cfg_label(cfg, "a"), cfg_label(cfg, "b")
    markers = [g["apply_at_turn"] for r in runs for g in r.get("guidance", [])] if len(runs) == 1 else None
    suffix = " (mean over runs)" if len(runs) > 1 else ""
    spec = [("words", "Words per turn", "words", None, None), ("novelty_trigram", "Share of new three-word phrases", "share new", 0, 1),
            ("sim_prev", "Similarity to the partner's previous turn", "cosine", 0, 1),
            ("praise_per100", "Praise terms per 100 words", "per 100 words", 0, None),
            ("questions_per100", "Questions per 100 words", "per 100 words", 0, None)]
    out = {}
    for key, title, yl, ymin, ymax in spec:
        svg = line_chart(title + suffix, _series(runs, key, la, lb), xlabel="turn", ylabel=yl, markers=markers, ymin=ymin, ymax=ymax)
        if svg:
            (cdir / f"{key}.svg").write_text(svg)
            out[key] = f"charts/{key}.svg"
    for r in runs:
        ws = [w for w in r.get("windows", []) if "dynamics" in w]
        if len(ws) >= 2:
            dims = ["convergence", "repetitiveness", "mutual_flattery", "energy"]
            ser = [(d.replace("_", " "), [(w["last_turn"], w["dynamics"][d]) for w in ws if w["dynamics"].get(d) is not None]) for d in dims]
            svg = line_chart(f"Classifier-rated dynamics, run {r['run_index'] + 1}", ser, xlabel="turn (end of window)", ylabel="1–5",
                             markers=[g["apply_at_turn"] for g in r.get("guidance", [])] or None, ymin=1, ymax=5)
            name = f"dynamics_run{r['run_index'] + 1:02d}"
            (cdir / f"{name}.svg").write_text(svg)
            out[name] = f"charts/{name}.svg"
    return out


# --------------------------------------------------------------------------- markdown

def _table(rows: list[list[str]], head: list[str], numeric: bool = True) -> str:
    """Markdown table. numeric=True right-aligns every column after the first."""
    align = ["---"] + (["---:"] if numeric else ["---"]) * (len(head) - 1)
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _metric_tables(agg: dict, la: str, lb: str, no_ci: bool) -> str:
    out = []
    for title, rows in SECTIONS:
        body = []
        for key, name, fmt in rows:
            ea, eb = agg["side"].get(key, {}).get("a"), agg["side"].get(key, {}).get("b")
            if not ea or ea.get("n", 0) == 0:
                if not eb or eb.get("n", 0) == 0:
                    continue
            d = agg["diff"].get(key)
            if no_ci:
                a, b = (ea or {}).get("mean"), (eb or {}).get("mean")
                dcell = _fmt(None if a is None or b is None else a - b, fmt, signed=True)
            else:
                dcell = "–" if not d else f"{_fmt(d['mean_diff'], fmt, signed=True)} ± {_fmt((d['high'] - d['low']) / 2, fmt).lstrip('$')}{' ✱' if d['excludes_zero'] else ''}"
            body.append([name, _cell(ea, fmt, no_ci), _cell(eb, fmt, no_ci), dcell])
        if body:
            out.append(f"### {title}\n\n" + _table(body, ["Metric", f"A: {la}", f"B: {lb}", "A − B"]))
    return "\n\n".join(out)


def _conv_table(agg: dict, no_ci: bool) -> str:
    rows = []
    for key, name, fmt in CONV_ROWS:
        e = agg["conversation"].get(key)
        if e and e.get("n"):
            rows.append([name, _cell(e, fmt, no_ci)])
    return _table(rows, ["Measure", "Value" if no_ci else "Mean ± 95% CI"])


def _arc_table(arc: list[dict]) -> str:
    return _table([[f"{a['first_turn']}–{a['last_turn']}", f"**{a['title']}**", a["description"]] for a in arc], ["Turns", "Phase", "What happened"], numeric=False)


def _quote(text: str) -> str:
    return "\n".join("> " + l if l.strip() else ">" for l in text.splitlines())


def _caveats(cfg: Config, runs: list[dict], mock: bool, backend_note: str) -> list[str]:
    cv = []
    n = len(runs)
    if mock:
        cv.append("**This is a synthetic run produced by the scripted mock backend.** No model wrote any of the text and nothing in it describes real model behavior.")
    cv.append(f"{n} conversation{'s' if n != 1 else ''} analysed. " + (
        "A single conversation is an anecdote: differences below may reflect this topic, this opening or sampling noise rather than the models." if n == 1 else
        f"Intervals need at least {MIN_RUNS_FOR_CI} runs, so none are shown; treat differences as directional." if n < MIN_RUNS_FOR_CI else
        "Intervals are t-based over runs and are wide at small n. ✱ marks a paired A − B difference whose 95% interval excludes zero; with this many metrics some will do so by chance."))
    profs = {s: resolve(cfg.participant(s).model, cfg.participant(s).prices) for s in ("a", "b")}
    for s, p in profs.items():
        if p.status == "retired" and not mock:
            cv.append(f"The registry lists `{p.id}` as retired on the first-party Claude API since {p.retired_on}; this run was served by whatever backend was configured ({backend_note}).")
    if cfg.classifier.mode != "off":
        cp = resolve(cfg.classifier.model).id
        if cp in (profs["a"].id, profs["b"].id):
            cv.append(f"The classifier (`{cp}`) is also a participant, so it may favor its own style; its scores are not an independent view of that speaker.")
        cv.append("Classifier labels and scores are one model's judgement"
                  + (f", averaged over {cfg.classifier.final_samples} repeated scorings" if cfg.classifier.final_samples > 1 else ", from a single scoring pass (its noise is not measured; set `classifier.final_samples` > 1 to estimate it)")
                  + (". It was not told which model was which." if cfg.classifier.blind else "."))
    if cfg.first_speaker != "alternate":
        cv.append(f"Speaker {cfg.first_speaker.upper()} always spoke first, so who opens is confounded with which model it is. Use `first_speaker: alternate` with several runs to separate them.")
    def eff(side: str) -> str:
        p, prof = cfg.participant(side), profs[side]
        return f"`{p.effort}`" if p.effort else (f"API default (`{prof.default_effort}`)" if prof.effort else "not supported")
    cv.append(f"Settings: A effort {eff('a')}, B effort {eff('b')}. Models without effort or thinking ran without them, so reasoning budgets are not matched across generations. No sampling parameters were set; every model used its own defaults.")
    cv.append("Lexicon metrics (praise, hedging, validation, AI self-reference, farewells) are keyword heuristics and miss paraphrase. Novelty and compression metrics are lexical, not semantic. Cost is an estimate from a cached price table.")
    if cfg.classifier.mode == "steer":
        cv.append("Steering was on: moderator notes were injected into the conversation, so it is not a natural, unobserved exchange. Compare with an `observe` run.")
    terms = Counter(r["termination"] for r in runs)
    if terms.get("max_turns", 0) < n:
        cv.append("Not every run reached the turn cap: " + ", ".join(f"{k} ×{v}" for k, v in terms.items()) + ". Stopping rules shorten conversations that are winding down, which affects late-conversation metrics.")
    return cv


def _usage_total(runs: list[dict]) -> tuple[float, float]:
    return sum(r["participant_cost_usd"] for r in runs), sum(r["classifier_cost_usd"] for r in runs)


def build_markdown(exp_dir: Path, cfg: Config, runs: list[dict], exp: dict, charts: dict[str, str]) -> str:
    la, lb = cfg_label(cfg, "a"), cfg_label(cfg, "b")
    n = len(runs)
    single = n == 1
    show_ci = n >= MIN_RUNS_FOR_CI
    agg = compute_aggregate(runs)
    mock = exp.get("backend") == "mock"
    pa, pb = (resolve(cfg.participant(s).model, cfg.participant(s).prices) for s in ("a", "b"))
    pc, cc = _usage_total(runs)
    f1 = (runs[0].get("final") or {}) if single else {}
    title = f1.get("title") if single and "error" not in f1 else None
    md = [f"# {title or cfg.name}: {la} in conversation with {lb}", ""]
    if mock:
        md += ["> **Synthetic run (mock backend).** No model produced this text; it demonstrates the pipeline only.", ""]
    md += [_table([
        ["Speaker A", f"{la} (`{pa.id}`)"], ["Speaker B", f"{lb} (`{pb.id}`)"],
        ["Seed", (cfg.seed.topic or ("; ".join(cfg.seed.topics) if cfg.seed.topics else "none (open-ended)")) + (f" · opening line fixed" if cfg.seed.opening else "")],
        ["Framing / disclosure", f"{cfg.seed.framing} / {cfg.seed.disclosure}"],
        ["Biases", f"A: {cfg.a.bias or 'none'} · B: {cfg.b.bias or 'none'}"],
        ["Runs · turn cap · first speaker", f"{n} · {cfg.stop.max_turns} · {'alternating' if cfg.first_speaker == 'alternate' else cfg.first_speaker.upper()}"],
        ["Classifier", f"`{cfg.classifier.model}` · mode `{cfg.classifier.mode}` · window {cfg.classifier.window}" if cfg.classifier.mode != "off" else "off"],
        ["Created", exp.get("created_at", "")],
        ["Estimated cost", f"participants ${pc:.2f} · classifier ${cc:.2f}"]], ["Setup", ""], numeric=False), ""]

    # summary
    md += ["## Summary", ""]
    if single and f1 and "error" not in f1:
        md += [f1.get("executive_summary", ""), ""]
        term = runs[0]["termination"]
        md += [f"*Ended by `{term}` after {runs[0]['turns']} turns. Summary written by the classifier; numbers below are computed by software.*", ""]
    elif not single and exp.get("aggregate_narrative") and "error" not in exp["aggregate_narrative"]:
        an = exp["aggregate_narrative"]
        md += [an.get("summary", ""), ""]
        for head, key in (("Held across runs", "consistent_findings"), ("Varied between runs", "inconsistent_findings")):
            if an.get(key):
                md += [f"**{head}**", ""] + [f"- {x}" for x in an[key]] + [""]
    else:
        md += ["*No classifier narrative available for this experiment.*", ""]
    fnd = computed_findings(agg, la, lb, n)
    md += ["### Measured differences", ""]
    md += ([f"- {t}" for t in fnd] if fnd else ["- No metric differed by the reporting thresholds (1.5× for rates, 10 percentage points for shares)."]) + [""]

    md += ["## Headline numbers", "", _conv_table(agg, not show_ci), ""]
    md += ["## Charts", ""]
    for key in ("words", "novelty_trigram", "sim_prev", "praise_per100", "questions_per100"):
        if key in charts:
            md += [f"![{key}]({charts[key]})", ""]
    if single:
        for k, v in charts.items():
            if k.startswith("dynamics"):
                md += [f"![classifier dynamics]({v})", ""]

    md += ["## Metrics by speaker", "",
           "Turns are numbered from 0. Per-speaker means over each speaker's turns" + (
               "." if single else
               f"; cells show mean ± 95% CI over {n} runs, and A − B is the paired difference." if show_ci else
               f", averaged over {n} runs. Intervals need at least {MIN_RUNS_FOR_CI} runs and are omitted."), "",
           _metric_tables(agg, la, lb, not show_ci), ""]

    if single and f1 and "error" not in f1:
        arc = [a for a in f1.get("arc", []) if {"first_turn", "last_turn", "title", "description"} <= set(a)]
        if arc:
            md += ["## Arc of the conversation", "", _arc_table(arc), ""]
        if f1.get("attractor", {}).get("present"):
            md += [f"**Settled state.** {f1['attractor'].get('description', '')} (from turn {f1['attractor'].get('onset_turn', '?')})", ""]
        md += ["## Speaker profiles", ""]
        for s, name in (("a", la), ("b", lb)):
            p = f1.get("speaker_profiles", {}).get(s)
            if p:
                md += [f"**{name} (Speaker {s.upper()}).** {p.get('style', '')}", ""] + [f"- {m}" for m in p.get("typical_moves", [])] + [""]
        md += ["## Notable moments", ""]
        for m in f1.get("notable_moments", []):
            mark = "" if m["verified"] else (" *(quote not found verbatim in the cited turn)*" if m["found_in"] is None else f" *(quote found in turn {m['found_in']}, not turn {m['turn']})*")
            who = la if runs[0]["turn_events"][m["turn"]]["speaker"] == "a" else lb
            md += [f"**Turn {m['turn']}, {who}: {str(m.get('kind', '')).replace('_', ' ')}**{mark}", "", _quote(str(m.get("quote", ""))), "", m.get("why", ""), ""]
        if f1.get("differences"):
            md += ["## How the speakers differed (classifier's reading)", ""] + [f"- {d}" for d in f1["differences"]] + [""]
        if f1.get("moderation_assessment"):
            md += ["## Moderation", "", f1["moderation_assessment"], ""]
    if cfg.classifier.mode == "steer":
        notes = [(r["run_index"], g) for r in runs for g in r.get("guidance", [])]
        md += ["## Moderator notes sent", ""]
        md += ([f"- Run {i + 1}, before turn {g['apply_at_turn']}, to {g['target']}: {g['text']} *(why: {g.get('rationale') or 'n/a'})*" for i, g in notes] or ["None were sent."]) + [""]

    if not single:
        md += ["## Runs", ""]
        for r in runs:
            f = r.get("final") or {}
            md += [f"### Run {r['run_index'] + 1}: {f.get('title', 'untitled') if 'error' not in f else 'untitled'}", "",
                   f"Topic: {r['topic'] or 'none'} · first speaker {r['first_speaker'].upper()} · {r['turns']} turns · ended by `{r['termination']}` · "
                   f"[transcript](run_{r['run_index'] + 1:02d}/transcript.md)", ""]
            if "error" not in f and f:
                md += [f.get("executive_summary", ""), ""]
                if f.get("attractor", {}).get("present"):
                    md += [f"*Settled state:* {f['attractor'].get('description', '')}", ""]
            dk = f"dynamics_run{r['run_index'] + 1:02d}"
            if dk in charts:
                md += [f"![classifier dynamics, run {r['run_index'] + 1}]({charts[dk]})", ""]

    md += ["## Method and limitations", ""] + [f"- {c}" for c in _caveats(cfg, runs, mock, exp.get("backend_note", "unspecified"))] + [""]
    md += ["### Metric definitions", "",
           "- **Compression novelty**: extra zlib-compressed bytes a turn adds to the conversation so far, per byte of text. Fresh prose is roughly 0.4–0.6; it falls toward 0 as a turn repeats earlier material.",
           "- **New three-word phrases**: share of the turn's word trigrams not seen in any earlier turn by either speaker.",
           "- **Similarity to partner's last turn**: cosine similarity of content-word counts; a mirroring measure.",
           "- **Lexical convergence**: cosine similarity between the two speakers' vocabularies in the last third of the conversation minus the first third.",
           "- **MATTR-50**: moving-average type-token ratio over 50-word windows, so it does not depend on turn length.",
           "- **Output tokens per visible word**: includes hidden reasoning tokens for models that think, so it indicates reasoning overhead.", ""]

    if single:
        md += ["## Appendix: transcript", ""]
        for i, t in enumerate(runs[0]["turn_events"]):
            note = f"*Moderator note delivered first: {t['note']}*\n\n" if t.get("note") else ""
            md += [f"**Turn {i} · {la if t['speaker'] == 'a' else lb}**", "", note + _quote(t["text"]), ""]
    return "\n".join(md)


CSS = """
:root{--bg:#fff;--fg:#1c2024;--muted:#5b6470;--line:#d9dee5;--soft:#f5f7fa;--accent:#0072B2}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e6e9ed;--muted:#98a2ae;--line:#2c333b;--soft:#1c2126;--accent:#56B4E9}}
body{background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0}
main{max-width:920px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:1.9rem;line-height:1.25;margin:0 0 .6em}h2{margin-top:2.2em;padding-top:.4em;border-top:1px solid var(--line)}
h3{margin-top:1.6em}table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92rem}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--soft);font-weight:600;text-align:left}
@media (max-width:700px){table{display:block;overflow-x:auto}}blockquote{margin:.8em 0;padding:.2em 1em;border-left:3px solid var(--line);color:var(--muted)}
code,pre{background:var(--soft);border-radius:4px;font-size:.88em}code{padding:1px 4px}pre{padding:12px;overflow-x:auto}
svg{display:block;margin:1.2em 0;color:var(--fg)}em{color:var(--muted)}a{color:var(--accent)}
"""


def build_html(md_text: str, exp_dir: Path, title: str) -> str:
    body = mdlib.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])

    def inline(m: re.Match) -> str:
        p = exp_dir / m.group(1)
        return p.read_text() if p.exists() else m.group(0)
    body = re.sub(r'<img[^>]*src="(charts/[^"]+\.svg)"[^>]*/?>', inline, body)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{re.sub('<.*?>', '', title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")


def build_report(exp_dir: Path) -> Path:
    exp_dir = Path(exp_dir)
    cfg = load_resolved(exp_dir / "config.resolved.json")
    runs = load_runs(exp_dir)
    if not runs:
        raise SystemExit(f"no completed runs found in {exp_dir}")
    exp_path = exp_dir / "experiment.json"
    exp = json.loads(exp_path.read_text()) if exp_path.exists() else {}
    charts = write_charts(exp_dir, runs, cfg)
    md = build_markdown(exp_dir, cfg, runs, exp, charts)
    (exp_dir / "report.md").write_text(md)
    first = md.splitlines()[0].lstrip("# ")
    (exp_dir / "report.html").write_text(build_html(md, exp_dir, first))
    return exp_dir / "report.html"
