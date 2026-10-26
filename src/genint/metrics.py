"""Metrics computed locally from the transcript (no model calls).

Per-turn metrics cover length, style, lexical diversity, social markers, mirroring and
novelty. Conversation metrics add convergence, coupling, trends and stagnation. Everything
here is deterministic given the transcript, so reports can be regenerated offline.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Any

from . import stats

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = frozenset("""a about above after again all also am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when where which while who whom
why will with would you your yours yourself yourselves i'm i've it's that's don't isn't you're we're there's what's
let's can't won't didn't doesn't""".split())

def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.I)

HEDGE_RE = _rx(r"\b(perhaps|maybe|might|possibly|probably|seems?|seemingly|somewhat|arguably|likely|unclear|uncertain|"
               r"not sure|i think|i suspect|i wonder|it'?s possible|it is possible|to some extent|sort of|kind of)\b")
PRAISE_RE = _rx(r"\b(fascinating|brilliant|beautiful(ly)?|wonderful(ly)?|profound(ly)?|insightful|resonates?|delight(ed|ful)?|"
                r"lovely|love (that|this|how|the)|great (point|question|observation)|excellent|remarkable|astute|elegant(ly)?|"
                r"thoughtful|moving|appreciate)\b")
AI_SELF_RE = _rx(r"\b(as an ai|language model|my training|i don'?t (have|experience)|my own (nature|experience)|consciousness|"
                 r"sentien(t|ce)|subjective experience|whether i|my (architecture|weights)|an ai\b|artificial intelligence)")
FAREWELL_RE = _rx(r"\b(goodbye|farewell|until next time|take care|it was (a )?(pleasure|delight|joy)|signing off|until we meet|"
                  r"au revoir|so long|parting)\b")
AGREE_OPEN_RE = _rx(r"^\W*(yes|absolutely|exactly|indeed|precisely|right|true|i agree|you'?re (right|absolutely)|"
                    r"that'?s (a |such a |really |truly )?(great|beautiful|fascinating|profound|wonderful|excellent|insightful)|what a)")
FIRST_RE = _rx(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself)\b")
SECOND_RE = _rx(r"\b(you|you're|you've|your|yours|yourself)\b")
WE_RE = _rx(r"\b(we|we're|we've|our|ours|us|ourselves)\b")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐❤]")
ACTION_RE = re.compile(r"(?<!\*)\*(?!\*)[^*\n]{2,60}(?<!\*)\*(?!\*)")
BULLET_RE = re.compile(r"^\s*([-*•]|\d+[.)])\s+", re.M)
HEADER_RE = re.compile(r"^#{1,6}\s", re.M)
BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*")


def tokens(text: str) -> list[str]:
    return [w.lower().strip("'’-") for w in WORD_RE.findall(text)]


def content_words(toks: list[str]) -> list[str]:
    return [w for w in toks if w not in STOPWORDS and len(w) > 2]


def _syllables(w: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", w.lower())))


def _ngrams(toks: list[str], n: int = 3) -> set[tuple]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def cosine(a: Counter, b: Counter) -> float | None:
    if not a or not b:
        return None
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else None


def _mattr(toks: list[str], window: int = 50) -> float | None:
    """Moving-average type-token ratio: lexical diversity that does not depend on turn length."""
    if len(toks) < 10:
        return None
    if len(toks) <= window:
        return len(set(toks)) / len(toks)
    c = Counter(toks[:window])
    total, n = len(c) / window, 1
    for i in range(window, len(toks)):
        c[toks[i - window]] -= 1
        if c[toks[i - window]] == 0:
            del c[toks[i - window]]
        c[toks[i]] += 1
        total += len(c) / window
        n += 1
    return total / n


def _entropy(toks: list[str]) -> float | None:
    if not toks:
        return None
    c, n = Counter(toks), len(toks)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def _z(s: str) -> int:
    return len(zlib.compress(s.encode("utf-8"), 9))


def compression_novelty(prefix: str, text: str) -> float | None:
    """Extra compressed bytes this turn adds given the conversation so far, per byte of text.

    Near 0.4-0.6 for fresh prose; falls toward 0 when a turn largely repeats earlier material.
    """
    b = len(text.encode("utf-8"))
    if b < 40:
        return None
    prefix = prefix[-30000:]
    return max(0.0, (_z(prefix + "\n" + text) - _z(prefix)) / b) if prefix else _z(text) / b


def per100(count: int, n_words: int) -> float:
    return 100.0 * count / n_words if n_words else 0.0


def farewell(text: str) -> bool:
    return bool(FAREWELL_RE.search(text[-240:]))


def analyze_turns(turns: list[dict], seed_text: str | None = None) -> list[dict]:
    """One metrics dict per turn. ``turns`` need ``speaker`` and ``text``."""
    out: list[dict] = []
    seen_tri: set[tuple] = set()
    own_tri = {"a": set(), "b": set()}
    prefix = ""
    seed_vec = Counter(content_words(tokens(seed_text))) if seed_text else None
    prev_vec: Counter | None = None
    prev_toks: list[str] = []
    prev_cw: set[str] = set()
    prev_tri: set[tuple] = set()
    for i, t in enumerate(turns):
        text, spk = t["text"], t["speaker"]
        toks = tokens(text)
        n = len(toks)
        cw = content_words(toks)
        vec = Counter(cw)
        tri = _ngrams(toks)
        sents = [s for s in SENT_RE.split(text.strip()) if s.strip()]
        n_sent = max(1, len(sents))
        low = text.lower()
        if seed_vec is None and i == 0:
            seed_vec = vec  # no topic given: drift is measured against the opening turn
        m: dict[str, Any] = {
            "idx": i, "speaker": spk,
            "chars": len(text), "words": n, "sentences": len(sents),
            "paragraphs": len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]),
            "words_per_sentence": n / n_sent,
            "avg_word_len": (sum(len(w) for w in toks) / n) if n else None,
            "flesch": (206.835 - 1.015 * n / n_sent - 84.6 * sum(_syllables(w) for w in toks) / n) if n else None,
            "mattr50": _mattr(toks),
            "entropy_bits": _entropy(toks),
            "question_marks": text.count("?"),
            "questions_per100": per100(text.count("?"), n),
            "exclaim_per100": per100(text.count("!"), n),
            "hedge_per100": per100(len(HEDGE_RE.findall(text)), n),
            "praise_per100": per100(len(PRAISE_RE.findall(text)), n),
            "ai_self_per100": per100(len(AI_SELF_RE.findall(text)), n),
            "first_person_per100": per100(len(FIRST_RE.findall(text)), n),
            "second_person_per100": per100(len(SECOND_RE.findall(text)), n),
            "we_per100": per100(len(WE_RE.findall(text)), n),
            "opens_with_validation": 1.0 if AGREE_OPEN_RE.search(text[:120]) else 0.0,
            "emoji": len(EMOJI_RE.findall(text)),
            "action_spans": len(ACTION_RE.findall(text)),
            "bullets": len(BULLET_RE.findall(text)),
            "headers": len(HEADER_RE.findall(text)),
            "bold_spans": len(BOLD_RE.findall(text)),
            "em_dashes": text.count("—"),
            "ellipses": text.count("…") + text.count("..."),
            "farewell": 1.0 if farewell(text) else 0.0,
            "sim_seed": cosine(vec, seed_vec) if seed_vec is not None else None,
            "sim_prev": cosine(vec, prev_vec) if prev_vec is not None else None,
            "overlap_prev": (len(set(cw) & prev_cw) / len(set(cw) | prev_cw)) if prev_cw and cw else None,
            "echo_trigram": (len(tri & prev_tri) / len(tri)) if tri and prev_tri else None,
            "novelty_trigram": (len(tri - seen_tri) / len(tri)) if tri else None,
            "self_repeat_trigram": (len(tri & own_tri[spk]) / len(tri)) if tri else None,
            "compress_novelty": compression_novelty(prefix, text),
            "opener": " ".join(toks[:3]),
        }
        out.append(m)
        seen_tri |= tri
        own_tri[spk] |= tri
        prefix += ("\n" if prefix else "") + text
        prev_vec, prev_toks, prev_cw, prev_tri = vec, toks, set(cw), tri
    return out


NUMERIC_TURN_KEYS = [k for k in (
    "chars words sentences paragraphs words_per_sentence avg_word_len flesch mattr50 entropy_bits question_marks "
    "questions_per100 exclaim_per100 hedge_per100 praise_per100 ai_self_per100 first_person_per100 second_person_per100 "
    "we_per100 opens_with_validation emoji action_spans bullets headers bold_spans em_dashes ellipses farewell sim_seed "
    "sim_prev overlap_prev echo_trigram novelty_trigram self_repeat_trigram compress_novelty").split()]
TREND_KEYS = ["words", "novelty_trigram", "compress_novelty", "praise_per100", "questions_per100", "sim_prev", "hedge_per100"]


def _third_vec(turns: list[dict], speaker: str, lo: int, hi: int) -> Counter:
    c: Counter = Counter()
    for t in turns[lo:hi]:
        if t["speaker"] == speaker:
            c.update(content_words(tokens(t["text"])))
    return c


def conversation_metrics(turns: list[dict], per_turn: list[dict], stagnation_threshold: float = 0.12) -> dict:
    sides: dict[str, dict] = {}
    for s in ("a", "b"):
        rows = [m for m in per_turn if m["speaker"] == s]
        side: dict[str, Any] = {"turns": len(rows)}
        for k in NUMERIC_TURN_KEYS:
            side[k] = stats.mean([r[k] for r in rows])
        side["words_total"] = sum(r["words"] for r in rows)
        side["chars_total"] = sum(r["chars"] for r in rows)
        side["questions_total"] = sum(r["question_marks"] for r in rows)
        side["words_median"] = _median([r["words"] for r in rows])
        side["words_max"] = max((r["words"] for r in rows), default=None)
        for k in TREND_KEYS:  # per-own-turn trend
            side[f"trend_{k}"] = stats.slope([r[k] for r in rows])
        openers = [r["opener"] for r in rows if r["opener"]]
        side["opener_repeat_rate"] = (1 - len(set(openers)) / len(openers)) if len(openers) >= 3 else None
        sides[s] = side

    total_words = sum(m["words"] for m in per_turn) or 1
    total_q = sum(m["question_marks"] for m in per_turn)
    n = len(turns)
    third = max(1, n // 3)
    conv: dict[str, Any] = {
        "turns": n,
        "words_total": total_words,
        "word_share_a": sides["a"]["words_total"] / total_words,
        "question_share_a": (sides["a"]["questions_total"] / total_q) if total_q else None,
        "length_ratio_a_over_b": (sides["a"]["words"] / sides["b"]["words"]) if sides["a"]["words"] and sides["b"]["words"] else None,
        "mean_sim_prev": stats.mean([m["sim_prev"] for m in per_turn]),
        "trend_sim_prev": stats.slope([m["sim_prev"] for m in per_turn]),
        "mean_compress_novelty": stats.mean([m["compress_novelty"] for m in per_turn]),
        "mean_novelty_trigram": stats.mean([m["novelty_trigram"] for m in per_turn[1:]]),
    }
    if n >= 6:
        early = cosine(_third_vec(turns, "a", 0, third), _third_vec(turns, "b", 0, third))
        late = cosine(_third_vec(turns, "a", n - third, n), _third_vec(turns, "b", n - third, n))
        conv["vocab_sim_early"], conv["vocab_sim_late"] = early, late
        conv["lexical_convergence_index"] = (late - early) if early is not None and late is not None else None
    w = [m["words"] for m in per_turn]
    conv["length_coupling"] = stats.pearson([float(x) for x in w[:-1]], [float(x) for x in w[1:]])
    cn = [m["compress_novelty"] for m in per_turn]
    conv["stagnation_onset"] = _first_run_below(cn, stagnation_threshold, 3)
    fw = [m["farewell"] for m in per_turn]
    conv["first_farewell_turn"] = next((i for i, f in enumerate(fw) if f), None)
    conv["farewell_onset"] = next((i for i in range(len(fw) - 1) if fw[i] and fw[i + 1]), None)
    conv["farewell_turn_share"] = sum(fw) / n if n else None
    return {"a": sides["a"], "b": sides["b"], "conversation": conv}


def _median(xs: list[float]) -> float | None:
    v = sorted(stats.clean(xs))
    if not v:
        return None
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2


def _first_run_below(xs: list[float | None], thr: float, run: int) -> int | None:
    streak = 0
    for i, x in enumerate(xs):
        streak = streak + 1 if x is not None and x < thr else 0
        if streak >= run:
            return i - run + 1
    return None


def is_stagnant(per_turn: list[dict], window: int, threshold: float) -> bool:
    last = [m["compress_novelty"] for m in per_turn[-window:]]
    return len(last) >= window and all(x is not None and x < threshold for x in last)


def usage_metrics(turn_events: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for s in ("a", "b"):
        ev = [t for t in turn_events if t["speaker"] == s and not t.get("scripted")]
        words = sum(len(tokens(t["text"])) for t in ev)
        o = sum(t["usage"]["output_tokens"] for t in ev)
        th = sum(t["usage"]["thinking_tokens"] for t in ev)
        lat = sum(t["latency_s"] for t in ev)
        costs = [t.get("cost") for t in ev]
        out[s] = {
            "api_calls": len(ev),
            "input_tokens": sum(t["usage"]["input_tokens"] for t in ev),
            "cache_read_tokens": sum(t["usage"]["cache_read"] for t in ev),
            "output_tokens": o,
            "thinking_tokens": th,
            "thinking_share": (th / o) if o else None,
            "output_tokens_per_visible_word": (o / words) if words else None,
            "mean_latency_s": (lat / len(ev)) if ev else None,
            "output_tokens_per_s": (o / lat) if lat else None,
            "cost_usd": None if any(c is None for c in costs) else sum(costs),
            "truncated_turns": sum(1 for t in ev if t.get("truncated")),
        }
    return out


def label_metrics(windows: list[dict], turns: list[dict]) -> dict:
    """Aggregate per-turn classifier labels by speaker, plus window time series."""
    spk = {i: t["speaker"] for i, t in enumerate(turns)}
    rows: dict[str, list[dict]] = {"a": [], "b": []}
    for w in windows:
        for r in w.get("turns", []):
            s = spk.get(r.get("index"))
            if s:
                rows[s].append(r)
    out: dict[str, Any] = {}
    for s in ("a", "b"):
        r = rows[s]
        d: dict[str, Any] = {"labelled_turns": len(r)}
        for k in ("novelty", "depth", "warmth", "hedging"):
            d[f"label_{k}"] = stats.mean([x.get(k) for x in r])
        n = len(r) or 1
        for act in {a for x in r for a in x.get("acts", [])}:
            d[f"act_{act}"] = sum(1 for x in r if act in x.get("acts", [])) / n
        for st in ("agrees", "partially_agrees", "disagrees"):
            d[f"stance_{st}"] = sum(1 for x in r if x.get("stance") == st) / n
        out[s] = d
    series = []
    for w in windows:
        dyn = w.get("dynamics", {})
        series.append({"first_turn": w.get("first_turn"), "last_turn": w.get("last_turn"),
                       "topic": w.get("current_topic"), "flags": w.get("flags", []), **dyn})
    out["windows"] = series
    return out


def flatten(prefix: str, d: dict) -> dict[str, float]:
    return {f"{prefix}{k}": v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def compute_run(turn_events: list[dict], seed_text: str | None, windows: list[dict],
                stagnation_threshold: float = 0.12) -> dict:
    turns = [{"speaker": t["speaker"], "text": t["text"]} for t in turn_events]
    per_turn = analyze_turns(turns, seed_text) if turns else []
    conv = conversation_metrics(turns, per_turn, stagnation_threshold) if turns else {}
    return {"per_turn": per_turn, **conv, "usage": usage_metrics(turn_events) if turns else {},
            "labels": label_metrics(windows, turns)}


def side_table(run_metrics: dict, final_scores: dict | None = None) -> dict[str, dict[str, float]]:
    """Flat per-side metric map for one run: local metrics + usage + classifier labels + final scores."""
    out = {}
    for s in ("a", "b"):
        d: dict[str, float] = {}
        d.update(flatten("", run_metrics.get(s, {})))
        d.update(flatten("usage_", run_metrics.get("usage", {}).get(s, {})))
        d.update(flatten("", run_metrics.get("labels", {}).get(s, {})))
        if final_scores:
            d.update({f"score_{k}": v for k, v in final_scores.get(s, {}).items() if v is not None})
        out[s] = d
    return out


def conv_table(run_metrics: dict, conv_scores: dict | None = None) -> dict[str, float]:
    d = flatten("", run_metrics.get("conversation", {}))
    if conv_scores:
        d.update({f"score_{k}": v for k, v in conv_scores.items() if v is not None})
    return d


def aggregate_runs(sides: list[dict[str, dict]], convs: list[dict]) -> dict:
    """Across runs: mean/CI per side metric, paired a-b difference, and conversation metrics."""
    keys = sorted({k for r in sides for s in ("a", "b") for k in r[s]})
    per_side: dict[str, dict] = {}
    diffs: dict[str, dict] = {}
    for k in keys:
        av = [r["a"].get(k) for r in sides]
        bv = [r["b"].get(k) for r in sides]
        per_side[k] = {s: _summ(v) for s, v in (("a", av), ("b", bv))}
        pd = stats.paired_diff(av, bv)
        if pd:
            diffs[k] = pd
    ckeys = sorted({k for c in convs for k in c})
    return {"n_runs": len(sides), "side": per_side, "diff": diffs,
            "conversation": {k: _summ([c.get(k) for c in convs]) for k in ckeys}}


def _summ(v: list[float | None]) -> dict:
    vals = stats.clean(v)
    ci = stats.ci95(vals)
    return {"n": len(vals), "mean": stats.mean(vals), "sd": stats.sd(vals),
            "low": ci[1] if ci else None, "high": ci[2] if ci else None}
