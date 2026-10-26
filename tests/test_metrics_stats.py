import math

import pytest

from genint import metrics as M
from genint import stats

FRESH = ("The harbour at dawn smells of diesel and wet rope. Fishermen argue about weather while gulls circle the "
         "ice plant, and a radio somewhere insists the tide tables are wrong again.")


def T(*pairs):
    return [{"speaker": s, "text": t} for s, t in pairs]


def test_basic_counts_and_lexicons():
    t = T(("a", "That's a fascinating point! Perhaps we might think about it? I wonder what you think.\n\n- one\n- two"))
    m = M.analyze_turns(t)[0]
    assert m["question_marks"] == 1 and m["bullets"] == 2 and m["paragraphs"] == 2
    assert m["praise_per100"] > 0 and m["hedge_per100"] > 0 and m["opens_with_validation"] == 1.0
    assert m["first_person_per100"] > 0 and m["second_person_per100"] > 0 and m["we_per100"] > 0
    assert M.analyze_turns(T(("a", "Plain statement of fact about tides.")))[0]["opens_with_validation"] == 0.0


def test_novelty_drops_when_text_repeats():
    t = T(("a", FRESH), ("b", FRESH), ("a", "Quite different: sourdough starters need regular feeding, warmth and patience."))
    m = M.analyze_turns(t)
    assert m[0]["novelty_trigram"] == 1.0 and m[1]["novelty_trigram"] == 0.0 and m[2]["novelty_trigram"] > 0.9
    assert m[1]["self_repeat_trigram"] == 0.0 and m[1]["echo_trigram"] == 1.0     # copied from partner, not from itself
    assert m[1]["compress_novelty"] < m[0]["compress_novelty"] * 0.5 and m[2]["compress_novelty"] > m[1]["compress_novelty"]
    assert m[1]["sim_prev"] == pytest.approx(1.0)


def test_self_repeat_distinguishes_own_reuse():
    t = T(("a", FRESH), ("b", "Something unrelated about orchestras, brass sections and rehearsal schedules."), ("a", FRESH))
    m = M.analyze_turns(t)
    assert m[2]["self_repeat_trigram"] == 1.0 and m[2]["echo_trigram"] == 0.0


def test_farewell_detection_uses_the_tail_only():
    assert M.farewell("blah " * 100 + "Goodbye, take care!")
    assert not M.farewell("Goodbye is a word we should discuss. " + "More discussion follows. " * 30)


def test_seed_similarity_uses_topic_or_first_turn():
    on = M.analyze_turns(T(("a", "habit formation loops cues routines rewards"), ("b", "Cues and rewards drive each habit loop.")), "habit cues rewards")
    off = M.analyze_turns(T(("a", "Orchids prefer indirect light."), ("b", "Quantum chromodynamics describes gluons.")), "habit cues rewards")
    assert on[1]["sim_seed"] > 0.3 and (off[1]["sim_seed"] or 0) == 0


def test_conversation_metrics_shapes():
    texts = [FRESH + f" Variation {i} adds mackerel number {i * 7}." for i in range(12)]
    turns = [{"speaker": "ab"[i % 2], "text": t} for i, t in enumerate(texts)]
    pt = M.analyze_turns(turns)
    c = M.conversation_metrics(turns, pt)
    assert c["a"]["turns"] == c["b"]["turns"] == 6
    assert c["conversation"]["word_share_a"] == pytest.approx(0.5, abs=0.02)
    assert "lexical_convergence_index" in c["conversation"] and c["conversation"]["farewell_onset"] is None
    assert c["a"]["words_median"] > 0 and c["a"]["trend_words"] is not None


def test_stagnation_onset_found():
    xs = [0.5, 0.4, 0.05, 0.04, 0.03, 0.4]
    assert M._first_run_below(xs, 0.12, 3) == 2 and M._first_run_below(xs, 0.12, 4) is None


def test_usage_metrics_reasoning_overhead():
    ev = [{"speaker": "a", "text": "one two three four five", "scripted": False, "latency_s": 2.0, "cost": 0.01,
           "truncated": False, "usage": {"input_tokens": 100, "output_tokens": 50, "cache_read": 0, "cache_write": 0, "thinking_tokens": 40}},
          {"speaker": "b", "text": "one two", "scripted": False, "latency_s": 1.0, "cost": None, "truncated": False,
           "usage": {"input_tokens": 10, "output_tokens": 4, "cache_read": 0, "cache_write": 0, "thinking_tokens": 0}}]
    u = M.usage_metrics(ev)
    assert u["a"]["thinking_share"] == 0.8 and u["a"]["output_tokens_per_visible_word"] == 10.0
    assert u["b"]["cost_usd"] is None and u["a"]["cost_usd"] == 0.01


def test_aggregate_runs_paired_diff_and_ci():
    sides = [{"a": {"words": 100.0 + i}, "b": {"words": 50.0 + i}} for i in range(4)]
    agg = M.aggregate_runs(sides, [{"turns": 8.0}] * 4)
    d = agg["diff"]["words"]
    assert d["mean_diff"] == pytest.approx(50.0) and d["excludes_zero"] is True
    assert agg["side"]["words"]["a"]["mean"] == pytest.approx(101.5) and agg["side"]["words"]["a"]["low"] < 101.5 < agg["side"]["words"]["a"]["high"]


def test_stats_helpers():
    assert stats.slope([1, 2, 3, 4, 5]) == pytest.approx(1.0) and stats.slope([1, 2]) is None
    assert stats.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0) and stats.pearson([1, 1, 1], [1, 2, 3]) is None
    m, lo, hi = stats.ci95([1, 2, 3])
    assert m == 2 and lo == pytest.approx(2 - 4.303 / math.sqrt(3)) and stats.ci95([1]) is None
    assert stats.mean([None, 2.0, float("nan"), 4.0]) == 3.0 and stats.paired_diff([1], [1]) is None
