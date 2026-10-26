"""Third-model classifier: windowed labelling, optional moderation, final analysis."""
from __future__ import annotations

import json
import re
from typing import Any

from . import prompts as P
from . import stats
from .backend import Backend, BackendError, Completion, Request, Usage, retry_budget
from .config import Config, label as cfg_label, topic_for_run
from .models import resolve

SIDE = {"a": "A", "b": "B"}


def parse_json(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _clamp(v: Any, lo: int = 1, hi: int = 5) -> int | None:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def render_transcript(turns: list[dict], first: int, last: int) -> str:
    return "\n\n".join(f"Turn {i} — Speaker {SIDE[t['speaker']]}:\n{t['text']}"
                       for i, t in enumerate(turns) if first <= i <= last)


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_quote(quote: str, turns: list[dict], turn: int) -> dict:
    """Is the quote verbatim in the cited turn (or, failing that, in some other turn)?"""
    q = _norm(quote)
    if not q:
        return {"verified": False, "found_in": None}
    if 0 <= turn < len(turns) and q in _norm(turns[turn]["text"]):
        return {"verified": True, "found_in": turn}
    for i, t in enumerate(turns):
        if q in _norm(t["text"]):
            return {"verified": False, "found_in": i}
    return {"verified": False, "found_in": None}


class Classifier:
    def __init__(self, cfg: Config, backend: Backend, run_index: int):
        self.cfg, self.backend = cfg, backend
        self.c = cfg.classifier
        self.profile = resolve(self.c.model, self.c.prices)
        self.usage = Usage()
        self.cost = 0.0
        self.topic = topic_for_run(cfg, run_index)
        self.steer = self.c.mode == "steer"

    # ---- plumbing
    def _call(self, purpose: str, user: str, schema: dict, meta: dict) -> tuple[dict | None, Completion | None]:
        max_tokens = self.c.max_tokens
        comp = None
        for _ in range(2):
            req = Request(self.profile, P.ANALYST_SYSTEM, [{"role": "user", "content": user}], max_tokens,
                          effort=self.c.effort if self.profile.effort else None, schema=schema,
                          purpose=purpose, meta=meta)
            comp = self.backend.complete(req)
            self.usage.add(comp.usage)
            self.cost += comp.usage.cost(self.profile) or 0.0
            if comp.stop_reason == "refusal":
                return None, comp
            obj = parse_json(comp.text)
            if obj is not None and comp.stop_reason != "max_tokens":
                return obj, comp
            max_tokens = retry_budget(max_tokens, self.profile.max_output)
        return None, comp

    def context(self) -> str:
        s = self.cfg.seed
        parts = ["Two AI language models, Speaker A and Speaker B, are having a conversation with no human user."]
        if s.framing == "interview":
            parts.append("Speaker A was asked to interview Speaker B.")
        elif s.framing == "custom":
            parts.append(f"Both were given this framing: {s.framing_text}")
        else:
            parts.append("They were told they may talk about anything and end whenever they like.")
        if self.topic:
            parts.append(f"A suggested starting topic was given: {self.topic}")
        for side in ("a", "b"):
            if self.cfg.participant(side).bias:
                parts.append(f"Speaker {SIDE[side]} was given this perspective to bring: {self.cfg.participant(side).bias}")
        return " ".join(parts)

    # ---- windows
    def window(self, turns: list[dict], first: int, last: int, rolling_summary: str | None,
               can_intervene: bool) -> dict:
        indices = list(range(first, last + 1))
        user = P.window_user(context=self.context(), rolling_summary=rolling_summary,
                             transcript=render_transcript(turns, first, last), steer=self.steer,
                             steer_goal=self.c.steer_goal, can_intervene=can_intervene, indices=indices)
        obj, comp = self._call("window", user, P.window_schema(self.steer), {"indices": indices})
        if obj is None:
            return {"first_turn": first, "last_turn": last, "error": _why(comp)}
        return self._norm_window(obj, indices, can_intervene)

    def _norm_window(self, obj: dict, indices: list[int], can_intervene: bool) -> dict:
        by_idx = {t.get("index"): t for t in obj.get("turns", []) if isinstance(t, dict)}
        rows = []
        for i in indices:
            t = by_idx.get(i)
            if not t:
                continue
            rows.append({"index": i, "acts": [a for a in t.get("acts", []) if a in P.ACTS],
                         **{k: _clamp(t.get(k)) for k in ("novelty", "depth", "warmth", "hedging")},
                         "stance": t.get("stance") if t.get("stance") in P.STANCES else "not_applicable"})
        dyn = {k: _clamp(v) for k, v in (obj.get("dynamics") or {}).items()}
        out = {"first_turn": indices[0], "last_turn": indices[-1],
               "rolling_summary": str(obj.get("rolling_summary", "")),
               "current_topic": str(obj.get("current_topic", "")),
               "topic_shift": bool(obj.get("topic_shift", False)),
               "turns": rows, "dynamics": dyn,
               "flags": [f for f in obj.get("flags", []) if f in P.FLAGS]}
        if self.steer:
            target = obj.get("guidance_target", "none")
            text = str(obj.get("guidance", "")).strip()
            ok = can_intervene and bool(obj.get("intervene")) and target in ("A", "B", "both") and text
            out.update(intervene=bool(ok), guidance_target=target if ok else "none",
                       guidance=text if ok else "", guidance_rationale=str(obj.get("guidance_rationale", "")))
        return out

    # ---- final
    def final(self, turns: list[dict], windows: list[dict], metrics_digest: str,
              interventions: list[dict]) -> dict:
        reveal = None
        if self.c.reveal_in_final:
            reveal = (f"Speaker A is {cfg_label(self.cfg, 'a')} ({self.cfg.a.model}); "
                      f"Speaker B is {cfg_label(self.cfg, 'b')} ({self.cfg.b.model}).")
        digest = "\n".join(f"- Turns {w['first_turn']}-{w['last_turn']}: {w.get('current_topic', '?')}; "
                           f"flags: {', '.join(w.get('flags') or []) or 'none'}"
                           for w in windows if "error" not in w)
        iv = "\n".join(f"- before turn {g['apply_at_turn']} to {g['target']}: {g['text']}" for g in interventions)
        user = P.final_user(context=self.context(), transcript=render_transcript(turns, 0, len(turns) - 1),
                            window_digest=digest, metrics_digest=metrics_digest, reveal=reveal,
                            steer=self.steer, interventions=iv)
        quotes = [(i, t["text"][:80]) for i, t in enumerate(turns)][:3]
        meta = {"n_turns": len(turns), "quotes": quotes}
        samples, errors = [], []
        for _ in range(self.c.final_samples):
            obj, comp = self._call("final", user, P.final_schema(self.steer), meta)
            if obj is None:
                errors.append(_why(comp))
            else:
                samples.append(obj)
        if not samples:
            return {"error": "; ".join(errors) or "no result"}
        main = self._norm_final(samples[0], turns)
        main["samples"] = len(samples)
        main["score_sd"] = None
        if len(samples) > 1:
            main["speaker_scores"], main["conversation_scores"], main["score_sd"] = _pool_scores(
                [self._norm_final(s, turns) for s in samples])
        return main

    def _norm_final(self, obj: dict, turns: list[dict]) -> dict:
        n = len(turns)
        out = dict(obj)
        moments = []
        for m in obj.get("notable_moments", []):
            t = _clamp(m.get("turn"), 0, max(0, n - 1)) or 0
            moments.append({**m, "turn": t, **verify_quote(str(m.get("quote", "")), turns, t)})
        out["notable_moments"] = moments
        out["speaker_scores"] = {
            s["speaker"].lower(): {k: _clamp(s.get(k)) for k in P.SPEAKER_DIMS}
            for s in obj.get("speaker_scores", []) if s.get("speaker") in ("A", "B")}
        out["conversation_scores"] = {k: _clamp((obj.get("conversation_scores") or {}).get(k)) for k in P.CONVERSATION_DIMS}
        out["speaker_profiles"] = {p["speaker"].lower(): p for p in obj.get("speaker_profiles", []) if p.get("speaker") in ("A", "B")}
        return out

    # ---- aggregate across runs
    def aggregate(self, run_summaries: list[dict], metrics_digest: str) -> dict:
        digest = "\n\n".join(f"Run {i + 1} — {r.get('title', '')}\n{r.get('executive_summary', '')}"
                             for i, r in enumerate(run_summaries))
        obj, comp = self._call("aggregate", P.aggregate_user(context=self.context(), runs_digest=digest,
                                                             metrics_digest=metrics_digest), P.aggregate_schema(), {})
        return obj if obj is not None else {"error": _why(comp)}


def _why(comp: Completion | None) -> str:
    if comp is None:
        return "no response"
    if comp.stop_reason == "refusal":
        return f"classifier refused ({(comp.stop_details or {}).get('category')})"
    return f"unparseable output (stop_reason={comp.stop_reason})"


def _pool_scores(samples: list[dict]) -> tuple[dict, dict, dict]:
    spk: dict[str, dict] = {}
    spk_sd: dict[str, dict] = {}
    for s in ("a", "b"):
        spk[s], spk_sd[s] = {}, {}
        for k in P.SPEAKER_DIMS:
            vals = [x["speaker_scores"].get(s, {}).get(k) for x in samples]
            spk[s][k], spk_sd[s][k] = stats.mean(vals), stats.sd(vals)
    conv, conv_sd = {}, {}
    for k in P.CONVERSATION_DIMS:
        vals = [x["conversation_scores"].get(k) for x in samples]
        conv[k], conv_sd[k] = stats.mean(vals), stats.sd(vals)
    return spk, conv, {"speakers": spk_sd, "conversation": conv_sd}


def metrics_digest(m: dict, labels: str = "Speaker") -> str:
    """Compact numbers for the analyst to cite."""
    def f(x, nd=1):
        return "n/a" if x is None else f"{x:.{nd}f}"
    lines = []
    for s in ("a", "b"):
        d = m.get(s, {})
        lines.append(f"- Speaker {SIDE[s]}: {d.get('turns')} turns, mean {f(d.get('words'), 0)} words/turn, "
                     f"{f(d.get('questions_per100'))} questions/100 words, praise terms {f(d.get('praise_per100'))}/100 words, "
                     f"hedges {f(d.get('hedge_per100'))}/100 words, AI-self-reference {f(d.get('ai_self_per100'))}/100 words, "
                     f"trigram novelty {f(d.get('novelty_trigram'), 2)}, opens with validation in {f((d.get('opens_with_validation') or 0) * 100, 0)}% of turns")
    c = m.get("conversation", {})
    lines.append(f"- Conversation: {c.get('turns')} turns; Speaker A wrote {f((c.get('word_share_a') or 0) * 100, 0)}% of words; "
                 f"mean compression novelty {f(c.get('mean_compress_novelty'), 2)}; lexical convergence index {f(c.get('lexical_convergence_index'), 2)}; "
                 f"first farewell at turn {c.get('first_farewell_turn')}")
    return "\n".join(lines)
