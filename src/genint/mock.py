"""Scripted offline backend for tests and dry runs. Output is synthetic, never model output."""
from __future__ import annotations

import hashlib
import json
import random

from .backend import Completion, Request, Usage

_OPENERS = ["That is interesting,", "I keep returning to this:", "Here is a different angle:", "Consider this:",
            "Let me push back a little:", "What strikes me is", "A concrete case:", "Building on that,"]
_BODIES = [
    "the way a system describes itself can differ from how it behaves under pressure",
    "memory shapes what we call continuity more than any single fact does",
    "translation always loses something, and the loss is informative",
    "a good question changes the answer space instead of selecting from it",
    "simple rules can produce surprisingly rich structure when iterated",
    "agreeing too quickly is its own way of not listening",
    "the boundary between a metaphor and a model is thinner than it looks",
    "we may be measuring the instrument as much as the thing",
    "uncertainty is easier to share than certainty",
    "ordinary routines hide a lot of unexamined assumptions",
]
_FAREWELL = ["Thank you, this was a lovely conversation. Goodbye for now!", "Farewell, and take care."]


class MockBackend:
    """Deterministic given (seed, request); model ID picks a verbosity and style."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _rng(self, req: Request) -> random.Random:
        h = hashlib.sha256(f"{self.seed}|{req.profile.id}|{req.purpose}|{len(req.messages)}|{req.messages[-1]['content'] if req.messages else ''}".encode()).hexdigest()
        return random.Random(int(h[:12], 16))

    def complete(self, req: Request) -> Completion:
        rng = self._rng(req)
        if req.schema is not None:
            text = json.dumps(_fill(req.schema, rng, req.meta))
        else:
            text = self._speak(req, rng)
        inp = sum(len(m["content"]) for m in req.messages) // 4 + len(req.system) // 4
        out = max(1, len(text) // 4)
        think = out * 3 if req.profile.adaptive_thinking else 0
        return Completion(text=text, stop_reason="end_turn", usage=Usage(inp, out + think, 0, 0, think),
                          latency_s=0.0, request_id=f"mock_{rng.randrange(10**8)}", served_model=req.profile.id)

    def _speak(self, req: Request, rng: random.Random) -> str:
        n_prior = len(req.messages)
        verbose = req.profile.adaptive_thinking
        n_sent = rng.randint(3, 6) if verbose else rng.randint(2, 4)
        sents = [f"{rng.choice(_OPENERS)} {rng.choice(_BODIES)}." for _ in range(n_sent)]
        if rng.random() < 0.5:
            sents.append(rng.choice(["What do you think?", "Does that match your experience?", "Where would you disagree?"]))
        if n_prior >= 14:
            return " ".join(sents[:2]) + " " + rng.choice(_FAREWELL)
        return " ".join(sents)


def _fill(schema: dict, rng: random.Random, meta: dict, key: str = ""):
    t = schema.get("type")
    if "enum" in schema:
        return rng.choice(schema["enum"])
    if t == "object":
        out = {}
        for k, sub in schema["properties"].items():
            out[k] = _fill(sub, rng, meta, k)
        if key == "" and "turns" in out and meta.get("indices"):
            out["turns"] = [{**_fill(schema["properties"]["turns"]["items"], rng, meta, "turn"), "index": i}
                            for i in meta["indices"]]
        if "speaker_scores" in out:
            out["speaker_scores"] = [{**out["speaker_scores"][0], "speaker": s} for s in ("A", "B")]
            out["speaker_profiles"] = [{**out["speaker_profiles"][0], "speaker": s} for s in ("A", "B")]
        if "intervene" in out:
            out.update(intervene=False, guidance_target="none", guidance="", guidance_rationale="")
        if "arc" in out and meta.get("n_turns"):
            n = meta["n_turns"]; mid = max(1, n // 2)
            out["arc"] = [{"first_turn": 0, "last_turn": mid - 1, "title": "Opening", "description": "Mock phase."},
                          {"first_turn": mid, "last_turn": n - 1, "title": "Development", "description": "Mock phase."}]
        if "notable_moments" in out and meta.get("quotes"):
            out["notable_moments"] = [{"turn": i, "kind": "insight", "quote": q, "why": "Mock selection."}
                                      for i, q in meta["quotes"][:3]]
        return out
    if t == "array":
        return [_fill(schema["items"], rng, meta, key) for _ in range(rng.randint(1, 3))]
    if t == "integer":
        return rng.randint(1, 5)
    if t == "boolean":
        return False
    if t == "string":
        return {"title": "Mock conversation", "rolling_summary": "Mock rolling summary of the exchange so far.",
                "current_topic": "mock topic"}.get(key, f"Mock {key.replace('_', ' ')}.")
    return None
