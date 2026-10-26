"""Runs one conversation, logging every event to an append-only JSONL file so a run can resume."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import metrics as M
from . import prompts as P
from .backend import Backend, BackendError, Request, Usage, retry_budget
from .classifier import Classifier, SIDE, metrics_digest, render_transcript
from .config import Config, first_speaker_for_run, label, topic_for_run
from .models import resolve


def other(side: str) -> str:
    return "b" if side == "a" else "a"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventLog:
    def __init__(self, path: Path):
        self.path = path

    def append(self, ev: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def build_messages(turns: list[dict], speaker: str, first_speaker: str, pending_note: str | None = None) -> list[dict]:
    """The conversation as ``speaker`` sees it: own turns are assistant, the partner's are user.

    Moderator notes stay attached to the partner message they arrived with, so the history is
    append-only and identical across calls.
    """
    msgs: list[dict] = []
    if speaker == first_speaker:
        msgs.append({"role": "user", "content": P.KICKOFF})
    for t in turns:
        if t["speaker"] == speaker:
            if t.get("note") and msgs and msgs[-1]["role"] == "user":
                msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + P.NOTE_TEMPLATE.format(text=t["note"])}
            msgs.append({"role": "assistant", "content": t["text"]})
        else:
            msgs.append({"role": "user", "content": t["text"]})
    if pending_note and msgs and msgs[-1]["role"] == "user":
        msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + P.NOTE_TEMPLATE.format(text=pending_note)}
    return msgs


def system_prompt(cfg: Config, side: str, topic: str | None) -> str:
    me, you = cfg.participant(side), cfg.participant(other(side))
    return P.participant_system(
        side=side, framing=cfg.seed.framing, framing_text=cfg.seed.framing_text, topic=topic,
        bias=me.bias, extra=me.system_extra, disclosure=cfg.seed.disclosure,
        self_name=resolve(me.model, me.prices).display, other_name=resolve(you.model, you.prices).display)


def run_conversation(cfg: Config, backend: Backend, run_index: int, run_dir: Path,
                     progress: Callable[[str], None] | None = None) -> dict:
    """Run (or resume) one conversation. Returns the run summary, also written to run.json."""
    say = progress or (lambda s: None)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(run_dir / "events.jsonl")
    events = [e for e in log.read()]
    # an api_error end marker means the previous attempt was interrupted: resume past it
    events = [e for e in events if not (e["type"] == "end" and e.get("termination") == "api_error")]
    turns = [e for e in events if e["type"] == "turn"]
    windows = [e for e in events if e["type"] == "window"]
    guidance = [e for e in events if e["type"] == "guidance"]
    ended = next((e for e in events if e["type"] == "end"), None)

    topic = topic_for_run(cfg, run_index)
    first = first_speaker_for_run(cfg, run_index)
    systems = {s: system_prompt(cfg, s, topic) for s in ("a", "b")}
    profiles = {s: resolve(cfg.participant(s).model, cfg.participant(s).prices) for s in ("a", "b")}
    if not any(e["type"] == "meta" for e in events):
        log.append({"type": "meta", "run_index": run_index, "topic": topic, "first_speaker": first,
                    "started_at": utcnow(), "systems": systems,
                    "models": {s: cfg.participant(s).model for s in ("a", "b")}})

    clf = Classifier(cfg, backend, run_index) if cfg.classifier.mode != "off" else None
    clf_enabled = clf is not None
    pending: dict[str, str] = {}
    for g in guidance:  # notes issued but not yet delivered (resume case)
        for tgt in g["targets"]:
            if not any(t["speaker"] == tgt and t["idx"] >= g["apply_at_turn"] for t in turns):
                pending[tgt] = g["text"]
    cost = sum(t.get("cost") or 0.0 for t in turns)
    termination = ended["termination"] if ended else None
    last_guidance_turn = max((g["issued_after_turn"] for g in guidance), default=-999)

    def window_flush(upto: int, final: bool = False) -> None:
        nonlocal clf_enabled, last_guidance_turn
        if not clf_enabled:
            return
        start = max((w["last_turn"] for w in windows), default=-1) + 1   # errored windows count as done
        if start > upto:
            return
        can = (clf.steer and not final and len(guidance) < cfg.classifier.max_interventions
               and upto - last_guidance_turn >= cfg.classifier.cooldown_turns)
        prev = next((w["rolling_summary"] for w in reversed(windows) if w.get("rolling_summary")), None)
        try:
            w = clf.window(turns, start, upto, prev, can)
        except BackendError as e:
            say(f"  classifier unavailable: {e}")
            clf_enabled = False
            w = {"first_turn": start, "last_turn": upto, "error": str(e)}
        w = {"type": "window", **w}
        windows.append(w)
        log.append(w)
        if w.get("intervene"):
            targets = ["a", "b"] if w["guidance_target"] == "both" else [w["guidance_target"].lower()]
            nxt = upto + 1
            g = {"type": "guidance", "issued_after_turn": upto, "apply_at_turn": nxt, "targets": targets,
                 "target": w["guidance_target"], "text": w["guidance"], "rationale": w.get("guidance_rationale", "")}
            guidance.append(g)
            log.append(g)
            for tgt in targets:
                pending[tgt] = g["text"]
            last_guidance_turn = upto
            say(f"  moderator note to {w['guidance_target']}: {w['guidance']}")

    try:
        while termination is None:
            idx = len(turns)
            if idx >= cfg.stop.max_turns:
                termination = "max_turns"
                break
            side = first if idx % 2 == 0 else other(first)
            prof, part = profiles[side], cfg.participant(side)
            note = pending.pop(side, None)
            if idx == 0 and cfg.seed.opening:
                ev = {"type": "turn", "idx": idx, "speaker": side, "text": cfg.seed.opening.strip(),
                      "scripted": True, "note": None,
                      "model": prof.id, "served_model": None, "usage": vars(Usage()), "cost": 0.0,
                      "latency_s": 0.0, "stop_reason": "scripted", "stop_details": None, "request_id": None,
                      "thinking": "", "truncated": False, "attempts": 0, "ts": utcnow()}
            else:
                ev = _generate(backend, cfg, prof, part, side, idx, systems[side],
                               build_messages(turns, side, first, note))
                ev["note"] = note
            turns.append(ev)
            log.append(ev)
            cost += ev["cost"] or 0.0
            say(f"[{idx + 1:>2}/{cfg.stop.max_turns}] {label(cfg, side)}: {len(M.tokens(ev['text']))} words"
                f"{' (note delivered)' if note else ''}{' [' + ev['stop_reason'] + ']' if ev['stop_reason'] not in ('end_turn', 'scripted') else ''}")

            if ev["stop_reason"] == "refusal":
                termination = f"refusal_by_{side}"
            elif not ev["text"].strip():
                termination = f"empty_turn_by_{side}"
            if termination:
                break
            if clf and (idx + 1) % cfg.classifier.window == 0:
                window_flush(idx)
            if cfg.stop.farewell and len(turns) >= 2 and M.farewell(turns[-1]["text"]) and M.farewell(turns[-2]["text"]):
                termination = "farewell"
            elif cfg.stop.stagnation_window and M.is_stagnant(
                    M.analyze_turns(turns, topic), cfg.stop.stagnation_window, cfg.stop.stagnation_threshold):
                termination = "stagnation"
            elif cfg.stop.max_cost_usd is not None and cost >= cfg.stop.max_cost_usd:
                termination = "cost_cap"
    except BackendError as e:
        log.append({"type": "end", "termination": "api_error", "error": str(e), "ts": utcnow()})
        raise

    if ended is None:
        log.append({"type": "end", "termination": termination, "ts": utcnow()})
    if turns:
        window_flush(len(turns) - 1, final=True)

    run_metrics = M.compute_run(turns, topic, [w for w in windows if "error" not in w], cfg.stop.stagnation_threshold)
    final = None
    if clf_enabled and turns:
        say("  running final analysis")
        try:
            final = clf.final(turns, [w for w in windows if "error" not in w],
                              metrics_digest(run_metrics), guidance)
        except BackendError as e:
            final = {"error": str(e)}
    elif clf is not None:
        final = {"error": "classifier unavailable"}

    summary = {
        "run_index": run_index, "topic": topic, "first_speaker": first, "termination": termination,
        "turns": len(turns), "participant_cost_usd": cost,
        "classifier_cost_usd": clf.cost if clf else 0.0,
        "classifier_usage": vars(clf.usage) if clf else None,
        "metrics": run_metrics, "final": final, "guidance": guidance,
        "windows": [w for w in windows],
        "turn_events": turns,
    }
    (run_dir / "run.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (run_dir / "transcript.md").write_text(transcript_markdown(cfg, turns, topic, first, guidance))
    return summary


def _generate(backend: Backend, cfg: Config, prof, part, side: str, idx: int, system: str,
              messages: list[dict]) -> dict:
    max_tokens = part.max_tokens or min(prof.max_output, 8000)
    usage = Usage()
    t_total = 0.0
    comp = None
    for attempt in (1, 2):
        req = Request(prof, system, messages, max_tokens, effort=part.effort, thinking=part.thinking,
                      capture_thinking=part.capture_thinking, purpose="participant",
                      meta={"side": side, "idx": idx})
        comp = backend.complete(req)
        usage.add(comp.usage)
        t_total += comp.latency_s
        # all of the budget went to reasoning: one retry with room to speak
        if comp.stop_reason == "max_tokens" and not comp.text.strip() and attempt == 1 and retry_budget(max_tokens, prof.max_output) > max_tokens:
            max_tokens = retry_budget(max_tokens, prof.max_output)
            continue
        break
    return {"type": "turn", "idx": idx, "speaker": side, "text": comp.text.strip(), "scripted": False,
            "model": prof.id, "served_model": comp.served_model, "usage": vars(usage),
            "cost": usage.cost(prof), "latency_s": t_total, "stop_reason": comp.stop_reason,
            "stop_details": comp.stop_details, "request_id": comp.request_id, "thinking": comp.thinking,
            "truncated": comp.stop_reason == "max_tokens", "attempts": attempt, "ts": utcnow()}


def transcript_markdown(cfg: Config, turns: list[dict], topic: str | None, first: str,
                        guidance: list[dict]) -> str:
    notes = {(g["apply_at_turn"], t) for g in guidance for t in g["targets"]}
    out = [f"# Transcript: {cfg.name}", "",
           f"- Speaker A: {label(cfg, 'a')} (`{cfg.a.model}`)", f"- Speaker B: {label(cfg, 'b')} (`{cfg.b.model}`)",
           f"- First speaker: {SIDE[first]}", f"- Topic: {topic or '(none)'}", ""]
    for i, t in enumerate(turns):
        tag = " *(scripted opening)*" if t.get("scripted") else ""
        out += [f"## Turn {i} — {label(cfg, t['speaker'])} (Speaker {SIDE[t['speaker']]}){tag}", ""]
        if t.get("note"):
            out += [f"> **Moderator note delivered before this turn:** {t['note']}", ""]
        out += [t["text"], ""]
    return "\n".join(out)
