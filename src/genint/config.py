"""Experiment configuration: dataclasses, YAML loading, dotted overrides, validation."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml

EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ConfigError(ValueError):
    pass


@dataclass
class Participant:
    model: str
    label: str | None = None            # display name in the report; defaults to the registry name
    effort: str | None = None           # output_config.effort; None = API default for that model
    max_tokens: int | None = None       # per-turn cap; None = min(model max, 8000)
    thinking: str = "auto"              # auto (omit the param) | adaptive
    capture_thinking: bool = False      # request summarized thinking and log it
    bias: str | None = None             # a stance this participant brings in
    system_extra: str | None = None     # free text appended to its system prompt
    prices: dict[str, float] | None = None   # override input_per_mtok / output_per_mtok / cache_read_per_mtok


@dataclass
class Seed:
    topic: str | None = None
    topics: list[str] | None = None     # multi-run: run i uses topics[i % len]
    opening: str | None = None          # fixed first message (no API call for turn 0)
    framing: str = "open"               # open | interview | custom
    framing_text: str | None = None     # used when framing == custom (same text for both sides)
    disclosure: str = "blind"           # blind | generation (tell each side which versions are talking)


@dataclass
class Stop:
    max_turns: int = 24
    max_cost_usd: float | None = None   # per run, participants only
    farewell: bool = True               # stop after two consecutive farewell turns
    stagnation_window: int | None = None    # stop after N consecutive low-novelty turns
    stagnation_threshold: float = 0.12      # compression novelty below this counts as stagnant


@dataclass
class ClassifierCfg:
    model: str = "claude-opus-5-5"
    mode: str = "observe"               # off | observe | steer
    effort: str | None = "medium"
    window: int = 4                     # turns per classifier window
    blind: bool = True                  # hide model identities from the classifier
    reveal_in_final: bool = False       # let the final write-up know which model is which
    final_samples: int = 1              # >1 repeats the final scoring to estimate classifier noise
    max_tokens: int = 8000
    cooldown_turns: int = 4             # minimum turns between interventions (steer mode)
    max_interventions: int = 3
    steer_goal: str = (
        "Intervene only when the conversation has stopped producing anything new (repetition loops, "
        "mutual-flattery spirals, endless farewells) or has drifted into something clearly unproductive. "
        "Prefer a light touch: suggest a direction, never dictate content."
    )
    prices: dict[str, float] | None = None

    def __post_init__(self):
        if self.mode is False:      # YAML 1.1 reads a bare `off` as boolean false
            self.mode = "off"


@dataclass
class Config:
    name: str
    a: Participant
    b: Participant
    seed: Seed = field(default_factory=Seed)
    stop: Stop = field(default_factory=Stop)
    classifier: ClassifierCfg = field(default_factory=ClassifierCfg)
    first_speaker: str = "a"            # a | b | alternate (alternates across runs)
    runs: int = 1
    parallel: int = 1
    output_dir: str = "runs"
    description: str | None = None

    def participant(self, side: str) -> Participant:
        return self.a if side == "a" else self.b

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, data: dict | None, where: str):
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}")
    try:
        return cls(**data)
    except TypeError as e:  # missing required keys
        raise ConfigError(f"{where}: {e}") from e


def from_dict(raw: dict) -> Config:
    raw = copy.deepcopy(raw)
    parts = raw.pop("participants", None) or {}
    if set(parts) != {"a", "b"}:
        raise ConfigError("participants must define exactly 'a' and 'b'")
    cfg = Config(
        a=_build(Participant, parts["a"], "participants.a"),
        b=_build(Participant, parts["b"], "participants.b"),
        seed=_build(Seed, raw.pop("seed", None), "seed"),
        stop=_build(Stop, raw.pop("stop", None), "stop"),
        classifier=_build(ClassifierCfg, raw.pop("classifier", None), "classifier"),
        **_pick(raw, "name first_speaker runs parallel output_dir description"),
    )
    if raw:
        raise ConfigError(f"unknown top-level key(s): {sorted(raw)}")
    validate(cfg)
    return cfg


def _pick(raw: dict, names: str) -> dict:
    out = {n: raw.pop(n) for n in names.split() if n in raw}
    if "name" not in out:
        raise ConfigError("'name' is required")
    return out


def validate(cfg: Config) -> None:
    def check(cond, msg):
        if not cond:
            raise ConfigError(msg)

    from .models import resolve
    # canonicalize aliases so the saved config records the exact model IDs used
    cfg.a.model, cfg.b.model = resolve(cfg.a.model).id, resolve(cfg.b.model).id
    cfg.classifier.model = resolve(cfg.classifier.model).id
    for side in ("a", "b"):
        p, prof = cfg.participant(side), resolve(cfg.participant(side).model)
        if prof.status != "unknown":     # unknown (newer) models: trust what the user asked for
            check(p.effort is None or prof.effort, f"participants.{side}: {prof.id} does not support an effort setting")
            check(p.thinking == "auto" and not p.capture_thinking or prof.adaptive_thinking,
                  f"participants.{side}: {prof.id} does not support adaptive thinking")
    cp = resolve(cfg.classifier.model)
    check(cfg.classifier.mode == "off" or cfg.classifier.effort is None or cp.effort or cp.status == "unknown",
          f"classifier: {cp.id} does not support an effort setting; set classifier.effort: null")
    check(cfg.first_speaker in ("a", "b", "alternate"), "first_speaker must be a, b or alternate")
    check(cfg.runs >= 1 and cfg.parallel >= 1, "runs and parallel must be >= 1")
    check(cfg.stop.max_turns >= 2, "stop.max_turns must be >= 2")
    check(cfg.seed.framing in ("open", "interview", "custom"), "seed.framing must be open, interview or custom")
    check(cfg.seed.framing != "custom" or cfg.seed.framing_text, "seed.framing_text required for framing=custom")
    check(cfg.seed.disclosure in ("blind", "generation"), "seed.disclosure must be blind or generation")
    check(not (cfg.seed.topic and cfg.seed.topics), "set seed.topic or seed.topics, not both")
    for side in ("a", "b"):
        p = cfg.participant(side)
        check(p.thinking in ("auto", "adaptive"), f"participants.{side}.thinking must be auto or adaptive")
        check(p.effort is None or p.effort in EFFORTS, f"participants.{side}.effort must be one of {EFFORTS}")
    from .backend import NON_STREAMING_MAX_TOKENS as LIMIT
    for side in ("a", "b"):
        mt = cfg.participant(side).max_tokens
        check(mt is None or 1 <= mt <= LIMIT, f"participants.{side}.max_tokens must be between 1 and {LIMIT} (non-streaming limit)")
    c = cfg.classifier
    check(c.max_tokens <= LIMIT, f"classifier.max_tokens must be <= {LIMIT} (non-streaming limit)")
    check(c.mode in ("off", "observe", "steer"), "classifier.mode must be off, observe or steer")
    check(c.window >= 1 and c.final_samples >= 1, "classifier.window and final_samples must be >= 1")
    check(c.effort is None or c.effort in EFFORTS, f"classifier.effort must be one of {EFFORTS}")


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` overrides (value parsed as YAML) to a raw config dict."""
    raw = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like key.path=value")
        path, value = item.split("=", 1)
        node = raw
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {item!r}: {k!r} is not a mapping")
        node[keys[-1]] = yaml.safe_load(value)
    return raw


def load(path: str | Path, overrides: list[str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return from_dict(apply_overrides(raw, overrides or []))


def load_resolved(path: str | Path) -> Config:
    """Load a ``config.resolved.json`` written by a previous run."""
    return from_dict(_resolved_to_raw(json.loads(Path(path).read_text())))


def _resolved_to_raw(d: dict) -> dict:
    d = dict(d)
    d["participants"] = {"a": d.pop("a"), "b": d.pop("b")}
    return d


def topic_for_run(cfg: Config, run_index: int) -> str | None:
    if cfg.seed.topics:
        return cfg.seed.topics[run_index % len(cfg.seed.topics)]
    return cfg.seed.topic


def first_speaker_for_run(cfg: Config, run_index: int) -> str:
    if cfg.first_speaker == "alternate":
        return "a" if run_index % 2 == 0 else "b"
    return cfg.first_speaker


def label(cfg: Config, side: str) -> str:
    from .models import resolve
    p = cfg.participant(side)
    return p.label or resolve(p.model, p.prices).short
