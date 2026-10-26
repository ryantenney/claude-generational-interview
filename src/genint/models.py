"""Model registry: capabilities, status and (approximate) pricing per model ID.

Pricing and capability flags are cached knowledge and can go stale. Costs in
reports are estimates; override per model with ``prices:`` in the config.
``genint check`` verifies each model against the live Models API.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelProfile:
    id: str
    display: str                       # e.g. "Claude Opus 5.5"
    status: str = "active"             # active | deprecated | retired | unknown
    retired_on: str | None = None
    adaptive_thinking: bool = False    # accepts thinking={"type": "adaptive"}
    effort: bool = False               # accepts output_config.effort
    default_effort: str | None = None
    max_output: int = 4096
    auto_cache: bool = False           # top-level cache_control accepted
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None

    @property
    def short(self) -> str:
        return self.display.removeprefix("Claude ").strip()


def _p(id, display, **kw) -> ModelProfile:
    return ModelProfile(id, display, **kw)


_NEW = dict(adaptive_thinking=True, effort=True, max_output=128_000, auto_cache=True)

_PROFILES = [
    _p("claude-fable-5-1", "Claude Fable 5.1", default_effort="high", input_per_mtok=10, output_per_mtok=50, cache_read_per_mtok=0.25, **_NEW),
    _p("claude-fable-5", "Claude Fable 5", default_effort="high", input_per_mtok=10, output_per_mtok=50, cache_read_per_mtok=1.0, **_NEW),
    _p("claude-opus-5-5", "Claude Opus 5.5", default_effort="medium", input_per_mtok=4, output_per_mtok=20, cache_read_per_mtok=0.20, **_NEW),
    _p("claude-opus-5", "Claude Opus 5", default_effort="high", input_per_mtok=5, output_per_mtok=25, cache_read_per_mtok=0.50, **_NEW),
    _p("claude-opus-4-8", "Claude Opus 4.8", default_effort="high", input_per_mtok=5, output_per_mtok=25, cache_read_per_mtok=0.50, **_NEW),
    _p("claude-opus-4-7", "Claude Opus 4.7", default_effort="high", input_per_mtok=5, output_per_mtok=25, cache_read_per_mtok=0.50, **_NEW),
    _p("claude-opus-4-6", "Claude Opus 4.6", default_effort="high", input_per_mtok=5, output_per_mtok=25, cache_read_per_mtok=0.50, **_NEW),
    _p("claude-opus-4-5", "Claude Opus 4.5", max_output=64_000, input_per_mtok=5, output_per_mtok=25, cache_read_per_mtok=0.50),
    _p("claude-opus-4-1-20250805", "Claude Opus 4.1", status="retired", retired_on="2026-08-05", max_output=32_000, input_per_mtok=15, output_per_mtok=75, cache_read_per_mtok=1.5),
    _p("claude-opus-4-20250514", "Claude Opus 4", status="retired", retired_on="2026-06-15", max_output=32_000, input_per_mtok=15, output_per_mtok=75, cache_read_per_mtok=1.5),
    _p("claude-3-opus-20240229", "Claude Opus 3", status="retired", retired_on="2026-01-05", max_output=4096, input_per_mtok=15, output_per_mtok=75, cache_read_per_mtok=1.5),
    _p("claude-sonnet-5-5", "Claude Sonnet 5.5", default_effort="high", input_per_mtok=2, output_per_mtok=10, cache_read_per_mtok=0.20, **_NEW),
    _p("claude-sonnet-5", "Claude Sonnet 5", default_effort="high", input_per_mtok=2, output_per_mtok=10, cache_read_per_mtok=0.20, **_NEW),
    _p("claude-sonnet-4-6", "Claude Sonnet 4.6", default_effort="high", input_per_mtok=3, output_per_mtok=15, cache_read_per_mtok=0.30, **_NEW),
    _p("claude-sonnet-4-5", "Claude Sonnet 4.5", status="deprecated", retired_on="2026-11-30", max_output=64_000, input_per_mtok=3, output_per_mtok=15, cache_read_per_mtok=0.30),
    _p("claude-3-7-sonnet-20250219", "Claude Sonnet 3.7", status="retired", retired_on="2026-02-19", max_output=64_000, input_per_mtok=3, output_per_mtok=15, cache_read_per_mtok=0.30),
    _p("claude-3-5-sonnet-20241022", "Claude Sonnet 3.5", status="retired", retired_on="2025-10-28", max_output=8192, input_per_mtok=3, output_per_mtok=15, cache_read_per_mtok=0.30),
    _p("claude-haiku-5-5", "Claude Haiku 5.5", default_effort="medium", input_per_mtok=0.10, output_per_mtok=0.50, cache_read_per_mtok=0.01, **_NEW),
    _p("claude-haiku-4-5", "Claude Haiku 4.5", max_output=64_000, input_per_mtok=1, output_per_mtok=5, cache_read_per_mtok=0.10),
    _p("claude-3-5-haiku-20241022", "Claude Haiku 3.5", status="retired", retired_on="2026-02-19", max_output=8192, input_per_mtok=0.8, output_per_mtok=4, cache_read_per_mtok=0.08),
    _p("claude-3-haiku-20240307", "Claude Haiku 3", status="retired", retired_on="2026-04-20", max_output=4096, input_per_mtok=0.25, output_per_mtok=1.25, cache_read_per_mtok=0.03),
]

REGISTRY: dict[str, ModelProfile] = {p.id: p for p in _PROFILES}

_ALIASES = {
    "opus-5.5": "claude-opus-5-5", "opus-5-5": "claude-opus-5-5", "opus5.5": "claude-opus-5-5",
    "opus-5": "claude-opus-5", "opus-4.8": "claude-opus-4-8", "opus-4.7": "claude-opus-4-7",
    "opus-4.6": "claude-opus-4-6", "opus-4.5": "claude-opus-4-5",
    "opus-4.1": "claude-opus-4-1-20250805", "opus-4": "claude-opus-4-20250514",
    "opus-3": "claude-3-opus-20240229", "opus3": "claude-3-opus-20240229",
    "claude-3-opus": "claude-3-opus-20240229", "claude-opus-3": "claude-3-opus-20240229",
    "sonnet-5.5": "claude-sonnet-5-5", "sonnet-5": "claude-sonnet-5", "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet-4.5": "claude-sonnet-4-5", "sonnet-3.7": "claude-3-7-sonnet-20250219",
    "sonnet-3.5": "claude-3-5-sonnet-20241022",
    "haiku-5.5": "claude-haiku-5-5", "haiku-4.5": "claude-haiku-4-5",
    "haiku-3.5": "claude-3-5-haiku-20241022", "haiku-3": "claude-3-haiku-20240307",
    "fable-5.1": "claude-fable-5-1", "fable-5": "claude-fable-5",
}


def resolve(model: str, overrides: dict | None = None) -> ModelProfile:
    """Map an ID or short alias to a profile. Unknown IDs get a conservative profile.

    ``overrides`` may carry ``input_per_mtok`` / ``output_per_mtok`` / ``cache_read_per_mtok``.
    """
    key = model.strip()
    key = _ALIASES.get(key.lower(), key)
    profile = REGISTRY.get(key) or _p(key, key, status="unknown")
    if overrides:
        profile = replace(profile, **{k: v for k, v in overrides.items() if k in profile.__dataclass_fields__})
    return profile


def estimate_cost(profile: ModelProfile, input_tokens: int, output_tokens: int,
                  cache_read: int = 0, cache_write: int = 0) -> float | None:
    """Estimated USD. Cache writes are billed at 1.25x input; None if prices unknown."""
    if profile.input_per_mtok is None or profile.output_per_mtok is None:
        return None
    cr = profile.cache_read_per_mtok if profile.cache_read_per_mtok is not None else profile.input_per_mtok * 0.1
    return (input_tokens * profile.input_per_mtok
            + cache_write * profile.input_per_mtok * 1.25
            + cache_read * cr
            + output_tokens * profile.output_per_mtok) / 1e6
