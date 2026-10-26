"""LLM backends. ``build_api_kwargs`` is pure so request shapes can be tested offline.

No sampling parameters (temperature, top_p, top_k) are ever sent: current models reject
non-default values and the 2026-10-21 API version rejects them outright, so every model
runs at its own defaults. Refusal fallbacks are deliberately NOT enabled: a silent swap to
another model would contaminate the experiment, so a refusal ends the turn and is recorded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol

from .models import ModelProfile, estimate_cost


class BackendError(RuntimeError):
    """A request failed after the SDK's own retries."""


class ModelUnavailable(BackendError):
    """The model ID is unknown to the API (typically retired)."""


class RequestConfigError(ValueError):
    """The request asks for something the model does not support."""


# The SDK refuses non-streaming requests that could run longer than 10 minutes (about 21k max_tokens).
NON_STREAMING_MAX_TOKENS = 21_000

NO_CREDENTIALS = "no credentials: set ANTHROPIC_API_KEY or run `ant auth login`"


def retry_budget(current: int, model_max: int) -> int:
    """Token budget for one retry after the first attempt ran out, within the non-streaming limit."""
    return min(current * 2, model_max, NON_STREAMING_MAX_TOKENS)


@dataclass
class Request:
    profile: ModelProfile
    system: str
    messages: list[dict]
    max_tokens: int
    effort: str | None = None
    thinking: str = "auto"              # auto | adaptive
    capture_thinking: bool = False
    schema: dict | None = None          # JSON schema for structured output
    cache: bool | None = None           # None = use profile.auto_cache
    purpose: str = "participant"        # participant | window | final | aggregate
    meta: dict = field(default_factory=dict)    # free-form, used by the mock backend


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    thinking_tokens: int = 0

    def add(self, other: "Usage") -> "Usage":
        for k in asdict(self):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        return self

    def cost(self, profile: ModelProfile) -> float | None:
        return estimate_cost(profile, self.input_tokens, self.output_tokens, self.cache_read, self.cache_write)


@dataclass
class Completion:
    text: str
    stop_reason: str | None
    usage: Usage
    latency_s: float
    thinking: str = ""
    stop_details: dict | None = None
    request_id: str | None = None
    served_model: str | None = None


class Backend(Protocol):
    def complete(self, req: Request) -> Completion: ...


def build_api_kwargs(req: Request) -> dict[str, Any]:
    p = req.profile
    if req.max_tokens > NON_STREAMING_MAX_TOKENS:
        raise RequestConfigError(f"max_tokens {req.max_tokens} exceeds the non-streaming limit of {NON_STREAMING_MAX_TOKENS}")
    kw: dict[str, Any] = {"model": p.id, "max_tokens": req.max_tokens, "messages": req.messages}
    if req.system:
        kw["system"] = req.system
    if req.thinking == "adaptive" or req.capture_thinking:
        if not p.adaptive_thinking and p.status != "unknown":
            raise RequestConfigError(f"{p.id} does not support adaptive thinking")
        thinking: dict[str, Any] = {"type": "adaptive"}
        if req.capture_thinking:
            thinking["display"] = "summarized"
        kw["thinking"] = thinking
    output_config: dict[str, Any] = {}
    if req.effort:
        if not p.effort and p.status != "unknown":
            raise RequestConfigError(f"{p.id} does not accept an effort setting (got {req.effort!r})")
        output_config["effort"] = req.effort
    if req.schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": req.schema}
    if output_config:
        kw["output_config"] = output_config
    use_cache = p.auto_cache if req.cache is None else req.cache
    if use_cache:
        kw["cache_control"] = {"type": "ephemeral"}
    return kw


class AnthropicBackend:
    def __init__(self, client=None, max_retries: int = 4, timeout: float = 600.0):
        import anthropic
        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic(max_retries=max_retries, timeout=timeout)

    def complete(self, req: Request) -> Completion:
        a = self._anthropic
        kw = build_api_kwargs(req)
        t0 = time.monotonic()
        try:
            resp = self.client.messages.create(**kw)
        except a.NotFoundError as e:
            raise ModelUnavailable(f"{req.profile.id}: {getattr(e, 'message', e)}") from e
        except a.AuthenticationError as e:
            raise BackendError("authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`") from e
        except a.APIStatusError as e:
            rid = getattr(e, "request_id", None)
            raise BackendError(f"{req.profile.id}: HTTP {e.status_code}: {getattr(e, 'message', e)} (request {rid})") from e
        except a.APIConnectionError as e:
            raise BackendError(f"{req.profile.id}: connection error: {e}") from e
        except TypeError as e:      # the SDK raises TypeError at request time when no credential resolves
            if "authentication method" in str(e):
                raise BackendError(NO_CREDENTIALS) from e
            raise
        latency = time.monotonic() - t0

        text = "".join(b.text for b in resp.content if b.type == "text")
        thinking = "".join(getattr(b, "thinking", "") or "" for b in resp.content if b.type == "thinking")
        u = resp.usage
        details = getattr(u, "output_tokens_details", None)
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
            thinking_tokens=getattr(details, "thinking_tokens", 0) or 0,
        )
        sd = getattr(resp, "stop_details", None)
        return Completion(
            text=text, stop_reason=resp.stop_reason, usage=usage, latency_s=latency, thinking=thinking,
            stop_details=sd.model_dump() if sd is not None and hasattr(sd, "model_dump") else None,
            request_id=getattr(resp, "_request_id", None), served_model=resp.model,
        )

    def model_status(self, model_id: str) -> tuple[bool, str]:
        """Free check against the Models API: (available, detail)."""
        a = self._anthropic
        try:
            m = self.client.models.retrieve(model_id)
        except a.NotFoundError:
            return False, "not found (retired or misspelled)"
        except a.AuthenticationError:
            return False, "authentication failed"
        except TypeError as e:
            if "authentication method" in str(e):
                return False, NO_CREDENTIALS
            raise
        except a.APIError as e:
            return False, f"{type(e).__name__}: {getattr(e, 'message', e)}"
        return True, f"{m.display_name}, context {m.max_input_tokens}, max output {m.max_tokens}"
