import pytest

from genint.backend import Request, RequestConfigError, build_api_kwargs
from genint.models import resolve

MSGS = [{"role": "user", "content": "hi"}]
BANNED = {"temperature", "top_p", "top_k"}


def req(model, **kw):
    return Request(resolve(model), "sys", MSGS, kw.pop("max_tokens", 1000), **kw)


def test_opus55_request_shape():
    kw = build_api_kwargs(req("claude-opus-5-5", effort="medium"))
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in kw          # always-on; omitting is the documented way
    assert not BANNED & set(kw)


def test_thinking_summary_capture():
    kw = build_api_kwargs(req("claude-opus-5-5", capture_thinking=True))
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    kw = build_api_kwargs(req("claude-opus-5-5", thinking="adaptive"))
    assert kw["thinking"] == {"type": "adaptive"}


def test_opus3_request_has_no_modern_params():
    kw = build_api_kwargs(req("claude-3-opus-20240229"))
    assert set(kw) == {"model", "max_tokens", "messages", "system"}
    with pytest.raises(RequestConfigError):
        build_api_kwargs(req("claude-3-opus-20240229", effort="low"))
    with pytest.raises(RequestConfigError):
        build_api_kwargs(req("claude-3-opus-20240229", thinking="adaptive"))


def test_structured_output_and_effort_combine():
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    kw = build_api_kwargs(req("claude-opus-5-5", effort="high", schema=schema))
    assert kw["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": schema}}
    assert "tool_choice" not in kw       # forced tool use is rejected on current models


def test_cache_override():
    assert "cache_control" not in build_api_kwargs(req("claude-opus-5-5", cache=False))
    assert "cache_control" in build_api_kwargs(req("claude-3-opus-20240229", cache=True))


def test_empty_system_omitted():
    r = Request(resolve("claude-opus-5-5"), "", MSGS, 100)
    assert "system" not in build_api_kwargs(r)


def test_backend_translates_sdk_response():
    from types import SimpleNamespace as NS
    from genint.backend import AnthropicBackend

    class FakeMessages:
        def create(self, **kw):
            self.kw = kw
            return NS(content=[NS(type="thinking", thinking="plan"), NS(type="text", text="hello")],
                      stop_reason="end_turn", stop_details=None, model="claude-opus-5-5", _request_id="req_1",
                      usage=NS(input_tokens=10, output_tokens=40, cache_read_input_tokens=5,
                               cache_creation_input_tokens=0, output_tokens_details=NS(thinking_tokens=25)))
    client = NS(messages=FakeMessages())
    comp = AnthropicBackend(client=client).complete(req("claude-opus-5-5"))
    assert comp.text == "hello" and comp.thinking == "plan" and comp.request_id == "req_1"
    assert (comp.usage.output_tokens, comp.usage.thinking_tokens, comp.usage.cache_read) == (40, 25, 5)


def test_non_streaming_limit_enforced_everywhere():
    from genint.backend import NON_STREAMING_MAX_TOKENS, retry_budget
    with pytest.raises(RequestConfigError, match="non-streaming"):
        build_api_kwargs(req("claude-opus-5-5", max_tokens=32000))
    assert retry_budget(8000, 128000) == 16000 and retry_budget(16000, 128000) == NON_STREAMING_MAX_TOKENS
    assert retry_budget(2048, 4096) == 4096 and retry_budget(4096, 4096) == 4096
    from conftest import make_cfg
    from genint.config import ConfigError
    with pytest.raises(ConfigError, match="non-streaming"):
        make_cfg(participants={"a": {"model": "claude-opus-5-5", "max_tokens": 32000}, "b": {"model": "claude-3-opus-20240229"}})


def test_missing_credentials_become_a_clear_error_not_a_traceback():
    from genint.backend import AnthropicBackend, BackendError, NO_CREDENTIALS
    import anthropic
    be = AnthropicBackend(client=anthropic.Anthropic(api_key=None, auth_token=None))
    ok, detail = be.model_status("claude-opus-5-5")
    assert ok is False and detail == NO_CREDENTIALS
    with pytest.raises(BackendError, match="no credentials"):
        be.complete(req("claude-opus-5-5"))
