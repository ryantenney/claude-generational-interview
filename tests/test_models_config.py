import pytest

from genint import config as C
from genint.models import estimate_cost, resolve
from conftest import make_cfg


def test_aliases_and_registry():
    assert resolve("opus-5.5").id == "claude-opus-5-5"
    assert resolve("opus-3").id == "claude-3-opus-20240229"
    assert resolve("opus-3").status == "retired" and resolve("opus-3").retired_on == "2026-01-05"
    assert resolve("claude-opus-5-5").adaptive_thinking and resolve("claude-opus-5-5").effort
    assert not resolve("opus-3").effort and not resolve("opus-3").adaptive_thinking


def test_unknown_model_is_conservative():
    p = resolve("claude-future-9")
    assert p.status == "unknown" and not p.effort and not p.adaptive_thinking and not p.auto_cache
    assert estimate_cost(p, 1000, 1000) is None


def test_price_override_and_cost():
    p = resolve("claude-opus-5-5", {"input_per_mtok": 10.0})
    assert p.input_per_mtok == 10.0 and p.output_per_mtok == 20
    assert estimate_cost(resolve("claude-opus-5-5"), 1_000_000, 1_000_000) == pytest.approx(24.0)
    # cache reads are cheaper than fresh input
    assert estimate_cost(resolve("claude-opus-5-5"), 0, 0, cache_read=1_000_000) == pytest.approx(0.2)


def test_config_rejects_unknown_keys_and_bad_values():
    with pytest.raises(C.ConfigError, match="unknown key"):
        make_cfg(seed={"topc": "typo"})
    with pytest.raises(C.ConfigError):
        make_cfg(first_speaker="c")
    with pytest.raises(C.ConfigError):
        make_cfg(participants={"a": {"model": "x", "effort": "turbo"}, "b": {"model": "y"}})
    with pytest.raises(C.ConfigError, match="exactly"):
        C.from_dict({"name": "x", "participants": {"a": {"model": "m"}}})


def test_dotted_overrides_parse_yaml_values():
    raw = {"name": "x", "participants": {"a": {"model": "m"}, "b": {"model": "n"}}, "stop": {"max_turns": 10}}
    out = C.apply_overrides(raw, ["stop.max_turns=4", "participants.b.model=claude-opus-4-5", "seed.topic=hello world",
                                  "classifier.mode=off"])
    assert out["stop"]["max_turns"] == 4 and out["participants"]["b"]["model"] == "claude-opus-4-5"
    assert out["seed"]["topic"] == "hello world"
    assert C.from_dict(out).classifier.mode == "off"      # YAML reads bare `off` as False; config normalizes it
    with pytest.raises(C.ConfigError):
        C.apply_overrides(raw, ["nonsense"])


def test_yaml_file_with_bare_off(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("name: x\nparticipants: {a: {model: m}, b: {model: n}}\nclassifier: {mode: off}\n")
    assert C.load(f).classifier.mode == "off"


def test_per_run_topic_and_first_speaker():
    cfg = make_cfg(seed={"topic": None, "topics": ["t0", "t1"]}, first_speaker="alternate")
    assert [C.topic_for_run(cfg, i) for i in range(3)] == ["t0", "t1", "t0"]
    assert [C.first_speaker_for_run(cfg, i) for i in range(3)] == ["a", "b", "a"]
    with pytest.raises(C.ConfigError):
        make_cfg(seed={"topic": "x", "topics": ["y"]})


def test_shipped_configs_load():
    from pathlib import Path
    for f in Path(__file__).parent.parent.joinpath("configs").glob("*.yaml"):
        C.load(f)


def test_aliases_are_canonicalized_in_config():
    cfg = make_cfg(participants={"a": {"model": "opus-5.5"}, "b": {"model": "opus-3"}}, classifier={"model": "sonnet-5.5"})
    assert (cfg.a.model, cfg.b.model, cfg.classifier.model) == ("claude-opus-5-5", "claude-3-opus-20240229", "claude-sonnet-5-5")


def test_unsupported_settings_fail_at_config_time_for_known_models_only():
    with pytest.raises(C.ConfigError, match="does not support an effort"):
        make_cfg(participants={"a": {"model": "claude-opus-5-5"}, "b": {"model": "claude-3-opus-20240229", "effort": "low"}})
    with pytest.raises(C.ConfigError, match="adaptive thinking"):
        make_cfg(participants={"a": {"model": "claude-opus-5-5"}, "b": {"model": "opus-3", "thinking": "adaptive"}})
    # a model the registry has never heard of: the user knows best
    cfg = make_cfg(participants={"a": {"model": "claude-opus-9", "effort": "high"}, "b": {"model": "opus-3"}})
    from genint.backend import Request, build_api_kwargs
    from genint.models import resolve
    kw = build_api_kwargs(Request(resolve(cfg.a.model), "", [{"role": "user", "content": "x"}], 100, effort="high"))
    assert kw["output_config"] == {"effort": "high"} and "cache_control" not in kw
