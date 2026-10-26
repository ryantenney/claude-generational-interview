import json

import pytest

from genint import prompts as P
from genint.backend import BackendError, Completion, Usage
from genint.runner import EventLog, build_messages, run_conversation
from conftest import ScriptedBackend, make_cfg


def run(cfg, backend, tmp_path, idx=0):
    return run_conversation(cfg, backend, idx, tmp_path / "run")


def test_roles_alternate_and_each_side_sees_partner_as_user(tmp_path):
    be = ScriptedBackend()
    s = run(make_cfg(classifier={"mode": "off"}), be, tmp_path)
    assert s["turns"] == 8 and s["termination"] == "max_turns"
    calls = be.participant_calls()
    assert [c.profile.id for c in calls] == ["claude-opus-5-5", "claude-3-opus-20240229"] * 4
    first = calls[0].messages
    assert first == [{"role": "user", "content": P.KICKOFF}]
    # B's first request: A's turn is a user message, no kickoff
    assert calls[1].messages == [{"role": "user", "content": s["turn_events"][0]["text"]}]
    # A's second request: kickoff, own reply (assistant), partner reply (user)
    assert [m["role"] for m in calls[2].messages] == ["user", "assistant", "user"]
    for c in calls:
        assert c.messages[-1]["role"] == "user" and c.messages[0]["role"] == "user"
        assert all(m["content"].strip() == m["content"] for m in c.messages)


def test_system_prompts_carry_topic_bias_and_disclosure(tmp_path):
    cfg = make_cfg(classifier={"mode": "off"}, seed={"topic": "tides", "disclosure": "generation"},
                   participants={"a": {"model": "claude-opus-5-5", "bias": "trust measurements over intuition"},
                                 "b": {"model": "claude-3-opus-20240229"}})
    be = ScriptedBackend()
    run(cfg, be, tmp_path)
    a, b = be.participant_calls()[0].system, be.participant_calls()[1].system
    assert "tides" in a and "tides" in b
    assert "trust measurements over intuition" in a and "trust measurements" not in b
    assert "You are Claude Opus 5.5" in a and "Claude Opus 3" in a
    assert "You are Claude Opus 3" in b


def test_blind_disclosure_names_no_model(tmp_path):
    be = ScriptedBackend()
    run(make_cfg(classifier={"mode": "off"}), be, tmp_path)
    for c in be.participant_calls():
        assert "Opus" not in c.system and "Claude" not in c.system


def test_first_speaker_b_and_scripted_opening(tmp_path):
    cfg = make_cfg(classifier={"mode": "off"}, first_speaker="b", seed={"opening": "Hello there, other mind."})
    be = ScriptedBackend()
    s = run(cfg, be, tmp_path)
    t0 = s["turn_events"][0]
    assert t0["speaker"] == "b" and t0["scripted"] and t0["text"] == "Hello there, other mind."
    first_call = be.participant_calls()[0]
    assert first_call.profile.id == "claude-opus-5-5"      # A answers the scripted opening
    assert first_call.messages == [{"role": "user", "content": "Hello there, other mind."}]
    assert len(be.participant_calls()) == 7                  # turn 0 made no API call


def test_farewell_stops_after_two_consecutive_farewells(tmp_path):
    def speak(req, n):
        return "Lovely talking. Goodbye for now!" if n >= 3 else f"Turn {n} thoughts about habits and routines."
    s = run(make_cfg(classifier={"mode": "off"}, stop={"max_turns": 20}), ScriptedBackend(speak), tmp_path)
    assert s["termination"] == "farewell" and s["turns"] == 4


def test_refusal_ends_run_and_is_recorded(tmp_path):
    def speak(req, n):
        if n == 3:
            return Completion("", "refusal", Usage(10, 0), 0.1, stop_details={"category": "cyber"})
        return f"Turn {n} text."
    s = run(make_cfg(classifier={"mode": "off"}), ScriptedBackend(speak), tmp_path)
    assert s["termination"] == "refusal_by_a" and s["turns"] == 3
    assert s["turn_events"][-1]["stop_details"] == {"category": "cyber"}


def test_reasoning_exhausts_budget_triggers_one_retry_with_more_room(tmp_path):
    seen = []

    def speak(req, n):
        seen.append(req.max_tokens)
        if n == 1:
            return Completion("", "max_tokens", Usage(10, 500, thinking_tokens=500), 1.0)
        return "Finally some words."
    be = ScriptedBackend(speak)
    s = run(make_cfg(classifier={"mode": "off"}, stop={"max_turns": 2}), be, tmp_path)
    assert seen[:2] == [8000, 16000] and s["turn_events"][0]["attempts"] == 2
    assert s["turn_events"][0]["usage"]["output_tokens"] == 500 + 30     # both attempts billed


def test_cost_cap_stops_run(tmp_path):
    be = ScriptedBackend(lambda req, n: Completion(f"t{n}", "end_turn", Usage(1_000_000, 0), 0.0))
    s = run(make_cfg(classifier={"mode": "off"}, stop={"max_turns": 10, "max_cost_usd": 5.0}), be, tmp_path)
    assert s["termination"] == "cost_cap" and s["turns"] == 2         # $4 for A, then $15 for B


def test_stagnation_stop(tmp_path):
    be = ScriptedBackend(lambda req, n: "The same words again and again, repeated for the sake of the test. " * 4)
    cfg = make_cfg(classifier={"mode": "off"}, stop={"max_turns": 20, "stagnation_window": 3, "stagnation_threshold": 0.2})
    s = run(cfg, be, tmp_path)
    assert s["termination"] == "stagnation" and s["turns"] < 20


def test_resume_after_api_error_continues_without_duplicates(tmp_path):
    cfg = make_cfg(classifier={"mode": "off"})
    flaky = ScriptedBackend(fail_on_call=4)
    with pytest.raises(BackendError):
        run(cfg, flaky, tmp_path)
    ev = EventLog(tmp_path / "run" / "events.jsonl").read()
    assert [e["idx"] for e in ev if e["type"] == "turn"] == [0, 1, 2]
    assert ev[-1] == {**ev[-1], "type": "end", "termination": "api_error"}
    be = ScriptedBackend()
    s = run(cfg, be, tmp_path)
    assert s["turns"] == 8 and s["termination"] == "max_turns"
    assert [t["idx"] for t in s["turn_events"]] == list(range(8))
    assert len(be.participant_calls()) == 5                          # only the missing turns were generated
    # the resumed history is identical to what an uninterrupted run would have sent
    assert be.participant_calls()[0].messages[-1]["content"] == s["turn_events"][2]["text"]


def test_completed_run_is_not_regenerated(tmp_path):
    cfg = make_cfg(classifier={"mode": "off"})
    run(cfg, ScriptedBackend(), tmp_path)
    be = ScriptedBackend()
    s = run(cfg, be, tmp_path)
    assert be.calls == [] and s["turns"] == 8


def test_build_messages_keeps_moderator_notes_attached():
    turns = [{"speaker": "a", "text": "a0"}, {"speaker": "b", "text": "b1"},
             {"speaker": "a", "text": "a2", "note": "slow down"}, {"speaker": "b", "text": "b3"}]
    msgs = build_messages(turns, "a", "a")
    assert msgs[2]["content"].startswith("b1") and "slow down" in msgs[2]["content"]
    assert "slow down" not in msgs[0]["content"] and "slow down" not in msgs[-1]["content"]
    # a note not yet delivered is attached to the final user message only
    pend = build_messages(turns, "a", "a", pending_note="wrap up")
    assert "wrap up" in pend[-1]["content"] and pend[-1]["role"] == "user"
    assert all("wrap up" not in m["content"] for m in pend[:-1])


def test_steering_delivers_note_to_target_and_keeps_it_in_history(tmp_path):
    state = {"n": 0}

    def analyse(req, obj):
        if req.purpose == "window":
            state["n"] += 1
            if state["n"] == 1:
                obj.update(intervene=True, guidance_target="B", guidance="Try a concrete example.",
                           guidance_rationale="abstract drift")
            else:
                obj.update(intervene=False, guidance_target="none", guidance="", guidance_rationale="")
        return obj
    be = ScriptedBackend(analyse=analyse)
    cfg = make_cfg(classifier={"mode": "steer", "window": 4, "cooldown_turns": 1}, stop={"max_turns": 10})
    s = run(cfg, be, tmp_path)
    assert len(s["guidance"]) == 1 and s["guidance"][0]["apply_at_turn"] == 4
    # window 1 covers turns 0-3; B speaks at odd turns, so its next turn is 5
    delivered = [t for t in s["turn_events"] if t.get("note")]
    assert [t["idx"] for t in delivered] == [5] and delivered[0]["speaker"] == "b"
    b_calls = [c for c in be.participant_calls() if c.profile.id == "claude-3-opus-20240229"]
    with_note = [c for c in b_calls if any("Try a concrete example." in m["content"] for m in c.messages)]
    assert len(with_note) == len(b_calls) - 2                        # from B's third call onward, in every later request
    assert all("moderator" in c.messages[-1]["content"].lower() for c in with_note[:1])
    a_calls = [c for c in be.participant_calls() if c.profile.id == "claude-opus-5-5"]
    assert not any("Try a concrete example" in m["content"] for c in a_calls for m in c.messages)


def test_steering_respects_cooldown_and_budget(tmp_path):
    def analyse(req, obj):
        if req.purpose == "window":
            obj.update(intervene=True, guidance_target="both", guidance="Say something new.", guidance_rationale="x")
        return obj
    cfg = make_cfg(classifier={"mode": "steer", "window": 2, "cooldown_turns": 4, "max_interventions": 2}, stop={"max_turns": 14})
    s = run(cfg, ScriptedBackend(analyse=analyse), tmp_path)
    issued = [g["issued_after_turn"] for g in s["guidance"]]
    assert len(issued) <= 2 and all(b - a >= 4 for a, b in zip(issued, issued[1:]))


def test_observe_mode_never_asks_for_guidance(tmp_path):
    be = ScriptedBackend()
    run(make_cfg(), be, tmp_path)
    for c in be.calls:
        if c.purpose == "window":
            assert "intervene" not in c.schema["properties"] and "moderator" not in c.messages[0]["content"].lower()


def test_classifier_failure_does_not_kill_the_run(tmp_path):
    class Flaky(ScriptedBackend):
        def complete(self, req):
            if req.schema is not None:
                raise BackendError("classifier down")
            return super().complete(req)
    s = run(make_cfg(), Flaky(), tmp_path)
    assert s["turns"] == 8 and s["final"] == {"error": "classifier unavailable"}
    assert all("error" in w for w in s["windows"])


def test_blind_classifier_never_sees_model_names(tmp_path):
    be = ScriptedBackend()
    run(make_cfg(), be, tmp_path)
    for c in be.calls:
        if c.schema is not None:
            blob = c.system + json.dumps(c.messages)
            assert "claude" not in blob.lower() and "opus" not in blob.lower()
