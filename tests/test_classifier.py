import json

from genint import prompts as P
from genint.classifier import Classifier, _clamp, parse_json, render_transcript, verify_quote
from conftest import ScriptedBackend, make_cfg

TURNS = [{"speaker": "a", "text": "The sea “keeps” its own — counsel."}, {"speaker": "b", "text": "I agree, more or less."}]


def test_parse_json_tolerates_fences_and_rejects_junk():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("not json") is None and parse_json("[1,2]") is None


def test_quote_verification_normalizes_punctuation_and_locates_misattributed_quotes():
    assert verify_quote('The sea "keeps" its own - counsel.', TURNS, 0) == {"verified": True, "found_in": 0}
    assert verify_quote("I agree, more or less.", TURNS, 0) == {"verified": False, "found_in": 1}
    assert verify_quote("invented words", TURNS, 0) == {"verified": False, "found_in": None}
    assert verify_quote("", TURNS, 0)["verified"] is False


def test_clamp():
    assert _clamp(9) == 5 and _clamp(-3) == 1 and _clamp("4") == 4 and _clamp(None) is None and _clamp("x") is None


def test_render_transcript_labels_turns_and_speakers():
    out = render_transcript(TURNS, 0, 1)
    assert out.startswith("Turn 0 — Speaker A:") and "Turn 1 — Speaker B:" in out


def test_schemas_are_strict_objects():
    def walk(s):
        if s.get("type") == "object":
            assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
            for v in s["properties"].values():
                walk(v)
        if s.get("type") == "array":
            walk(s["items"])
    for steer in (False, True):
        walk(P.window_schema(steer)); walk(P.final_schema(steer))
    walk(P.aggregate_schema())
    assert "intervene" in P.window_schema(True)["properties"] and "intervene" not in P.window_schema(False)["properties"]


def test_window_normalization_drops_unknown_turns_and_bad_enums():
    cfg = make_cfg()
    clf = Classifier(cfg, ScriptedBackend(), 0)
    raw = {"rolling_summary": "s", "current_topic": "t", "topic_shift": False,
           "turns": [{"index": 2, "acts": ["question", "bogus"], "novelty": 9, "depth": 0, "warmth": 3, "hedging": 3, "stance": "weird"},
                     {"index": 99, "acts": [], "novelty": 1, "depth": 1, "warmth": 1, "hedging": 1, "stance": "agrees"}],
           "dynamics": {"convergence": 7, "repetitiveness": 1, "mutual_flattery": 2, "energy": 3}, "flags": ["repetition_loop", "nope"]}
    w = clf._norm_window(raw, [2, 3], can_intervene=False)
    assert [t["index"] for t in w["turns"]] == [2]
    t = w["turns"][0]
    assert t["acts"] == ["question"] and t["novelty"] == 5 and t["depth"] == 1 and t["stance"] == "not_applicable"
    assert w["dynamics"]["convergence"] == 5 and w["flags"] == ["repetition_loop"]


def test_intervention_is_ignored_when_not_allowed_or_malformed():
    cfg = make_cfg(classifier={"mode": "steer"})
    clf = Classifier(cfg, ScriptedBackend(), 0)
    base = {"rolling_summary": "", "current_topic": "", "topic_shift": False, "turns": [], "dynamics": {}, "flags": [],
            "intervene": True, "guidance_target": "B", "guidance": "do x", "guidance_rationale": "r"}
    assert clf._norm_window(base, [0], can_intervene=False)["intervene"] is False
    assert clf._norm_window({**base, "guidance": "  "}, [0], can_intervene=True)["intervene"] is False
    assert clf._norm_window({**base, "guidance_target": "none"}, [0], can_intervene=True)["intervene"] is False
    ok = clf._norm_window(base, [0], can_intervene=True)
    assert ok["intervene"] and ok["guidance_target"] == "B" and ok["guidance"] == "do x"


def test_unparseable_output_is_retried_with_more_room_then_reported():
    from genint.backend import Completion, Usage
    seen = []

    class Bad(ScriptedBackend):
        def complete(self, req):
            seen.append(req.max_tokens)
            return Completion("{broken", "max_tokens", Usage(1, 1), 0.0)
    clf = Classifier(make_cfg(), Bad(), 0)
    w = clf.window(TURNS, 0, 1, None, False)
    assert seen == [8000, 16000] and "error" in w and "unparseable" in w["error"]


def test_final_scores_averaged_over_samples_and_quotes_verified():
    cfg = make_cfg(classifier={"final_samples": 3})
    n = {"i": 0}

    def analyse(req, obj):
        if req.purpose == "final":
            n["i"] += 1
            for s in obj["speaker_scores"]:
                s["warmth"] = n["i"] + 1          # 2, 3, 4
            obj["notable_moments"] = [{"turn": 0, "kind": "insight", "quote": "counsel", "why": "w"}]
    clf = Classifier(cfg, ScriptedBackend(analyse=analyse), 0)
    f = clf.final(TURNS, [], "digest", [])
    assert f["samples"] == 3 and f["speaker_scores"]["a"]["warmth"] == 3.0
    assert f["score_sd"]["speakers"]["a"]["warmth"] == 1.0
    assert f["notable_moments"][0]["verified"] is True


def test_reveal_only_in_final_when_enabled():
    cfg = make_cfg(classifier={"reveal_in_final": True})
    be = ScriptedBackend()
    clf = Classifier(cfg, be, 0)
    clf.window(TURNS, 0, 1, None, False)
    clf.final(TURNS, [], "d", [])
    win = json.dumps(be.calls[0].messages); fin = json.dumps(be.calls[1].messages)
    assert "Opus" not in win and "Speaker A is Opus 5.5" in fin and "Opus 3" in fin
