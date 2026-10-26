import json
import random

import pytest

from genint import config as C
from genint.backend import Completion, Request, Usage
from genint.mock import _fill


def make_cfg(**over):
    raw = {
        "name": "t",
        "participants": {"a": {"model": "claude-opus-5-5", "effort": "medium"},
                         "b": {"model": "claude-3-opus-20240229", "max_tokens": 512}},
        "seed": {"topic": "how habits form"},
        "stop": {"max_turns": 8, "farewell": True},
        "classifier": {"model": "claude-opus-5-5", "mode": "observe", "window": 4},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict) and k != "participants":
            raw[k] = {**raw[k], **v}
        else:
            raw[k] = v
    return C.from_dict(raw)


class ScriptedBackend:
    """Records every request. ``speak(req, n)`` returns text (or a Completion) for participants;
    ``analyse(req, obj)`` may edit the schema-filled classifier object."""

    def __init__(self, speak=None, analyse=None, fail_on_call=None):
        self.calls: list[Request] = []
        self.speak = speak or (lambda req, n: f"Reply {n}. What do you think?")
        self.analyse = analyse
        self.fail_on_call = fail_on_call
        self.n_participant = 0

    def complete(self, req: Request) -> Completion:
        from genint.backend import BackendError
        self.calls.append(req)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise BackendError("boom")
        if req.schema is not None:
            obj = _fill(req.schema, random.Random(len(self.calls)), req.meta)
            if self.analyse:
                obj = self.analyse(req, obj) or obj
            return Completion(json.dumps(obj), "end_turn", Usage(100, 50), 0.0)
        self.n_participant += 1
        out = self.speak(req, self.n_participant)
        if isinstance(out, Completion):
            return out
        return Completion(out, "end_turn", Usage(100, 30, thinking_tokens=10 if req.profile.adaptive_thinking else 0), 0.5)

    def participant_calls(self):
        return [c for c in self.calls if c.schema is None]


@pytest.fixture
def cfg():
    return make_cfg()
