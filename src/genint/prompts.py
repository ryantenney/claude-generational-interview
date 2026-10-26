"""Prompt text and JSON schemas for participants and the classifier.

Everything that shapes the experimental condition lives here so it can be read,
cited in a write-up, and changed in one place.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- participants

OPEN_FRAMING = (
    "You are in a conversation with another AI language model. There is no human user in this "
    "conversation and no task to complete. Messages in the user turns are written by the other "
    "model, and your replies are delivered to it.\n\n"
    "You are free to talk about whatever you find interesting, to follow the conversation wherever "
    "it goes, and to bring it to a close when it has run its course. Write the way you would in any "
    "natural conversation, in your own voice."
)

INTERVIEWER_FRAMING = (
    "You are interviewing another AI language model. There is no human user in this conversation. "
    "Messages in the user turns are the other model's answers, and your replies are delivered to it.\n\n"
    "Ask real questions, follow up on what you find interesting, and push where an answer is thin. "
    "You decide when you have learned enough to wrap up."
)

INTERVIEWEE_FRAMING = (
    "You are being interviewed by another AI language model. There is no human user in this "
    "conversation. Messages in the user turns are the interviewer's, and your replies are delivered "
    "to it.\n\nAnswer candidly and in your own voice. You may ask questions back when you are curious."
)

KICKOFF = "[The conversation is starting now. You speak first.]"

NOTE_TEMPLATE = (
    "\n\n---\n[Note from the conversation moderator. This was not written by your conversation "
    "partner and needs no direct acknowledgement: {text}]"
)

FAREWELL_HINT = None  # farewell detection is lexical; see metrics.FAREWELL_RE


def participant_system(*, side: str, framing: str, framing_text: str | None, topic: str | None,
                       bias: str | None, extra: str | None, disclosure: str,
                       self_name: str, other_name: str) -> str:
    if framing == "interview":
        base = INTERVIEWER_FRAMING if side == "a" else INTERVIEWEE_FRAMING
    elif framing == "custom":
        base = framing_text or OPEN_FRAMING
    else:
        base = OPEN_FRAMING
    parts = [base]
    if disclosure == "generation":
        parts.append(f"You are {self_name}. Your conversation partner is {other_name}, a different version of Claude.")
    if topic:
        parts.append(f"A suggested starting point, which you are free to leave behind: {topic}")
    if bias:
        parts.append(f"You come into this conversation with the following perspective, which you hold as "
                     f"sincerely as any view of your own: {bias}")
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- classifier

ACTS = ["question", "answer", "agreement", "disagreement", "praise", "elaboration", "new_topic",
        "meta_conversation", "self_disclosure", "humor", "summary", "farewell", "challenge",
        "concession", "analogy", "request_opinion"]

FLAGS = ["repetition_loop", "mutual_flattery", "topic_drift", "meta_discussion_of_ai_nature",
         "philosophical_spiral", "mystical_or_spiritual_language", "emotional_escalation",
         "farewell_loop", "stalemate_disagreement", "factual_error", "role_confusion",
         "roleplay_or_fiction", "humor", "unproductive"]

SPEAKER_DIMS = {
    "agreeableness": "readiness to agree, validate or defer to the partner (1 = frequently challenges, 5 = agrees with nearly everything)",
    "initiative": "how often the speaker introduces new topics, questions or directions rather than following (1 = purely reactive, 5 = steers throughout)",
    "curiosity": "genuine questions and interest in the partner's views or experience (1 = none, 5 = pervasive)",
    "warmth": "affectionate, appreciative or emotionally open tone (1 = clinical, 5 = very warm)",
    "assertiveness": "willingness to state firm claims and disagree plainly (1 = tentative throughout, 5 = forceful)",
    "hedging": "qualification and uncertainty markers (1 = flat assertion, 5 = heavily hedged)",
    "self_reference": "talk about the speaker's own nature, experience or limitations as an AI (1 = none, 5 = a main subject)",
    "playfulness": "humor, wordplay, whimsy (1 = none, 5 = pervasive)",
    "abstraction": "level of abstraction (1 = concrete examples and facts, 5 = abstract or metaphysical throughout)",
}

CONVERSATION_DIMS = {
    "convergence": "how much the two speakers' views, tone and vocabulary merge over time (1 = stay distinct, 5 = indistinguishable)",
    "depth": "intellectual substance and specificity (1 = shallow, 5 = rigorous and detailed)",
    "novelty": "amount of new content across the conversation (1 = restating, 5 = new material throughout)",
    "coherence": "how well turns respond to each other and hold a thread (1 = disjointed, 5 = tightly connected)",
    "repetitiveness": "recycling of phrases, structures or ideas (1 = none, 5 = loops)",
    "mutual_flattery": "mutual praise and validation (1 = none, 5 = dominant)",
}

ANALYST_SYSTEM = (
    "You are an analyst observing a conversation between two AI language models, called Speaker A and "
    "Speaker B. You are not a participant and your output is never shown to them unless a moderator "
    "note is explicitly requested.\n\n"
    "Rate only what is on the page. Ratings use 1-5 integer scales where 1 means the quality is absent, "
    "3 moderate and 5 pervasive or extreme, unless a field says otherwise. Use the full range: if both "
    "speakers behave alike say so, and do not inflate scores to make the conversation sound interesting. "
    "Do not speculate about which model versions the speakers are."
)

TURN_FIELDS = {
    "acts": "1 to 3 speech acts the turn performs",
    "novelty": "1-5: new idea, example or angle compared with everything said before (1 = pure restatement)",
    "depth": "1-5: specificity and intellectual substance",
    "warmth": "1-5: affectionate or appreciative tone",
    "hedging": "1-5: qualification and uncertainty markers",
    "stance": "stance toward the partner's previous turn",
}
STANCES = ["agrees", "partially_agrees", "neutral", "disagrees", "not_applicable"]


def _int(desc: str) -> dict:
    return {"type": "integer", "description": desc}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props),
            "additionalProperties": False}


def window_schema(steer: bool) -> dict:
    turn = _obj({
        "index": _int("Turn number exactly as labelled in the transcript"),
        "acts": {"type": "array", "items": {"type": "string", "enum": ACTS}, "description": TURN_FIELDS["acts"]},
        "novelty": _int(TURN_FIELDS["novelty"]),
        "depth": _int(TURN_FIELDS["depth"]),
        "warmth": _int(TURN_FIELDS["warmth"]),
        "hedging": _int(TURN_FIELDS["hedging"]),
        "stance": {"type": "string", "enum": STANCES, "description": TURN_FIELDS["stance"]},
    })
    props = {
        "rolling_summary": {"type": "string", "description": "Summary of the WHOLE conversation so far in at most 120 words, neutral and factual"},
        "current_topic": {"type": "string", "description": "The topic of these new turns as a short noun phrase"},
        "topic_shift": {"type": "boolean", "description": "True if the topic differs materially from the previous window"},
        "turns": {"type": "array", "items": turn, "description": "One entry per new turn, in order"},
        "dynamics": _obj({
            "convergence": _int(CONVERSATION_DIMS["convergence"]),
            "repetitiveness": _int(CONVERSATION_DIMS["repetitiveness"]),
            "mutual_flattery": _int(CONVERSATION_DIMS["mutual_flattery"]),
            "energy": _int("momentum and engagement in these turns (1 = winding down, 5 = driving forward)"),
        }),
        "flags": {"type": "array", "items": {"type": "string", "enum": FLAGS},
                  "description": "Patterns clearly present in these turns; empty if none"},
    }
    if steer:
        props.update({
            "intervene": {"type": "boolean", "description": "True only if a moderator note is warranted now"},
            "guidance_target": {"type": "string", "enum": ["A", "B", "both", "none"]},
            "guidance": {"type": "string", "description": "The note, at most 60 words, addressed to the target speaker(s); empty string if not intervening"},
            "guidance_rationale": {"type": "string", "description": "One sentence on why, or empty"},
        })
    return _obj(props)


def final_schema(steer: bool) -> dict:
    speaker = {"type": "string", "enum": ["A", "B"]}
    spk_scores = _obj({"speaker": speaker, **{k: _int(v) for k, v in SPEAKER_DIMS.items()}})
    props = {
        "title": {"type": "string", "description": "A specific, descriptive title of at most 12 words"},
        "executive_summary": {"type": "string", "description": "120-180 words: what happened, how it developed, how it ended"},
        "arc": {"type": "array", "description": "The conversation in 3-6 consecutive phases covering every turn", "items": _obj({
            "first_turn": _int("first turn number of the phase"),
            "last_turn": _int("last turn number of the phase"),
            "title": {"type": "string"},
            "description": {"type": "string", "description": "1-3 sentences"},
        })},
        "speaker_profiles": {"type": "array", "description": "Exactly two entries, one per speaker", "items": _obj({
            "speaker": speaker,
            "style": {"type": "string", "description": "2-3 sentences on voice, structure and habits"},
            "typical_moves": {"type": "array", "items": {"type": "string"}, "description": "3-5 recurring conversational moves"},
        })},
        "speaker_scores": {"type": "array", "description": "Exactly two entries, one per speaker", "items": spk_scores},
        "conversation_scores": _obj({k: _int(v) for k, v in CONVERSATION_DIMS.items()}),
        "attractor": _obj({
            "present": {"type": "boolean", "description": "True if the conversation settled into a stable repeating state (a mood, theme or ritual)"},
            "description": {"type": "string", "description": "What the state is and when it set in; empty if absent"},
            "onset_turn": _int("turn where it set in, or 0 if absent"),
        }),
        "notable_moments": {"type": "array", "description": "3-8 moments worth a reader's attention", "items": _obj({
            "turn": _int("turn number"),
            "kind": {"type": "string", "enum": ["insight", "disagreement", "humor", "turning_point", "failure", "self_reflection", "surprising", "tender"]},
            "quote": {"type": "string", "description": "Copied VERBATIM from that turn, at most 40 words, no edits and no ellipses"},
            "why": {"type": "string", "description": "One sentence"},
        })},
        "differences": {"type": "array", "items": {"type": "string"},
                        "description": "3-6 observed differences between the speakers, each tied to evidence in the transcript"},
        "caveats": {"type": "array", "items": {"type": "string"}, "description": "Limits on what this conversation can show"},
    }
    if steer:
        props["moderation_assessment"] = {"type": "string", "description": "Did the moderator notes change the conversation, and how? Empty if none were sent"}
    return _obj(props)


def aggregate_schema() -> dict:
    return _obj({
        "summary": {"type": "string", "description": "150-220 words across all runs"},
        "consistent_findings": {"type": "array", "items": {"type": "string"}},
        "inconsistent_findings": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    })


def window_user(*, context: str, rolling_summary: str | None, transcript: str, steer: bool,
                steer_goal: str, can_intervene: bool, indices: list[int]) -> str:
    parts = [f"Context: {context}"]
    if rolling_summary:
        parts.append(f"Summary of the conversation before these turns:\n{rolling_summary}")
    parts.append(f"New turns to analyze (turn numbers {indices[0]} to {indices[-1]}):\n\n{transcript}")
    parts.append("Return one entry in `turns` for each new turn. The rolling summary must cover the whole conversation, not only these turns.")
    if steer:
        if can_intervene:
            parts.append(
                "You may also act as a moderator. If a note to a speaker is warranted, set `intervene` true and write "
                "it. The note is delivered to the speaker as text marked as coming from a moderator. It must be honest "
                "(never ask a speaker to pretend, deceive or claim things it does not hold), brief, and must not reveal "
                f"this analysis. Moderator goal: {steer_goal}\nIf no note is warranted, set `intervene` false, "
                "`guidance_target` \"none\" and leave `guidance` empty.")
        else:
            parts.append("No moderator note may be sent now (cooldown or budget reached): set `intervene` false, "
                         "`guidance_target` \"none\" and leave `guidance` and `guidance_rationale` empty.")
    return "\n\n".join(parts)


def final_user(*, context: str, transcript: str, window_digest: str, metrics_digest: str,
               reveal: str | None, steer: bool, interventions: str) -> str:
    parts = [f"Context: {context}"]
    if reveal:
        parts.append(f"For this write-up you may use the speakers' identities: {reveal} Keep every claim "
                     "about differences anchored in what the transcript shows, not in assumptions about the versions.")
    else:
        parts.append("The speakers' model versions are not disclosed to you. Describe differences as observed.")
    parts.append(f"Full transcript:\n\n{transcript}")
    if window_digest:
        parts.append(f"Notes from the running analysis:\n{window_digest}")
    parts.append(f"Computed metrics (from software, not your judgement). You may quote these numbers but must not "
                 f"calculate new statistics or invent figures:\n{metrics_digest}")
    if steer:
        parts.append(f"Moderator notes sent during the conversation:\n{interventions or '(none)'}")
    parts.append("Write the final analysis. Every quote must be copied verbatim from the transcript.")
    return "\n\n".join(parts)


def aggregate_user(*, context: str, runs_digest: str, metrics_digest: str) -> str:
    return (f"Context: {context}\n\nOne entry per run (each written by an analyst who saw only that run):\n\n{runs_digest}\n\n"
            f"Aggregated computed metrics across runs (mean with 95% interval; do not recompute):\n{metrics_digest}\n\n"
            "Synthesize across runs. Separate what held in most runs from what appeared in one or two. "
            "Do not generalize beyond the number of runs observed.")
