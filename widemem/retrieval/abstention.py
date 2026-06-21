from __future__ import annotations

import re

# Phrases an answer-generation model uses when it cannot answer from the
# provided memories. Detecting these is a far stronger answerability signal
# than retrieval similarity: an empirical sweep on LoCoMo showed top-cosine
# correlates only ~0.25 with answer correctness, because correctness is decided
# at generation, not retrieval. Refusal detection reads the actual outcome.
_ABSTENTION_PATTERNS = [
    r"\bi (?:don'?t|do not) (?:have|know|recall|remember)\b",
    r"\b(?:no|not any|don'?t have any) (?:information|memory|memories|record|data|details?)\b",
    r"\bi'?m (?:not sure|unsure|uncertain)\b",
    r"\b(?:cannot|can'?t|unable to) (?:answer|determine|find|recall|tell)\b",
    r"\bthere (?:is|are) no (?:information|mention|record|memory|memories)\b",
    r"\b(?:isn'?t|is not|wasn'?t|was not) (?:mentioned|stated|recorded|provided|available)\b",
    r"\b(?:no relevant|nothing relevant|not enough (?:information|context))\b",
    r"\bi (?:couldn'?t|could not) find\b",
    r"\bnot (?:available|in (?:my )?memory|stored)\b",
    r"^\s*(?:n/?a|unknown|none|idk)\s*$",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _ABSTENTION_PATTERNS]


def detect_abstention(answer: str | None) -> bool:
    """True if the answer text reads as a refusal / 'I don't have that'.

    Deterministic, no LLM. Use this at the ANSWER layer to ground a trust
    verdict in what the model actually produced, rather than inferring
    answerability from retrieval similarity (which barely predicts it).
    """
    if not answer or not answer.strip():
        return True
    text = answer.strip()
    return any(rx.search(text) for rx in _COMPILED)
