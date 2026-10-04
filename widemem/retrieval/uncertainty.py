"""Uncertainty-aware retrieval: confidence assessment, frustration detection, and response guidance."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from widemem.core.types import (
    MemorySearchResult,
    RetrievalConfidence,
    UncertaintyMode,
)

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS = {
    # Calibrated for text-embedding-3-small, whose cosine similarity has a high
    # baseline: unrelated short texts routinely score ~0.35-0.50, so the old
    # high=0.45 read an unrelated memory as "high confidence / safe to answer"
    # (the explain=True false positive). Genuine matches sit ~0.6+. Used for any
    # embedding model without its own entry below.
    "high": 0.60,
    "moderate": 0.50,
    "low": 0.30,
}

_MODEL_THRESHOLDS = {
    # Measured on tests/fixtures/confidence_minilm.json: facts stored by a real
    # add() with llama3.1:8b, 62 labeled queries (38 answerable, 12 questions
    # about the same person with no stored answer, 12 unrelated), split by person.
    # Top-1 similarity: unrelated 0.07-0.35, answerable 0.42-0.88. The answer
    # fact with the subject stripped ("moved to Boston", the shape llama3.1:8b
    # often stores) scores 0.12-0.57; recall at MODERATE+ on those goes from 19%
    # to 77% (train) and 17% to 58% (holdout) at moderate=0.30; 0.38 would lose
    # half of that. Cost: hard negatives at MODERATE+ go from 8/12 to 12/12, one
    # unrelated query (0.351) moves LOW -> MODERATE, and precision on the whole
    # fixture drops from 0.771 to 0.745. MiniLM confidence cannot separate
    # "answer stored" from "something about this person stored": the name in the
    # query dominates the score. Separating them is follow-up research.
    "all-minilm-l6-v2": {"high": 0.60, "moderate": 0.30, "low": 0.20},
}

_warned_non_monotonic: set[tuple[float, float, float]] = set()


def _model_key(embedding_model: object) -> str:
    if not isinstance(embedding_model, str):
        return ""
    return embedding_model.strip().lower().rsplit("/", 1)[-1]


def get_confidence_thresholds(embedding_model: Optional[str] = None) -> dict[str, float]:
    """Thresholds for the given embedding model; WIDEMEM_CONFIDENCE_* env vars win."""
    defaults = _MODEL_THRESHOLDS.get(_model_key(embedding_model), _DEFAULT_THRESHOLDS)
    thresholds = {
        "high": float(os.environ.get("WIDEMEM_CONFIDENCE_HIGH", defaults["high"])),
        "moderate": float(os.environ.get("WIDEMEM_CONFIDENCE_MODERATE", defaults["moderate"])),
        "low": float(os.environ.get("WIDEMEM_CONFIDENCE_LOW", defaults["low"])),
    }
    key = (thresholds["high"], thresholds["moderate"], thresholds["low"])
    if not key[0] >= key[1] >= key[2] and key not in _warned_non_monotonic:
        _warned_non_monotonic.add(key)
        logger.warning(
            "Confidence thresholds are not ordered high >= moderate >= low "
            "(high=%s, moderate=%s, low=%s); check WIDEMEM_CONFIDENCE_* overrides.",
            *key,
        )
    return thresholds

FRUSTRATION_SIGNALS = (
    "i told you", "i already said", "remember when i", "i mentioned",
    "you forgot", "you should know", "we talked about", "i said before",
    "don't you remember", "how could you forget", "i literally told",
    "we discussed", "you should remember", "i specifically said",
)


def assess_confidence(
    results: list[MemorySearchResult], embedding_model: Optional[str] = None
) -> RetrievalConfidence:
    """Assess how confident we are that the search results are relevant."""
    if not results:
        return RetrievalConfidence.NONE

    top_result = results[0]
    top_sim = (
        top_result.raw_similarity_score
        if top_result.raw_similarity_score is not None
        else top_result.similarity_score
    )
    thresholds = get_confidence_thresholds(embedding_model)

    if top_sim >= thresholds["high"]:
        return RetrievalConfidence.HIGH
    if top_sim >= thresholds["moderate"]:
        return RetrievalConfidence.MODERATE
    if top_sim >= thresholds["low"]:
        return RetrievalConfidence.LOW
    return RetrievalConfidence.NONE


def detect_frustration(query: str) -> bool:
    """Detect if the user is frustrated about a forgotten fact."""
    q = query.lower()
    return any(signal in q for signal in FRUSTRATION_SIGNALS)


def build_uncertainty_guidance(
    confidence: RetrievalConfidence,
    mode: UncertaintyMode,
    results: list[MemorySearchResult],
) -> dict | None:
    """Build guidance about how to handle uncertain retrieval.

    Returns None if confidence is HIGH (answer normally).
    Otherwise returns a dict with:
        action: "answer" | "hedge" | "refuse" | "offer_guess"
        message: human-readable uncertainty note
        related: list of related memory snippets (if any)
    """
    if confidence == RetrievalConfidence.HIGH:
        return None

    if confidence == RetrievalConfidence.NONE:
        if mode == UncertaintyMode.STRICT:
            return {"action": "refuse", "message": "I don't have any memories about this."}
        if mode == UncertaintyMode.HELPFUL:
            return {"action": "refuse", "message": "I don't have specific information about this stored."}
        return {
            "action": "offer_guess",
            "message": "I don't have this in my memory. I can take a guess based on what I do know, if you'd like.",
        }

    related = [r.memory.content[:80] for r in results[:3]] if results else []

    if confidence == RetrievalConfidence.LOW:
        if mode == UncertaintyMode.STRICT:
            return {"action": "refuse", "message": "I'm not confident I have relevant information about this."}
        if mode == UncertaintyMode.HELPFUL:
            return {
                "action": "hedge",
                "message": "I don't have a direct answer, but I know some related things.",
                "related": related,
            }
        return {
            "action": "offer_guess",
            "message": "I'm not sure about this, but I have some related memories. Want me to piece something together?",
            "related": related,
        }

    # MODERATE confidence
    if mode == UncertaintyMode.STRICT:
        return {"action": "hedge", "message": "I have some information but I'm not fully certain."}
    return {
        "action": "answer",
        "message": "Based on what I remember (though I'm not 100% certain):",
    }


def extract_forgotten_fact(query: str) -> Optional[str]:
    """Try to extract the fact the user is reminding us about.

    Examples:
        "I told you my blood type is O negative!" → "blood type is O negative"
        "Remember I'm allergic to peanuts?" → "allergic to peanuts"
        "You forgot I live in San Francisco" → "live in San Francisco"
    """
    q = query.strip()
    # Patterns: "I told you [fact]", "Remember [fact]", "You forgot [fact]"
    patterns = [
        r"(?:i\s+told\s+you|i\s+already\s+said|i\s+mentioned)\s+(?:that\s+)?(.+?)[\.\!\?]?$",
        r"(?:remember\s+(?:that\s+)?(?:i\s+)?(?:said\s+)?(?:that\s+)?)(.+?)[\.\!\?]?$",
        r"(?:you\s+forgot|don'?t\s+you\s+remember)\s+(?:that\s+)?(?:i\s+)?(.+?)[\.\!\?]?$",
        r"(?:i\s+specifically\s+said|i\s+literally\s+told\s+you)\s+(?:that\s+)?(.+?)[\.\!\?]?$",
    ]
    vague_phrases = {"something", "something important", "that", "this", "it",
                     "stuff", "things", "that thing", "about that", "about it",
                     "this already", "that already", "this before", "that before"}
    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            fact = match.group(1).strip().rstrip("!.?")
            if len(fact) > 8 and fact.lower() not in vague_phrases:
                return fact
    return None


def build_frustration_response(
    query: str,
    confidence: RetrievalConfidence,
    mode: UncertaintyMode,
) -> Optional[dict]:
    """Handle frustrated user who thinks the system forgot something.

    Returns None if no frustration detected.
    Otherwise returns guidance for how to respond, including
    the extracted fact to pin.

    Only HIGH confidence reassures. MODERATE means a memory about the same
    person or topic exists, not that this fact was stored: under
    all-MiniLM-L6-v2, "I told you I moved to Boston!" scores 0.42 (MODERATE)
    against "live in San Francisco" alone, and 0.80 (HIGH) once Boston is
    stored. Reassuring on MODERATE would drop a fact the user is restating.
    """
    if not detect_frustration(query):
        return None

    fact = extract_forgotten_fact(query)

    if confidence == RetrievalConfidence.HIGH:
        return {
            "action": "reassure",
            "message": "I do have some information about this. Let me check.",
            "pin_fact": None,
        }

    if fact:
        return {
            "action": "recover_and_pin",
            "message": "Sorry about that. I'm saving this now with high importance so I won't forget again.",
            "pin_fact": fact,
            "pin_importance": 9.0,
        }

    return {
        "action": "apologize_and_ask",
        "message": "I'm sorry, I don't seem to have that stored. Could you tell me again? I'll make sure it sticks this time.",
        "pin_fact": None,
    }
