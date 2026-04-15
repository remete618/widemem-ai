"""Uncertainty-aware retrieval: confidence assessment, frustration detection, and response guidance."""

from __future__ import annotations

import os
import re
from typing import Optional

from widemem.core.types import (
    MemorySearchResult,
    RetrievalConfidence,
    UncertaintyMode,
)

_DEFAULT_THRESHOLDS = {
    "high": 0.65,
    "moderate": 0.45,
    "low": 0.25,
}


def get_confidence_thresholds() -> dict[str, float]:
    return {
        "high": float(os.environ.get("WIDEMEM_CONFIDENCE_HIGH", _DEFAULT_THRESHOLDS["high"])),
        "moderate": float(os.environ.get("WIDEMEM_CONFIDENCE_MODERATE", _DEFAULT_THRESHOLDS["moderate"])),
        "low": float(os.environ.get("WIDEMEM_CONFIDENCE_LOW", _DEFAULT_THRESHOLDS["low"])),
    }

FRUSTRATION_SIGNALS = (
    "i told you", "i already said", "remember when i", "i mentioned",
    "you forgot", "you should know", "we talked about", "i said before",
    "don't you remember", "how could you forget", "i literally told",
    "we discussed", "you should remember", "i specifically said",
)


def assess_confidence(results: list[MemorySearchResult]) -> RetrievalConfidence:
    """Assess how confident we are that the search results are relevant."""
    if not results:
        return RetrievalConfidence.NONE

    top_sim = results[0].similarity_score
    thresholds = get_confidence_thresholds()

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
            "message": "I don't have this in my memory — I can take a guess based on what I do know, if you'd like.",
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
            "message": "I'm not sure about this, but I have some related memories — want me to piece something together?",
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
    """
    if not detect_frustration(query):
        return None

    fact = extract_forgotten_fact(query)

    if confidence in (RetrievalConfidence.HIGH, RetrievalConfidence.MODERATE):
        return {
            "action": "reassure",
            "message": "I do have some information about this — let me check.",
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
