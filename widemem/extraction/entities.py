from __future__ import annotations

import re

# Zero-dependency, deterministic entity extraction. No spaCy, no LLM,
# no network. Heuristic by design: capitalized proper-noun spans,
# all-caps acronyms, and double-quoted spans. This is the lean baseline
# the project ethos calls for; an optional spaCy path can be added later
# as an extra if it measurably helps. Used as a retrieval signal, not a
# correctness boundary, so over- or under-capture is tolerable.

_PROPER = re.compile(r"\b[A-Z][a-z]*(?:['\-][A-Za-z]+)*(?:\s+[A-Z][a-z]*(?:['\-][A-Za-z]+)*)*\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_QUOTED = re.compile(r"\"([^\"]{2,64})\"")

# Ultra-common sentence-leading capitalized words that are not entities.
_STOP = {
    "the", "a", "an", "i", "it", "this", "that", "these", "those", "he",
    "she", "they", "we", "you", "but", "and", "if", "or", "so", "then",
    "when", "what", "where", "why", "how", "who", "which", "my", "your",
    "his", "her", "their", "our", "its", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "at", "for", "with", "as", "by", "from",
}

_MAX_ENTITIES = 24


def extract_entities(text: str | None) -> list[str]:
    """Return a deterministic, deduped, lowercased list of candidate
    entities from text. Empty list on falsy or entity-free input. Never
    raises on arbitrary input."""
    if not text:
        return []

    found: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        norm = re.sub(r"\s+", " ", raw).strip().lower()
        if len(norm) < 2 or norm in _STOP or norm.isdigit():
            return
        if norm in seen:
            return
        seen.add(norm)
        found.append(norm)

    try:
        for m in _QUOTED.finditer(text):
            _add(m.group(1))
        for m in _PROPER.finditer(text):
            _add(m.group(0))
        for m in _ACRONYM.finditer(text):
            _add(m.group(0))
    except (TypeError, re.error):
        return []

    return found[:_MAX_ENTITIES]
