from __future__ import annotations

from typing import List

from widemem.core.types import MemorySearchResult, MemoryTier

BROAD_KEYWORDS = {
    "tell me about", "describe", "who is", "what do you know about",
    "summary", "overview", "profile", "background", "everything about",
}

SPECIFIC_KEYWORDS = {
    "where", "when", "what is", "how old", "how much", "how many",
    "does", "did", "is", "was", "has", "have", "which", "name",
}


def classify_query(query: str) -> MemoryTier:
    q = query.lower().strip()

    for kw in BROAD_KEYWORDS:
        if kw in q:
            return MemoryTier.THEME

    for kw in SPECIFIC_KEYWORDS:
        if q.startswith(kw):
            return MemoryTier.FACT

    word_count = len(q.split())
    if word_count <= 4:
        return MemoryTier.FACT

    return MemoryTier.SUMMARY


def route_results(
    results: List[MemorySearchResult],
    preferred_tier: MemoryTier,
    min_results: int = 3,
) -> List[MemorySearchResult]:
    tier_results = [r for r in results if r.memory.tier == preferred_tier]

    if len(tier_results) >= min_results:
        return tier_results

    if preferred_tier == MemoryTier.THEME:
        fallback_order = [MemoryTier.SUMMARY, MemoryTier.FACT]
    elif preferred_tier == MemoryTier.SUMMARY:
        fallback_order = [MemoryTier.THEME, MemoryTier.FACT]
    else:
        fallback_order = [MemoryTier.SUMMARY, MemoryTier.THEME]

    combined = list(tier_results)
    for tier in fallback_order:
        if len(combined) >= min_results:
            break
        combined.extend(r for r in results if r.memory.tier == tier)

    return combined
