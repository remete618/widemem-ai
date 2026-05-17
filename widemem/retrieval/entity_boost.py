from __future__ import annotations

from collections import Counter
from typing import List

from widemem.core.types import MemorySearchResult

# Entity-aware additive re-rank. Mem0's current OSS open-domain
# mechanism, minus the graph: boost candidates whose entities overlap
# the query's, attenuated by how common the entity is in the candidate
# pool so a frequently-mentioned name does not flood results.
#
# Discipline (the part that protects multi-hop): this only reorders the
# already-retrieved, already-scored pool. It never adds candidates and
# never changes how many are returned, so it cannot resurrect a
# sub-similarity fact into existence and it is token-neutral. The
# empirical multi-hop non-regression guard is the mini-LoCoMo gate.


def apply_entity_boost(
    results: List[MemorySearchResult],
    query_entities: List[str],
    weight: float,
    attenuation: float,
) -> List[MemorySearchResult]:
    """Additively boost final_score for candidates sharing entities with
    the query, then re-sort. No-op (results untouched, order unchanged)
    when weight <= 0 or there are no query entities."""
    if weight <= 0 or not query_entities or not results:
        return results

    qset = set(query_entities)

    freq: Counter[str] = Counter()
    for r in results:
        for e in set(r.memory.entities):
            if e in qset:
                freq[e] += 1

    if not freq:
        return results

    denom = len(qset)
    for r in results:
        matched = qset.intersection(r.memory.entities)
        if not matched:
            continue
        s = 0.0
        for e in matched:
            n = freq.get(e, 1)
            s += 1.0 / (1.0 + attenuation * (n - 1) ** 2)
        r.final_score += weight * (s / denom)

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results
