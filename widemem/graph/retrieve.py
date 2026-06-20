from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from widemem.core.types import Memory, MemorySearchResult, MemoryTier
from widemem.extraction.entities import extract_entities
from widemem.graph.store import GraphStore


def _seeds_from_query(query: str) -> List[str]:
    """Lowercased, possessive-stripped entity anchors from the query."""
    seeds = []
    for e in extract_entities(query):
        s = e.strip().lower()
        if s.endswith("'s"):
            s = s[:-2]
        s = s.strip()
        if s:
            seeds.append(s)
    return seeds


def _parse_ts(value, fallback: datetime) -> datetime:
    if not value:
        return fallback
    try:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return fallback


def _result_from_metadata(mid: str, metadata: dict, now: datetime) -> MemorySearchResult:
    return MemorySearchResult(
        memory=Memory(
            id=mid,
            content=metadata.get("content", ""),
            user_id=metadata.get("user_id"),
            agent_id=metadata.get("agent_id"),
            importance=metadata.get("importance", 5.0),
            tier=MemoryTier(metadata.get("tier", "fact")),
            ymyl_category=metadata.get("ymyl_category"),
            created_at=_parse_ts(metadata.get("created_at"), now),
            updated_at=_parse_ts(metadata.get("updated_at"), now),
            entities=metadata.get("entities") or [],
        ),
        similarity_score=0.0,
    )


def augment_with_graph(
    ranked: List[MemorySearchResult],
    query: str,
    graph_store: GraphStore,
    vector_store,
    user_id: Optional[str],
    now: datetime,
    hops: int,
    max_nodes: int,
    boost_weight: float,
    max_inject: int,
) -> List[MemorySearchResult]:
    """Surface relationally-connected memories that pure vector similarity
    missed. Boosts in-pool memories the graph connects to the query's entities,
    and injects a bounded number of connected memories that never made the
    similarity pool. Returns a re-sorted list.

    Pure-numeric/date nodes (e.g. "2019") are not expanded THROUGH — they are
    leaf objects, so a shared year cannot bridge two unrelated people."""
    seeds = _seeds_from_query(query)
    if not seeds:
        return ranked

    mem_ids, _dist = graph_store.expand(
        seeds, user_id=user_id, hops=hops, max_nodes=max_nodes
    )
    if not mem_ids:
        return ranked

    in_pool = {r.memory.id for r in ranked}
    for r in ranked:
        if r.memory.id in mem_ids:
            r.final_score += boost_weight

    missing = [m for m in mem_ids if m not in in_pool]
    injected = 0
    for mid in missing:
        if injected >= max_inject:
            break
        got = vector_store.get(mid)
        if not got:
            continue
        res = _result_from_metadata(mid, got[1], now)
        if user_id and res.memory.user_id and res.memory.user_id != user_id:
            continue
        res.final_score = boost_weight
        ranked.append(res)
        injected += 1

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked
