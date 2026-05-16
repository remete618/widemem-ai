from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from widemem.core._time import as_utc
from widemem.core.types import MemorySearchResult, ScoringConfig, YMYLConfig
from widemem.scoring.decay import compute_recency_score
from widemem.scoring.importance import normalize_importance
from widemem.scoring.topics import get_topic_boost
from widemem.scoring.ymyl import classify_ymyl_detailed


def score_candidate(
    result: MemorySearchResult,
    config: ScoringConfig,
    now: datetime,
    time_after: Optional[datetime] = None,
    time_before: Optional[datetime] = None,
    topic_weights: Optional[Dict[str, float]] = None,
    ymyl_config: Optional[YMYLConfig] = None,
    temporal_boost_window: Optional[tuple] = None,
    temporal_boost_strength: float = 0.10,
) -> MemorySearchResult | None:
    """Score one candidate in-place and return it, or None when hard-filtered."""
    created_at = as_utc(result.memory.created_at)

    if time_after and created_at < time_after:
        return None
    if time_before and created_at > time_before:
        return None

    ymyl_config = ymyl_config or YMYLConfig()
    content = result.memory.content

    # Check stored YMYL classification first (from LLM extraction), fall back to regex
    mem_is_ymyl_strong = False
    if ymyl_config.enabled:
        if result.memory.ymyl_category is not None:
            mem_is_ymyl_strong = True
        else:
            ymyl_result = classify_ymyl_detailed(content, ymyl_config)
            mem_is_ymyl_strong = ymyl_result is not None and ymyl_result.is_strong

    if mem_is_ymyl_strong and ymyl_config.decay_immune:
        recency = 1.0
    else:
        recency = compute_recency_score(
            created_at=created_at,
            now=now,
            decay_function=config.decay_function,
            decay_rate=config.decay_rate,
        )

    importance = normalize_importance(result.memory.importance)

    final = (
        config.similarity_weight * result.similarity_score
        + config.importance_weight * importance
        + config.recency_weight * recency
    )

    boost_after: Optional[datetime] = None
    boost_before: Optional[datetime] = None
    if temporal_boost_window is not None:
        raw_after, raw_before = temporal_boost_window
        boost_after = as_utc(raw_after) if raw_after is not None else None
        boost_before = as_utc(raw_before) if raw_before is not None else None

    # Soft temporal boost: nudge in-window memories up rather than
    # excluding out-of-window ones.
    if temporal_boost_window is not None and (boost_after or boost_before):
        in_window = True
        if boost_after and created_at < boost_after:
            in_window = False
        if boost_before and created_at > boost_before:
            in_window = False
        if in_window:
            final += temporal_boost_strength

    if topic_weights:
        boost = get_topic_boost(content, topic_weights)
        final *= boost

    result.temporal_score = recency
    result.importance_score = importance
    result.final_score = final
    return result


def score_and_rank(
    results: list[MemorySearchResult],
    config: ScoringConfig,
    now: Optional[datetime] = None,
    time_after: Optional[datetime] = None,
    time_before: Optional[datetime] = None,
    topic_weights: Optional[Dict[str, float]] = None,
    ymyl_config: Optional[YMYLConfig] = None,
    similarity_first: bool = False,
    similarity_boost: float = 0.15,
    temporal_boost_window: Optional[tuple] = None,
    temporal_boost_strength: float = 0.10,
) -> list[MemorySearchResult]:
    """Score, rank, and (optionally) temporally bias a candidate pool.

    Filter vs boost semantics:
      time_after / time_before are HARD filters. Memories outside the
      window are excluded. Use when the caller is certain the answer
      lives in that range (explicit user intent).
      temporal_boost_window is a SOFT boost. In-window memories get
      ``temporal_boost_strength`` added to their final_score; out-of-
      window memories are NOT excluded. Use when the time range comes
      from heuristic parsing of a query (uncertain intent), so that a
      bad parse cannot wipe out the candidate pool.
    """
    now = as_utc(now) if now is not None else datetime.now(timezone.utc)
    time_after = as_utc(time_after) if time_after is not None else None
    time_before = as_utc(time_before) if time_before is not None else None
    ymyl_config = ymyl_config or YMYLConfig()

    scored = []
    for result in results:
        scored_result = score_candidate(
            result=result,
            config=config,
            now=now,
            time_after=time_after,
            time_before=time_before,
            topic_weights=topic_weights,
            ymyl_config=ymyl_config,
            temporal_boost_window=temporal_boost_window,
            temporal_boost_strength=temporal_boost_strength,
        )
        if scored_result is not None:
            scored.append(scored_result)

    scored.sort(key=lambda r: r.final_score, reverse=True)

    # Two-pass: ensure top similarity results aren't buried by importance scoring
    if similarity_first and len(scored) > 5:
        by_sim = sorted(scored, key=lambda r: r.similarity_score, reverse=True)
        top_sim_ids = {id(r) for r in by_sim[:5]}
        top_final = scored[0].final_score if scored else 1.0
        for r in scored:
            if id(r) in top_sim_ids:
                # Additive boost — ensures top-similarity results rise
                # Boost strength varies by retrieval mode (fast=0.10, balanced=0.15, deep=0.20)
                r.final_score += top_final * similarity_boost
        scored.sort(key=lambda r: r.final_score, reverse=True)

    return scored
