from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from widemem.core.types import MemorySearchResult, ScoringConfig, YMYLConfig
from widemem.scoring.decay import compute_recency_score
from widemem.scoring.importance import normalize_importance
from widemem.scoring.topics import get_topic_boost
from widemem.scoring.ymyl import classify_ymyl_detailed


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
) -> list[MemorySearchResult]:
    now = now or datetime.utcnow()
    ymyl_config = ymyl_config or YMYLConfig()

    scored = []
    for result in results:
        created_at = result.memory.created_at

        if time_after and created_at < time_after:
            continue
        if time_before and created_at > time_before:
            continue

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

        if topic_weights:
            boost = get_topic_boost(content, topic_weights)
            final *= boost

        result.temporal_score = recency
        result.importance_score = importance
        result.final_score = final
        scored.append(result)

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
