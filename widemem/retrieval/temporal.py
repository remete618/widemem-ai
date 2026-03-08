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
        ymyl_result = classify_ymyl_detailed(content, ymyl_config) if ymyl_config.enabled else None
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
    return scored
