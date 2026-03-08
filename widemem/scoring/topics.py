from __future__ import annotations

from typing import Dict, Optional


def get_topic_boost(
    content: str,
    topic_weights: Dict[str, float],
) -> float:
    if not topic_weights:
        return 1.0

    content_lower = content.lower()
    best_boost = 1.0

    for topic, weight in topic_weights.items():
        if topic.lower() in content_lower:
            best_boost = max(best_boost, weight)

    return best_boost


def get_topic_label(
    content: str,
    topic_weights: Dict[str, float],
) -> Optional[str]:
    if not topic_weights:
        return None

    content_lower = content.lower()
    for topic in topic_weights:
        if topic.lower() in content_lower:
            return topic

    return None
