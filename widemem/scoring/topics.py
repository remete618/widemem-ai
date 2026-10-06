from __future__ import annotations

import math
from typing import Dict, Optional


def validate_topic_weight(topic: str, weight: float) -> None:
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(
            f"topic weight for {topic!r} must be a finite number above 0, got {weight!r}"
        )


def get_topic_boost(
    content: str,
    topic_weights: Dict[str, float],
) -> float:
    """Multiplier for ``content``: the strongest matching boost (weight above 1.0)
    times the strongest matching suppression (weight below 1.0).

    Boosts do not compound, so overlapping topics such as "python" and "python
    programming" cannot stack into a runaway multiplier. A suppression still
    applies when a boost also matches, so a down-weighted topic stays down-weighted.
    """
    if not topic_weights:
        return 1.0

    content_lower = content.lower()
    best_boost = 1.0
    strongest_suppression = 1.0

    for topic, weight in topic_weights.items():
        validate_topic_weight(topic, weight)
        if topic.lower() in content_lower:
            best_boost = max(best_boost, weight)
            strongest_suppression = min(strongest_suppression, weight)

    return best_boost * strongest_suppression


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
