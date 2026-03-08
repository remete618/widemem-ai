from __future__ import annotations


def normalize_importance(importance: float) -> float:
    """Normalize importance from 0-10 scale to 0-1 scale."""
    return max(min(importance, 10.0), 0.0) / 10.0
