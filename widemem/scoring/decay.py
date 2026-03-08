from __future__ import annotations

import math
from datetime import datetime

from widemem.core.types import DecayFunction


def compute_recency_score(
    created_at: datetime,
    now: datetime,
    decay_function: DecayFunction = DecayFunction.EXPONENTIAL,
    decay_rate: float = 0.01,
) -> float:
    age_days = max((now - created_at).total_seconds() / 86400, 0.0)

    if decay_function == DecayFunction.NONE:
        return 1.0

    if decay_function == DecayFunction.EXPONENTIAL:
        return math.exp(-decay_rate * age_days)

    if decay_function == DecayFunction.LINEAR:
        return max(1.0 - decay_rate * age_days, 0.0)

    if decay_function == DecayFunction.STEP:
        if age_days < 7:
            return 1.0
        if age_days < 30:
            return 0.7
        if age_days < 90:
            return 0.4
        return 0.1

    return 1.0
