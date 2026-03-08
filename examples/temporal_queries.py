"""Temporal retrieval: time-weighted scoring and time range filters."""

from datetime import datetime, timedelta

from widemem import WideMemory, MemoryConfig
from widemem.core.types import DecayFunction, ScoringConfig

# Configure with exponential decay — recent memories score higher
config = MemoryConfig(
    scoring=ScoringConfig(
        decay_function=DecayFunction.EXPONENTIAL,
        decay_rate=0.02,  # Faster decay
        similarity_weight=0.4,
        importance_weight=0.3,
        recency_weight=0.3,  # Higher recency weight
    ),
)
mem = WideMemory(config=config)

mem.add("Started a new job at Google", user_id="alice")
mem.add("Learning Rust programming", user_id="alice")

# Search with time filter — only memories from last 7 days
results = mem.search(
    "what is alice doing",
    user_id="alice",
    time_after=datetime.utcnow() - timedelta(days=7),
)
for r in results:
    print(f"  [{r.final_score:.3f}] sim={r.similarity_score:.2f} "
          f"imp={r.importance_score:.2f} rec={r.temporal_score:.2f} "
          f"| {r.memory.content}")
