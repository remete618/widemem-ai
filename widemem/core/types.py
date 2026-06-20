from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, SecretStr


class MemoryTier(str, Enum):
    FACT = "fact"
    SUMMARY = "summary"
    THEME = "theme"


class MemoryAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NONE = "none"


class RetrievalMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


class RetrievalConfidence(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NONE = "none"


class UncertaintyMode(str, Enum):
    STRICT = "strict"
    HELPFUL = "helpful"
    CREATIVE = "creative"


class DecayFunction(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"
    NONE = "none"


class Fact(BaseModel):
    content: str
    importance: float = Field(default=5.0, ge=0.0, le=10.0)
    ymyl_category: Optional[str] = None


class ActionItem(BaseModel):
    action: MemoryAction
    fact: str
    target_id: Optional[str] = None
    importance: float = Field(default=5.0, ge=0.0, le=10.0)
    ymyl_category: Optional[str] = None


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    tier: MemoryTier = MemoryTier.FACT
    importance: float = Field(default=5.0, ge=0.0, le=10.0)
    content_hash: str = ""
    ymyl_category: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_time: Optional[datetime] = None
    entities: list[str] = Field(default_factory=list)


class MemorySearchResult(BaseModel):
    memory: Memory
    similarity_score: float = 0.0
    temporal_score: float = 1.0
    importance_score: float = 1.0
    final_score: float = 0.0


class SearchResult:
    """Wraps search results with confidence metadata.
    Behaves like a list for backward compatibility — existing code that
    iterates, indexes, or checks len() works unchanged."""

    __slots__ = ("results", "confidence", "has_relevant")

    def __init__(
        self,
        results: list,
        confidence: RetrievalConfidence = RetrievalConfidence.HIGH,
    ) -> None:
        self.results = results
        self.confidence = confidence
        self.has_relevant = confidence != RetrievalConfidence.NONE

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        return self.results[idx]

    def __bool__(self):
        return len(self.results) > 0


class ExplainedMemory(BaseModel):
    """A retrieved memory with its 'why matched' score breakdown, for the
    explain=True trust path."""
    content: str
    final_score: float
    similarity: float
    importance: float
    recency: float
    ymyl_category: Optional[str] = None


class RetrievalExplanation(BaseModel):
    """Trust verdict returned by search(explain=True). Tells an agent not just
    what was retrieved but whether it is safe to use in an answer."""
    answerable: bool
    confidence: float
    confidence_level: str
    requires_review: bool
    reason: str
    memories: list[ExplainedMemory] = Field(default_factory=list)


class HistoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str
    action: MemoryAction
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[SecretStr] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2000


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: Optional[SecretStr] = None
    base_url: Optional[str] = None
    dimensions: int = 1536


class VectorStoreConfig(BaseModel):
    provider: str = "faiss"
    path: Optional[str] = None
    url: Optional[str] = None
    """Connection URL for network-backed stores (pgvector, Qdrant Cloud).
    For pgvector: postgresql://user:pass@host:port/dbname?sslmode=require.
    Honored only by backends that accept a URL; ignored otherwise."""
    table_name: str = "widemem_vectors"
    """Table name for pgvector. Ignored by other backends."""


class ScoringConfig(BaseModel):
    decay_function: DecayFunction = DecayFunction.EXPONENTIAL
    decay_rate: float = 0.01
    similarity_weight: float = 0.5
    importance_weight: float = 0.3
    recency_weight: float = 0.2


YMYL_CATEGORIES = [
    "health", "medical", "financial", "legal",
    "safety", "insurance", "tax", "pharmaceutical",
]


class YMYLConfig(BaseModel):
    enabled: bool = False
    categories: list = Field(default_factory=lambda: list(YMYL_CATEGORIES))
    min_importance: float = 8.0
    decay_immune: bool = True
    force_active_retrieval: bool = True


class TopicConfig(BaseModel):
    weights: Dict[str, float] = Field(default_factory=dict)
    custom_topics: list = Field(default_factory=list)


RETRIEVAL_MODE_PRESETS = {
    RetrievalMode.FAST: {"top_k": 10, "fetch_k_multiplier": 3, "similarity_boost": 0.10, "enable_hierarchy": False},
    RetrievalMode.BALANCED: {"top_k": 25, "fetch_k_multiplier": 4, "similarity_boost": 0.15, "enable_hierarchy": True},
    RetrievalMode.DEEP: {"top_k": 50, "fetch_k_multiplier": 5, "similarity_boost": 0.20, "enable_hierarchy": True},
}


class MemoryConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    ymyl: YMYLConfig = Field(default_factory=YMYLConfig)
    topics: TopicConfig = Field(default_factory=TopicConfig)
    history_db_path: str = "~/.widemem/history.db"
    retrieval_mode: RetrievalMode = RetrievalMode.BALANCED
    uncertainty_mode: UncertaintyMode = UncertaintyMode.HELPFUL
    enable_hierarchy: bool = False
    enable_active_retrieval: bool = False
    active_retrieval_threshold: float = 0.6
    collect_extractions: bool = False
    extractions_db_path: str = "~/.widemem/extractions.db"
    ttl_days: Optional[int] = None
    parse_temporal_hints: bool = False
    """Auto-parse temporal hints from queries into a soft recency boost.

    When True, queries like "What happened in July 2023?" or "last month"
    are parsed into a time window that is applied as a SOFT BOOST in
    ranking: in-window memories are nudged up, out-of-window memories are
    NOT excluded. This is deliberate; a wrong heuristic parse cannot wipe
    out the candidate pool. For a HARD filter that excludes out-of-window
    memories, pass time_after/time_before explicitly to search() (caller
    intent), which always wins over parsed hints. Off by default for
    backwards compatibility; flip on per-instance via
    MemoryConfig(parse_temporal_hints=True). See widemem/retrieval/temporal.py
    for the boost-vs-filter semantics.
    """
    enable_hybrid_search: bool = False
    """Blend BM25 keyword scores into the vector similarity signal.

    When True, the vector-search candidate pool is reranked once via BM25
    over the candidates' content. The blended score (50/50 by default)
    replaces each candidate's similarity_score before the existing scoring
    pipeline (importance + recency + YMYL boosts) runs. Catches exact
    keyword matches that pure cosine similarity misses (names, dates,
    numeric identifiers). Off by default for backwards compatibility.
    Requires the [bm25] extra: pip install widemem-ai[bm25].
    """
    hybrid_bm25_weight: float = 0.5
    """Proportion of the blended similarity_score taken from the BM25 side
    when enable_hybrid_search is True. 0.0 disables BM25 (vector-only),
    1.0 disables vector (pure keyword). Range [0, 1], default 0.5."""
    enable_entity_index: bool = False
    """Extract and store lightweight entity tags on each memory at write
    time (zero-dependency, no LLM). Off by default for backwards
    compatibility and behavior parity: when False, nothing changes. When
    True, memories carry an `entities` list usable by entity-aware
    retrieval. Existing stores are populated without re-ingestion via
    WideMemory.backfill_entities()."""
    entity_boost_weight: float = 0.0
    """Weight of the entity-overlap additive re-rank at search time.
    Requires enable_entity_index. Default 0.0 is a strict no-op: the
    feature is fully behavior-neutral until this is set > 0 (typical
    starting point ~0.5). The boost only reorders the already-retrieved
    pool; it never adds candidates and never changes how many are
    returned, so it is token-neutral."""
    entity_boost_attenuation: float = 0.001
    """Attenuation k in 1 / (1 + k * (n - 1)^2), where n is how many
    pooled candidates share a query entity. Damps very common entities
    so a frequently-mentioned name does not dominate. Only active when
    entity_boost_weight > 0."""

    def get_retrieval_preset(self) -> dict:
        """Get the retrieval preset for the configured mode."""
        preset = dict(RETRIEVAL_MODE_PRESETS[self.retrieval_mode])
        if self.enable_hierarchy:
            preset["enable_hierarchy"] = True
        return preset
