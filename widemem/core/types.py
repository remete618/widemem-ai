from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class MemoryTier(str, Enum):
    FACT = "fact"
    SUMMARY = "summary"
    THEME = "theme"


class MemoryAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NONE = "none"


class DecayFunction(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"
    NONE = "none"


class Fact(BaseModel):
    content: str
    importance: float = Field(default=5.0, ge=0.0, le=10.0)


class ActionItem(BaseModel):
    action: MemoryAction
    fact: str
    target_id: Optional[str] = None
    importance: float = Field(default=5.0, ge=0.0, le=10.0)


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    tier: MemoryTier = MemoryTier.FACT
    importance: float = Field(default=5.0, ge=0.0, le=10.0)
    content_hash: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MemorySearchResult(BaseModel):
    memory: Memory
    similarity_score: float = 0.0
    temporal_score: float = 1.0
    importance_score: float = 1.0
    final_score: float = 0.0


class HistoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str
    action: MemoryAction
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2000


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    dimensions: int = 1536


class VectorStoreConfig(BaseModel):
    provider: str = "faiss"
    path: Optional[str] = None


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


class MemoryConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    ymyl: YMYLConfig = Field(default_factory=YMYLConfig)
    topics: TopicConfig = Field(default_factory=TopicConfig)
    history_db_path: str = "~/.widemem/history.db"
    enable_hierarchy: bool = False
    enable_active_retrieval: bool = False
    active_retrieval_threshold: float = 0.6
    collect_extractions: bool = False
    extractions_db_path: str = "~/.widemem/extractions.db"
    ttl_days: Optional[int] = None
