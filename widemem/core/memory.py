from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from widemem.conflict.batch_resolver import BatchConflictResolver
from widemem.core.pipeline import AddResult, MemoryPipeline
from widemem.core.types import (
    RETRIEVAL_MODE_PRESETS,
    HistoryEntry,
    Memory,
    MemoryConfig,
    MemorySearchResult,
    MemoryTier,
    RetrievalMode,
    ScoringConfig,
    SearchResult,
)
from widemem.extraction.collector import ExtractionCollector
from widemem.extraction.llm_extractor import LLMExtractor
from widemem.hierarchy.manager import HierarchyManager
from widemem.hierarchy.query_router import classify_query, route_results
from widemem.hierarchy.summarizer import MemorySummarizer
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.providers.embeddings.openai import OpenAIEmbedder
from widemem.providers.llm.base import BaseLLM
from widemem.providers.llm.openai import OpenAILLM
from widemem.retrieval.active import ActiveRetrieval, Clarification
from widemem.retrieval.temporal import score_and_rank
from widemem.retrieval.uncertainty import assess_confidence
from widemem.storage.history import HistoryStore
from widemem.storage.vector.base import BaseVectorStore
from widemem.storage.vector.faiss_store import FAISSVectorStore


class WideMemory:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        llm: Optional[BaseLLM] = None,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        self.config = config or MemoryConfig()

        self.llm = llm or self._create_llm()
        self.embedder = embedder or self._create_embedder()
        self.vector_store = vector_store or self._create_vector_store()
        self._history_store = HistoryStore(self.config.history_db_path)

        self._collector = (
            ExtractionCollector(self.config.extractions_db_path)
            if self.config.collect_extractions else None
        )
        collector = self._collector
        extractor = LLMExtractor(
            self.llm,
            collector=collector,
            ymyl_config=self.config.ymyl,
            custom_topics=self.config.topics.custom_topics,
        )
        resolver = BatchConflictResolver(self.llm)
        active = (
            ActiveRetrieval(self.llm, similarity_threshold=self.config.active_retrieval_threshold)
            if self.config.enable_active_retrieval else None
        )

        ymyl_active = None
        if not active and self.config.ymyl.enabled and self.config.ymyl.force_active_retrieval:
            ymyl_active = ActiveRetrieval(
                self.llm, similarity_threshold=self.config.active_retrieval_threshold
            )

        self.pipeline = MemoryPipeline(
            extractor=extractor,
            resolver=resolver,
            embedder=self.embedder,
            vector_store=self.vector_store,
            history=self._history_store,
            active_retrieval=active,
            ymyl_active_retrieval=ymyl_active,
            ymyl_config=self.config.ymyl,
        )

        self._hierarchy = HierarchyManager(
            summarizer=MemorySummarizer(self.llm),
            embedder=self.embedder,
            vector_store=self.vector_store,
            history=self._history_store,
        )

    def close(self) -> None:
        self._history_store.close()
        if self._collector:
            self._collector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def add(
        self,
        text: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        on_clarification: Optional[Callable[[List[Clarification]], Optional[List[str]]]] = None,
    ) -> AddResult:
        return self.pipeline.process(
            text=text,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            on_clarification=on_clarification,
        )

    def add_batch(
        self,
        texts: List[str],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> List[AddResult]:
        results = []
        for text in texts:
            result = self.add(text, user_id=user_id, agent_id=agent_id, run_id=run_id)
            results.append(result)
        return results

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: Optional[int] = None,
        time_after: Optional[datetime] = None,
        time_before: Optional[datetime] = None,
        tier: Optional[MemoryTier] = None,
        mode: Optional[RetrievalMode] = None,
    ) -> SearchResult:
        # Resolve retrieval preset: per-query mode > config mode > defaults
        preset = self.config.get_retrieval_preset()
        if mode is not None:
            preset = dict(RETRIEVAL_MODE_PRESETS[mode])
        effective_top_k = min(top_k or preset["top_k"], 1000)

        embedding = self.embedder.embed(query)

        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if tier:
            filters["tier"] = tier.value

        scoring_config, sim_first = self._adapt_scoring(query, self.config.scoring)
        fetch_multiplier = preset["fetch_k_multiplier"] if sim_first else 3
        fetch_k = effective_top_k * fetch_multiplier
        raw_results = self.vector_store.search(
            vector=embedding,
            top_k=fetch_k,
            filters=filters or None,
        )

        now = datetime.utcnow()
        ttl_cutoff = now - timedelta(days=self.config.ttl_days) if self.config.ttl_days else None

        search_results = []
        for id, score, metadata in raw_results:
            created_at = (
                datetime.fromisoformat(metadata["created_at"])
                if "created_at" in metadata else now
            )
            if ttl_cutoff and created_at < ttl_cutoff:
                continue
            search_results.append(MemorySearchResult(
                memory=Memory(
                    id=id,
                    content=metadata.get("content", ""),
                    user_id=metadata.get("user_id"),
                    agent_id=metadata.get("agent_id"),
                    importance=metadata.get("importance", 5.0),
                    tier=MemoryTier(metadata.get("tier", "fact")),
                    created_at=created_at,
                    updated_at=(
                        datetime.fromisoformat(metadata["updated_at"])
                        if "updated_at" in metadata else now
                    ),
                ),
                similarity_score=score,
            ))

        ranked = score_and_rank(
            results=search_results,
            config=scoring_config,
            time_after=time_after,
            time_before=time_before,
            topic_weights=self.config.topics.weights or None,
            ymyl_config=self.config.ymyl if self.config.ymyl.enabled else None,
            similarity_first=sim_first,
            similarity_boost=preset.get("similarity_boost", 0.15),
        )

        use_hierarchy = preset.get("enable_hierarchy", self.config.enable_hierarchy)
        if use_hierarchy and tier is None:
            preferred = classify_query(query)
            ranked = route_results(ranked, preferred)

        final = ranked[:effective_top_k]
        confidence = assess_confidence(final)
        return SearchResult(results=final, confidence=confidence)

    def summarize(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        force: bool = False,
    ) -> List[Memory]:
        return self._hierarchy.maybe_summarize(
            user_id=user_id,
            agent_id=agent_id,
            force=force,
        )

    def pin(
        self,
        text: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        importance: float = 9.0,
    ) -> AddResult:
        """Store a memory with elevated importance. Use when the user explicitly
        asks to remember something, corrects a forgotten fact, or repeats
        information that should not be forgotten."""
        result = self.pipeline.process(text=text, user_id=user_id, agent_id=agent_id)

        for memory in result.memories:
            if memory.importance < importance:
                memory.importance = importance
                embedding = self.embedder.embed(memory.content)
                self.vector_store.update(
                    id=memory.id,
                    vector=embedding,
                    metadata=self.pipeline._memory_to_metadata(memory),
                )

        return result

    def get(self, memory_id: str) -> Optional[Memory]:
        result = self.vector_store.get(memory_id)
        if result is None:
            return None
        _, metadata = result
        return Memory(
            id=memory_id,
            content=metadata.get("content", ""),
            user_id=metadata.get("user_id"),
            agent_id=metadata.get("agent_id"),
            importance=metadata.get("importance", 5.0),
            tier=MemoryTier(metadata.get("tier", "fact")),
        )

    def delete(self, memory_id: str) -> None:
        self.vector_store.delete(memory_id)

    def get_history(self, memory_id: str) -> List[HistoryEntry]:
        return self._history_store.get_history(memory_id)

    def count(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tier: Optional[MemoryTier] = None,
    ) -> int:
        filters: Dict[str, str] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if tier:
            filters["tier"] = tier.value
        items = self.vector_store.list_all(filters=filters or None, max_results=100000)
        return len(items)

    def export_json(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        filters: Dict[str, str] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        items = self.vector_store.list_all(filters=filters or None, max_results=100000)
        memories = []
        for id, metadata in items:
            memories.append({"id": id, **metadata})
        return json.dumps({"memories": memories, "count": len(memories)}, indent=2, default=str)

    def import_json(self, data: str) -> int:
        parsed = json.loads(data)
        imported = 0
        for item in parsed.get("memories", []):
            memory_id = item.get("id")
            content = item.get("content", "")
            if not content or len(content) > 50000:
                continue
            existing = self.vector_store.get(memory_id) if memory_id else None
            if existing:
                continue
            raw_importance = item.get("importance", 5.0)
            try:
                importance = max(0.0, min(10.0, float(raw_importance)))
            except (TypeError, ValueError):
                importance = 5.0
            raw_tier = item.get("tier", "fact")
            try:
                tier = MemoryTier(raw_tier)
            except ValueError:
                tier = MemoryTier.FACT
            embedding = self.embedder.embed(content)
            memory = Memory(
                id=memory_id or Memory().id,
                content=content,
                user_id=item.get("user_id"),
                agent_id=item.get("agent_id"),
                importance=importance,
                tier=tier,
                content_hash=item.get("content_hash", ""),
            )
            metadata = {
                "content": memory.content,
                "user_id": memory.user_id,
                "agent_id": memory.agent_id,
                "run_id": memory.run_id,
                "tier": memory.tier.value,
                "importance": memory.importance,
                "content_hash": memory.content_hash,
                "created_at": item.get("created_at", memory.created_at.isoformat()),
                "updated_at": item.get("updated_at", memory.updated_at.isoformat()),
            }
            self.vector_store.insert(id=memory.id, vector=embedding, metadata=metadata)
            imported += 1
        return imported

    @staticmethod
    def _adapt_scoring(query: str, default: ScoringConfig) -> tuple:
        """Adapt scoring weights based on query type. Returns (ScoringConfig, similarity_first)."""
        q = query.lower().strip()

        # Temporal queries — boost recency, reduce importance
        temporal_signals = ("when ", "what time", "what date", "how long ago",
                           "last time", "recently", "before the", "after the",
                           "how recent", "what year", "what month")
        if any(q.startswith(s) or s in q for s in temporal_signals):
            return ScoringConfig(
                decay_function=default.decay_function,
                decay_rate=default.decay_rate,
                similarity_weight=0.40,
                importance_weight=0.10,
                recency_weight=0.50,
            ), False

        # Simple factual queries — boost similarity, reduce importance, enable similarity_first
        factual_starts = ("where ", "who ", "what is ", "what was ", "what does ",
                         "what did ", "what are ", "what were ", "what do ",
                         "how old ", "how much ", "how many ",
                         "which ", "name ", "is ", "was ",
                         "does ", "did ", "has ", "have ")
        is_short_what = q.startswith("what ") and len(q.split()) <= 10
        multi_hop_signals = ("relationship between", "how does", "compare", "contrast",
                            "connection between", "relate to", "in common")
        is_multi_hop = any(s in q for s in multi_hop_signals)
        if (any(q.startswith(s) for s in factual_starts) or is_short_what) and not is_multi_hop:
            return ScoringConfig(
                decay_function=default.decay_function,
                decay_rate=default.decay_rate,
                similarity_weight=0.75,
                importance_weight=0.10,
                recency_weight=0.15,
            ), True  # similarity_first for factual queries

        # Multi-hop / broad / default — keep configured weights, no similarity_first
        return default, False

    def _create_llm(self) -> BaseLLM:
        provider = self.config.llm.provider
        if provider == "openai":
            return OpenAILLM(self.config.llm)
        if provider == "anthropic":
            from widemem.providers.llm.anthropic import AnthropicLLM
            return AnthropicLLM(self.config.llm)
        if provider == "ollama":
            from widemem.providers.llm.ollama import OllamaLLM
            return OllamaLLM(self.config.llm)
        raise ValueError(
            f"Unknown LLM provider: {provider}. Supported: openai, anthropic, ollama"
        )

    def _create_embedder(self) -> BaseEmbedder:
        provider = self.config.embedding.provider
        if provider == "openai":
            return OpenAIEmbedder(self.config.embedding)
        if provider == "sentence-transformers":
            from widemem.providers.embeddings.sentence_transformers import (
                SentenceTransformerEmbedder,
            )
            return SentenceTransformerEmbedder(self.config.embedding)
        raise ValueError(
            f"Unknown embedding provider: {provider}. Supported: openai, sentence-transformers"
        )

    def _create_vector_store(self) -> BaseVectorStore:
        provider = self.config.vector_store.provider
        if provider == "faiss":
            return FAISSVectorStore(
                self.config.vector_store,
                dimensions=self.config.embedding.dimensions,
            )
        if provider == "qdrant":
            from widemem.storage.vector.qdrant_store import QdrantVectorStore
            return QdrantVectorStore(
                self.config.vector_store,
                dimensions=self.config.embedding.dimensions,
            )
        raise ValueError(
            f"Unknown vector store provider: {provider}. Supported: faiss, qdrant"
        )
