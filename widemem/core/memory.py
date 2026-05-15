from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from widemem.conflict.batch_resolver import BatchConflictResolver
from widemem.core._time import as_utc
from widemem.core.pipeline import AddResult, MemoryPipeline
from widemem.core.types import (
    RETRIEVAL_MODE_PRESETS,
    EmbeddingConfig,
    HistoryEntry,
    LLMConfig,
    Memory,
    MemoryConfig,
    MemorySearchResult,
    MemoryTier,
    RetrievalMode,
    ScoringConfig,
    SearchResult,
)
from widemem.extraction.collector import ExtractionCollector
from widemem.extraction.datetime_parse import parse_leading_datetime
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
from widemem.retrieval.temporal_parser import looks_temporal, parse_temporal_hints
from widemem.retrieval.uncertainty import assess_confidence
from widemem.storage.history import HistoryStore
from widemem.storage.vector.base import BaseVectorStore


def _parse_ts_opt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return as_utc(datetime.fromisoformat(value))
    except (ValueError, TypeError):
        return None


def _parse_ts(value: Optional[str], fallback: datetime) -> datetime:
    return _parse_ts_opt(value) or fallback


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

    MAX_TEXT_LENGTH = 50_000

    def add(
        self,
        text: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        on_clarification: Optional[Callable[[List[Clarification]], Optional[List[str]]]] = None,
    ) -> AddResult:
        if not text or not text.strip():
            return AddResult(memories=[])
        if len(text) > self.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text too long ({len(text)} chars). Maximum is {self.MAX_TEXT_LENGTH}."
            )
        event_time = as_utc(timestamp) if timestamp else parse_leading_datetime(text)
        return self.pipeline.process(
            text=text,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            event_time=event_time,
            on_clarification=on_clarification,
        )

    def add_batch(
        self,
        texts: List[str],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> List[AddResult]:
        results = []
        for text in texts:
            result = self.add(text, user_id=user_id, agent_id=agent_id, run_id=run_id, timestamp=timestamp)
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
        final = self._search_ranked(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            top_k=top_k,
            time_after=time_after,
            time_before=time_before,
            tier=tier,
            mode=mode,
        )
        confidence = assess_confidence(final)
        return SearchResult(results=final, confidence=confidence)

    async def search_stream(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: Optional[int] = None,
        time_after: Optional[datetime] = None,
        time_before: Optional[datetime] = None,
        tier: Optional[MemoryTier] = None,
        mode: Optional[RetrievalMode] = None,
    ) -> AsyncGenerator[MemorySearchResult, None]:
        """Yield search results one at a time.

        Example:
            >>> async for result in memory.search_stream("where does alice live", user_id="alice"):
            ...     print(result.memory.content)
        """
        final = self._search_ranked(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            top_k=top_k,
            time_after=time_after,
            time_before=time_before,
            tier=tier,
            mode=mode,
        )
        for result in final:
            yield result

    def _search_ranked(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        top_k: Optional[int] = None,
        time_after: Optional[datetime] = None,
        time_before: Optional[datetime] = None,
        tier: Optional[MemoryTier] = None,
        mode: Optional[RetrievalMode] = None,
    ) -> list[MemorySearchResult]:
        # Resolve retrieval preset: per-query mode > config mode > defaults
        preset = self.config.get_retrieval_preset()
        if mode is not None:
            preset = dict(RETRIEVAL_MODE_PRESETS[mode])
        effective_top_k = min(top_k or preset["top_k"], 1000)

        embedding = self.embedder.embed(query)

        # Auto-parse temporal hints when enabled and the caller did not pass
        # explicit time_after/time_before. The parsed window is used as a
        # SOFT BOOST in score_and_rank (not a hard filter), so a wrong parse
        # cannot wipe out the candidate pool. Explicit caller-set time_after
        # / time_before still filter, since the caller is asserting intent.
        # See widemem/retrieval/temporal.py for the boost semantics.
        temporal_boost_window: Optional[tuple] = None
        if (
            self.config.parse_temporal_hints
            and time_after is None
            and time_before is None
            and looks_temporal(query)
        ):
            parsed_after, parsed_before = parse_temporal_hints(query)
            if parsed_after is not None or parsed_before is not None:
                temporal_boost_window = (parsed_after, parsed_before)

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

        now = datetime.now(timezone.utc)
        ttl_cutoff = now - timedelta(days=self.config.ttl_days) if self.config.ttl_days else None

        search_results = []
        for id, score, metadata in raw_results:
            created_at = _parse_ts(metadata.get("created_at"), now)
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
                    ymyl_category=metadata.get("ymyl_category"),
                    created_at=created_at,
                    updated_at=_parse_ts(metadata.get("updated_at"), now),
                    event_time=_parse_ts_opt(metadata.get("event_time")),
                ),
                similarity_score=score,
            ))

        # Hybrid retrieval: blend BM25 keyword scores into similarity_score
        # within the candidate pool. Default off; opt-in via
        # MemoryConfig.enable_hybrid_search. The BM25 weight scales by
        # detected query type: disabled for multi-hop queries (where the
        # importance-weighted vector signal already wins), full for
        # factual queries (where keyword matching helps surface specific
        # terms), reduced for temporal and broad queries. See
        # WideMemory._adapt_bm25_weight for the mapping.
        if self.config.enable_hybrid_search and search_results:
            effective_bm25_weight = self._adapt_bm25_weight(
                query, self.config.hybrid_bm25_weight
            )
            if effective_bm25_weight > 0:
                from widemem.retrieval.hybrid import blend_hybrid_scores
                blend_hybrid_scores(
                    search_results,
                    query,
                    bm25_weight=effective_bm25_weight,
                )

        ranked = score_and_rank(
            results=search_results,
            config=scoring_config,
            time_after=time_after,
            time_before=time_before,
            topic_weights=self.config.topics.weights or None,
            ymyl_config=self.config.ymyl if self.config.ymyl.enabled else None,
            similarity_first=sim_first,
            similarity_boost=preset.get("similarity_boost", 0.15),
            temporal_boost_window=temporal_boost_window,
        )

        use_hierarchy = preset.get("enable_hierarchy", self.config.enable_hierarchy)
        if use_hierarchy and tier is None:
            preferred = classify_query(query)
            ranked = route_results(ranked, preferred)

        return ranked[:effective_top_k]

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
        if not text or not text.strip():
            return AddResult(memories=[])
        if len(text) > self.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text too long ({len(text)} chars). Maximum is {self.MAX_TEXT_LENGTH}."
            )
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
        kwargs: Dict[str, Any] = {
            "id": memory_id,
            "content": metadata.get("content", ""),
            "user_id": metadata.get("user_id"),
            "agent_id": metadata.get("agent_id"),
            "run_id": metadata.get("run_id"),
            "tier": MemoryTier(metadata.get("tier", "fact")),
            "importance": metadata.get("importance", 5.0),
            "content_hash": metadata.get("content_hash", ""),
            "ymyl_category": metadata.get("ymyl_category"),
        }
        for ts_field in ("created_at", "updated_at", "event_time"):
            ts_str = metadata.get(ts_field)
            if ts_str:
                try:
                    kwargs[ts_field] = as_utc(datetime.fromisoformat(ts_str))
                except (ValueError, TypeError):
                    pass
        return Memory(**kwargs)

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
                id=memory_id or str(uuid.uuid4()),
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
            if item.get("event_time"):
                metadata["event_time"] = item["event_time"]
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

    @staticmethod
    def _adapt_bm25_weight(query: str, configured_weight: float) -> float:
        """Scale hybrid BM25 weight by detected query type.

        Multi-hop queries need importance-weighted retrieval to surface
        the right connecting facts; BM25's keyword matching dilutes that
        signal (the v1.5 regression had 41 multi-hop failures where the
        BM25 blend pulled keyword-matching but topically-wrong memories
        to the top of the pool). For multi-hop we disable BM25 entirely.

        Factual queries benefit from keyword matching for specific terms
        (names, dates, numbers). We pass the configured weight through
        unchanged.

        Temporal and broad queries get reduced weight to keep BM25 from
        dominating signals that matter more for those question shapes
        (recency for temporal, importance for broad).

        Classification follows the same signal lists as _adapt_scoring
        to keep query-type behavior consistent across the scoring and
        the hybrid paths.
        """
        q = query.lower().strip()

        multi_hop_signals = ("relationship between", "how does", "compare", "contrast",
                            "connection between", "relate to", "in common")
        if any(s in q for s in multi_hop_signals):
            return 0.0

        temporal_signals = ("when ", "what time", "what date", "how long ago",
                           "last time", "recently", "before the", "after the",
                           "how recent", "what year", "what month")
        if any(q.startswith(s) or s in q for s in temporal_signals):
            return configured_weight * 0.4

        factual_starts = ("where ", "who ", "what is ", "what was ", "what does ",
                         "what did ", "what are ", "what were ", "what do ",
                         "how old ", "how much ", "how many ",
                         "which ", "name ")
        is_short_what = q.startswith("what ") and len(q.split()) <= 10
        if any(q.startswith(s) for s in factual_starts) or is_short_what:
            return configured_weight

        # Broad / unknown
        return configured_weight * 0.6

    def _create_llm(self) -> BaseLLM:
        provider = self._resolve_provider(self.config.llm.provider, self.config.llm.api_key, "llm")
        if provider == "openai":
            return OpenAILLM(self.config.llm)
        if provider == "anthropic":
            from widemem.providers.llm.anthropic import AnthropicLLM
            return AnthropicLLM(self.config.llm)
        if provider == "ollama":
            from widemem.providers.llm.ollama import OllamaLLM
            config = self.config.llm
            if config.model == "gpt-4o-mini":
                config = LLMConfig(provider="ollama", model="llama3.2", base_url=config.base_url)
            return OllamaLLM(config)
        raise ValueError(
            f"Unknown LLM provider: {provider}. Supported: openai, anthropic, ollama"
        )

    def _create_embedder(self) -> BaseEmbedder:
        provider = self._resolve_provider(self.config.embedding.provider, self.config.embedding.api_key, "embedding")
        if provider == "openai":
            return OpenAIEmbedder(self.config.embedding)
        if provider == "sentence-transformers":
            from widemem.providers.embeddings.sentence_transformers import (
                SentenceTransformerEmbedder,
            )
            return SentenceTransformerEmbedder(self.config.embedding)
        if provider == "ollama":
            from widemem.providers.embeddings.ollama import OllamaEmbedder
            config = self.config.embedding
            if config.model == "text-embedding-3-small":
                config = EmbeddingConfig(provider="ollama", model="nomic-embed-text", dimensions=768, base_url=config.base_url)
            return OllamaEmbedder(config)
        raise ValueError(
            f"Unknown embedding provider: {provider}. Supported: openai, sentence-transformers, ollama"
        )

    @staticmethod
    def _resolve_provider(configured: str, api_key: Any, kind: str) -> str:
        """If provider is 'openai' (default) but no API key is available, fall back to Ollama."""
        if configured != "openai":
            return configured
        if api_key is not None:
            return "openai"
        import os
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        import logging
        logging.getLogger(__name__).info(
            "No OpenAI API key found for %s provider, falling back to Ollama (local). "
            "Set OPENAI_API_KEY or configure a provider explicitly to use OpenAI.",
            kind,
        )
        return "ollama"

    def _create_vector_store(self) -> BaseVectorStore:
        provider = self.config.vector_store.provider
        if provider == "faiss":
            from widemem.storage.vector.faiss_store import FAISSVectorStore
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
        if provider == "pgvector":
            from widemem.storage.vector.pgvector_store import PgVectorStore
            return PgVectorStore(
                self.config.vector_store,
                dimensions=self.config.embedding.dimensions,
            )
        raise ValueError(
            f"Unknown vector store provider: {provider}. "
            "Supported: faiss, qdrant, pgvector"
        )
