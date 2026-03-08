from __future__ import annotations

from typing import Dict, List, Optional

from widemem.core.types import Memory, MemoryAction, MemoryTier
from widemem.hierarchy.summarizer import MemorySummarizer
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.storage.history import HistoryStore
from widemem.storage.vector.base import BaseVectorStore
from widemem.utils.hashing import content_hash


class HierarchyManager:
    def __init__(
        self,
        summarizer: MemorySummarizer,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        history: HistoryStore,
        summarize_threshold: int = 10,
        theme_threshold: int = 3,
    ) -> None:
        self.summarizer = summarizer
        self.embedder = embedder
        self.vector_store = vector_store
        self.history = history
        self.summarize_threshold = summarize_threshold
        self.theme_threshold = theme_threshold

    def maybe_summarize(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        force: bool = False,
    ) -> List[Memory]:
        facts = self._get_memories_by_tier(MemoryTier.FACT, user_id=user_id, agent_id=agent_id)

        if not force and len(facts) < self.summarize_threshold:
            return []

        groups = self.summarizer.group_facts(facts)
        summaries = []

        for label, group_facts in groups:
            summary_text, importance = self.summarizer.summarize_group(group_facts)
            if not summary_text:
                continue

            memory = Memory(
                content=summary_text,
                user_id=user_id,
                agent_id=agent_id,
                tier=MemoryTier.SUMMARY,
                importance=importance,
                content_hash=content_hash(summary_text),
                metadata={"source_label": label, "source_count": len(group_facts)},
            )

            embedding = self.embedder.embed(summary_text)
            self.vector_store.insert(
                id=memory.id,
                vector=embedding,
                metadata=self._memory_to_metadata(memory),
            )
            self.history.log(memory.id, MemoryAction.ADD, new_content=summary_text)
            summaries.append(memory)

        if len(summaries) >= self.theme_threshold:
            theme_memories = self._synthesize_themes(summaries, user_id=user_id, agent_id=agent_id)
            summaries.extend(theme_memories)

        return summaries

    def _synthesize_themes(
        self,
        summaries: List[Memory],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Memory]:
        theme_text, importance = self.summarizer.synthesize_theme(summaries)
        if not theme_text:
            return []

        memory = Memory(
            content=theme_text,
            user_id=user_id,
            agent_id=agent_id,
            tier=MemoryTier.THEME,
            importance=importance,
            content_hash=content_hash(theme_text),
            metadata={"source_count": len(summaries)},
        )

        embedding = self.embedder.embed(theme_text)
        self.vector_store.insert(
            id=memory.id,
            vector=embedding,
            metadata=self._memory_to_metadata(memory),
        )
        self.history.log(memory.id, MemoryAction.ADD, new_content=theme_text)
        return [memory]

    def _get_memories_by_tier(
        self,
        tier: MemoryTier,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        max_results: int = 1000,
    ) -> List[Memory]:
        filters: Dict[str, str] = {"tier": tier.value}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id

        results = self.vector_store.list_all(
            filters=filters,
            max_results=max_results,
        )

        return [
            Memory(
                id=id,
                content=metadata.get("content", ""),
                user_id=metadata.get("user_id"),
                agent_id=metadata.get("agent_id"),
                tier=MemoryTier(metadata.get("tier", "fact")),
                importance=metadata.get("importance", 5.0),
            )
            for id, metadata in results
        ]

    def _memory_to_metadata(self, memory: Memory) -> dict:
        meta = {
            "content": memory.content,
            "user_id": memory.user_id,
            "agent_id": memory.agent_id,
            "run_id": memory.run_id,
            "tier": memory.tier.value,
            "importance": memory.importance,
            "content_hash": memory.content_hash,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
        }
        for k, v in memory.metadata.items():
            if k not in meta:
                meta[k] = v
        return meta
