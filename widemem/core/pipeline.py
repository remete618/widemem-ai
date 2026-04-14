from __future__ import annotations

from typing import Callable, List, Optional

from widemem.conflict.batch_resolver import BatchConflictResolver
from widemem.core.types import (
    ActionItem,
    Fact,
    Memory,
    MemoryAction,
    MemorySearchResult,
    YMYLConfig,
)
from widemem.extraction.base import BaseExtractor
from widemem.providers.embeddings.base import BaseEmbedder
from widemem.retrieval.active import ActiveRetrieval, Clarification
from widemem.scoring.ymyl import is_ymyl_strong
from widemem.storage.history import HistoryStore
from widemem.storage.vector.base import BaseVectorStore
from widemem.utils.hashing import content_hash


class AddResult:
    __slots__ = ("memories", "clarifications")

    def __init__(
        self,
        memories: List[Memory],
        clarifications: Optional[List[Clarification]] = None,
    ) -> None:
        self.memories = memories
        self.clarifications = clarifications or []

    @property
    def has_clarifications(self) -> bool:
        return len(self.clarifications) > 0


class MemoryPipeline:
    def __init__(
        self,
        extractor: BaseExtractor,
        resolver: BatchConflictResolver,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        history: HistoryStore,
        active_retrieval: Optional[ActiveRetrieval] = None,
        ymyl_active_retrieval: Optional[ActiveRetrieval] = None,
        ymyl_config: Optional[YMYLConfig] = None,
    ) -> None:
        self.extractor = extractor
        self.resolver = resolver
        self.embedder = embedder
        self.vector_store = vector_store
        self.history = history
        self.active_retrieval = active_retrieval
        self.ymyl_active_retrieval = ymyl_active_retrieval
        self.ymyl_config = ymyl_config or YMYLConfig()

    def process(
        self,
        text: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        on_clarification: Optional[Callable[[List[Clarification]], Optional[List[str]]]] = None,
    ) -> AddResult:
        facts = self.extractor.extract(text)
        if not facts:
            return AddResult(memories=[])

        existing = self._find_existing(facts, user_id=user_id, agent_id=agent_id)

        clarifications: List[Clarification] = []
        active = self.active_retrieval
        if not active and self.ymyl_active_retrieval and self.ymyl_config.enabled:
            has_ymyl_fact = any(
                is_ymyl_strong(f.content, self.ymyl_config) for f in facts
            )
            if has_ymyl_fact:
                active = self.ymyl_active_retrieval

        if active and existing:
            clarifications = active.detect_conflicts(facts, existing)

            if clarifications and on_clarification:
                responses = on_clarification(clarifications)
                if responses is None:
                    return AddResult(memories=[], clarifications=clarifications)

        actions = self.resolver.resolve(facts, existing)
        existing_hashes = {
            m.memory.metadata.get("content_hash") or content_hash(m.memory.content)
            for m in existing
        }
        memories = self._execute_actions(actions, user_id=user_id, agent_id=agent_id, run_id=run_id, existing_hashes=existing_hashes)
        return AddResult(memories=memories, clarifications=clarifications)

    def _find_existing(
        self,
        facts: list[Fact],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> list[MemorySearchResult]:
        all_results: dict[str, MemorySearchResult] = {}

        embeddings = self.embedder.embed_batch([f.content for f in facts])

        filters: dict[str, str] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id

        for embedding in embeddings:
            results = self.vector_store.search(
                vector=embedding,
                top_k=5,
                filters=filters or None,
            )
            for id, score, metadata in results:
                if id not in all_results:
                    all_results[id] = MemorySearchResult(
                        memory=Memory(
                            id=id,
                            content=metadata.get("content", ""),
                            user_id=metadata.get("user_id"),
                            agent_id=metadata.get("agent_id"),
                            importance=metadata.get("importance", 5.0),
                        ),
                        similarity_score=score,
                    )

        return list(all_results.values())

    def _hash_exists(self, hash_val: str, existing_hashes: set) -> bool:
        return hash_val in existing_hashes

    def _execute_actions(
        self,
        actions: list[ActionItem],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        existing_hashes: Optional[set] = None,
    ) -> list[Memory]:
        existing_hashes = existing_hashes or set()
        results = []

        for action in actions:
            if action.action == MemoryAction.ADD:
                new_hash = content_hash(action.fact)
                if self._hash_exists(new_hash, existing_hashes):
                    continue

                memory = Memory(
                    content=action.fact,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    importance=action.importance,
                    content_hash=new_hash,
                )
                embedding = self.embedder.embed(action.fact)
                self.vector_store.insert(
                    id=memory.id,
                    vector=embedding,
                    metadata=self._memory_to_metadata(memory),
                )
                existing_hashes.add(new_hash)
                self.history.log(memory.id, MemoryAction.ADD, new_content=action.fact)
                results.append(memory)

            elif action.action == MemoryAction.UPDATE and action.target_id:
                existing = self.vector_store.get(action.target_id)
                old_content = None
                if existing:
                    old_content = existing[1].get("content")

                new_hash = content_hash(action.fact)
                if existing and existing[1].get("content_hash") == new_hash:
                    continue

                memory = Memory(
                    id=action.target_id,
                    content=action.fact,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    importance=action.importance,
                    content_hash=new_hash,
                )
                embedding = self.embedder.embed(action.fact)
                self.vector_store.update(
                    id=action.target_id,
                    vector=embedding,
                    metadata=self._memory_to_metadata(memory),
                )
                self.history.log(
                    action.target_id, MemoryAction.UPDATE,
                    old_content=old_content, new_content=action.fact,
                )
                results.append(memory)

            elif action.action == MemoryAction.DELETE and action.target_id:
                existing = self.vector_store.get(action.target_id)
                old_content = existing[1].get("content") if existing else None
                self.vector_store.delete(action.target_id)
                self.history.log(
                    action.target_id, MemoryAction.DELETE,
                    old_content=old_content,
                )

        return results

    def _memory_to_metadata(self, memory: Memory) -> dict:
        return {
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
