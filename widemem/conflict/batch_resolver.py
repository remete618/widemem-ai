from __future__ import annotations

import logging

from widemem.conflict.prompts import (
    BATCH_CONFLICT_RESOLUTION_PROMPT,
    BATCH_CONFLICT_RESOLUTION_SYSTEM,
)
from widemem.core.exceptions import ProviderError
from widemem.core.types import ActionItem, Fact, MemoryAction, MemorySearchResult
from widemem.providers.llm.base import BaseLLM
from widemem.utils.hashing import content_hash
from widemem.utils.id_mapping import IDMapper

logger = logging.getLogger(__name__)


class BatchConflictResolver:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def resolve(
        self,
        new_facts: list[Fact],
        existing_memories: list[MemorySearchResult],
    ) -> list[ActionItem]:
        if not new_facts:
            return []

        if not existing_memories:
            return [
                ActionItem(action=MemoryAction.ADD, fact=f.content, importance=f.importance)
                for f in new_facts
            ]

        mapper = IDMapper()
        for mem_result in existing_memories:
            mapper.add(mem_result.memory.id)

        new_facts_str = "\n".join(
            f"[{i}] {f.content} (importance: {f.importance})"
            for i, f in enumerate(new_facts)
        )
        existing_str = "\n".join(
            f"[{mapper.to_int(m.memory.id)}] {m.memory.content}"
            for m in existing_memories
        )

        prompt = BATCH_CONFLICT_RESOLUTION_PROMPT.format(
            new_facts=new_facts_str,
            existing_memories=existing_str,
        )

        try:
            result = self.llm.generate_json(prompt, system=BATCH_CONFLICT_RESOLUTION_SYSTEM)
        except (OSError, ConnectionError, TimeoutError, RuntimeError, ProviderError) as exc:
            logger.warning("Conflict resolver LLM call failed (%s), falling back to ADD with dedup", exc)
            existing_hashes = {content_hash(m.memory.content) for m in existing_memories}
            return [
                ActionItem(action=MemoryAction.ADD, fact=f.content, importance=f.importance)
                for f in new_facts
                if content_hash(f.content) not in existing_hashes
            ]

        actions = []
        seen_indices: set[int] = set()
        for item in result.get("actions", []):
            fact_idx = item.get("fact_index")
            if fact_idx is None or not isinstance(fact_idx, int) or fact_idx < 0 or fact_idx >= len(new_facts):
                continue
            if fact_idx in seen_indices:
                continue
            seen_indices.add(fact_idx)

            target_id = None
            raw_target = item.get("target_id")
            if raw_target is not None:
                try:
                    target_id = mapper.to_uuid(int(raw_target))
                except (ValueError, TypeError):
                    target_id = None

            action_str = item.get("action", "add").lower()
            try:
                action = MemoryAction(action_str)
            except ValueError:
                action = MemoryAction.ADD

            if action in (MemoryAction.UPDATE, MemoryAction.DELETE) and target_id is None:
                action = MemoryAction.ADD
            actions.append(ActionItem(
                action=action,
                fact=new_facts[fact_idx].content,
                target_id=target_id,
                importance=float(item.get("importance", new_facts[fact_idx].importance)),
            ))

        for i, fact in enumerate(new_facts):
            if i not in seen_indices:
                actions.append(ActionItem(
                    action=MemoryAction.ADD,
                    fact=fact.content,
                    importance=fact.importance,
                ))

        return actions
