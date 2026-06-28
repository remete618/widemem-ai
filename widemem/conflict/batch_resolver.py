from __future__ import annotations

import logging
from collections.abc import Sequence

from widemem.conflict.prompts import (
    BATCH_CONFLICT_RESOLUTION_LINKED_PROMPT,
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
        linked_memories_by_fact: Sequence[Sequence[MemorySearchResult]] | None = None,
    ) -> list[ActionItem]:
        if not new_facts:
            return []

        if not existing_memories:
            return [
                ActionItem(
                    action=MemoryAction.ADD,
                    fact=f.content,
                    importance=f.importance,
                    ymyl_category=f.ymyl_category,
                )
                for f in new_facts
            ]

        if linked_memories_by_fact is None:
            return self._resolve_flat(new_facts, existing_memories)

        linked_groups = self._normalize_linked_groups(new_facts, linked_memories_by_fact)
        mapper = self._build_mapper(existing_memories, linked_groups)
        prompt = self._build_linked_prompt(new_facts, existing_memories, linked_groups, mapper)

        try:
            result = self.llm.generate_json(prompt, system=BATCH_CONFLICT_RESOLUTION_SYSTEM)
        except (OSError, ConnectionError, TimeoutError, RuntimeError, ProviderError) as exc:
            logger.warning(
                "Conflict resolver LLM call failed (%s), falling back to ADD with dedup",
                exc,
            )
            existing_hashes = {content_hash(m.memory.content) for m in existing_memories}
            return [
                ActionItem(
                    action=MemoryAction.ADD,
                    fact=f.content,
                    importance=f.importance,
                    ymyl_category=f.ymyl_category,
                )
                for f in new_facts
                if content_hash(f.content) not in existing_hashes
            ]

        return self._parse_actions(
            new_facts=new_facts,
            result=result,
            mapper=mapper,
            linked_groups=linked_groups,
            use_linked_validation=True,
        )

    def _resolve_flat(
        self,
        new_facts: list[Fact],
        existing_memories: list[MemorySearchResult],
    ) -> list[ActionItem]:
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
            logger.warning(
                "Conflict resolver LLM call failed (%s), falling back to ADD with dedup",
                exc,
            )
            existing_hashes = {content_hash(m.memory.content) for m in existing_memories}
            return [
                ActionItem(
                    action=MemoryAction.ADD,
                    fact=f.content,
                    importance=f.importance,
                    ymyl_category=f.ymyl_category,
                )
                for f in new_facts
                if content_hash(f.content) not in existing_hashes
            ]

        return self._parse_actions(
            new_facts=new_facts,
            result=result,
            mapper=mapper,
            linked_groups=None,
            use_linked_validation=False,
        )

    def _build_mapper(
        self,
        existing_memories: list[MemorySearchResult],
        linked_groups: list[list[MemorySearchResult]],
    ) -> IDMapper:
        mapper = IDMapper()
        for mem_result in existing_memories:
            mapper.add(mem_result.memory.id)
        for group in linked_groups:
            for mem_result in group:
                mapper.add(mem_result.memory.id)
        return mapper

    def _normalize_linked_groups(
        self,
        new_facts: list[Fact],
        linked_memories_by_fact: Sequence[Sequence[MemorySearchResult]],
    ) -> list[list[MemorySearchResult]]:
        groups: list[list[MemorySearchResult]] = []
        for idx in range(len(new_facts)):
            raw_group = (
                list(linked_memories_by_fact[idx])
                if idx < len(linked_memories_by_fact)
                else []
            )
            seen: set[str] = set()
            deduped: list[MemorySearchResult] = []
            for mem in raw_group:
                mem_id = mem.memory.id
                if mem_id in seen:
                    continue
                seen.add(mem_id)
                deduped.append(mem)
            groups.append(deduped)
        return groups

    def _build_linked_prompt(
        self,
        new_facts: list[Fact],
        existing_memories: list[MemorySearchResult],
        linked_groups: list[list[MemorySearchResult]],
        mapper: IDMapper,
    ) -> str:
        new_facts_str = "\n".join(
            f"[{i}] {f.content} (importance: {f.importance})"
            for i, f in enumerate(new_facts)
        )
        linked_sections = []
        for i, (fact, group) in enumerate(zip(new_facts, linked_groups)):
            linked_ids = []
            for mem in group:
                mapped_id = mapper.to_int(mem.memory.id)
                if mapped_id is not None:
                    linked_ids.append(mapped_id)
            linked_lines = "\n".join(
                f"  [{mapper.to_int(mem.memory.id)}] {mem.memory.content}"
                for mem in group
                if mapper.to_int(mem.memory.id) is not None
            )
            linked_sections.append(
                f"[{i}] {fact.content} (importance: {fact.importance})\n"
                f"linked_memory_ids: {linked_ids}\n"
                f"linked_candidates:\n{linked_lines if linked_lines else '  []'}"
            )
        linked_memories = "\n\n".join(linked_sections)
        existing_str = "\n".join(
            f"[{mapper.to_int(m.memory.id)}] {m.memory.content}"
            for m in existing_memories
            if mapper.to_int(m.memory.id) is not None
        )
        return BATCH_CONFLICT_RESOLUTION_LINKED_PROMPT.format(
            new_facts=new_facts_str,
            linked_memories=linked_memories,
            existing_memories=existing_str,
        )

    def _parse_actions(
        self,
        new_facts: list[Fact],
        result: dict,
        mapper: IDMapper,
        linked_groups: list[list[MemorySearchResult]] | None,
        use_linked_validation: bool,
    ) -> list[ActionItem]:
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

            if use_linked_validation:
                action, target_id = self._validate_linked_action(
                    action=action,
                    target_id=target_id,
                    fact=new_facts[fact_idx],
                    linked_group=(
                        linked_groups[fact_idx]
                        if linked_groups and fact_idx < len(linked_groups)
                        else []
                    ),
                )
            elif action in (MemoryAction.UPDATE, MemoryAction.DELETE) and target_id is None:
                action = MemoryAction.ADD
            actions.append(ActionItem(
                action=action,
                fact=new_facts[fact_idx].content,
                target_id=target_id,
                importance=float(item.get("importance", new_facts[fact_idx].importance)),
                ymyl_category=new_facts[fact_idx].ymyl_category,
            ))

        for i, fact in enumerate(new_facts):
            if i not in seen_indices:
                actions.append(ActionItem(
                    action=MemoryAction.ADD,
                    fact=fact.content,
                    importance=fact.importance,
                    ymyl_category=fact.ymyl_category,
                ))

        return actions

    def _validate_linked_action(
        self,
        action: MemoryAction,
        target_id: str | None,
        fact: Fact,
        linked_group: list[MemorySearchResult],
    ) -> tuple[MemoryAction, str | None]:
        valid_ids = {mem.memory.id for mem in linked_group}
        if action not in (MemoryAction.UPDATE, MemoryAction.DELETE):
            return action, None

        if target_id in valid_ids:
            return action, target_id

        if linked_group:
            fallback = linked_group[0]
            fallback_id = fallback.memory.id
            if (
                action == MemoryAction.UPDATE
                and content_hash(fallback.memory.content) == content_hash(fact.content)
            ):
                return MemoryAction.NONE, None
            return action, fallback_id

        if action == MemoryAction.UPDATE:
            return MemoryAction.ADD, None
        return MemoryAction.NONE, None
