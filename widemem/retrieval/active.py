from __future__ import annotations

from typing import Callable, List, Optional

from pydantic import BaseModel

from widemem.core.types import Fact, MemorySearchResult
from widemem.providers.llm.base import BaseLLM
from widemem.retrieval.prompts import (
    CONTRADICTION_DETECTION_PROMPT,
    CONTRADICTION_DETECTION_SYSTEM,
)
from widemem.utils.id_mapping import IDMapper


class Clarification(BaseModel):
    new_fact: str
    existing_content: str
    existing_memory_id: Optional[str] = None
    conflict_type: str  # "contradiction" or "ambiguity"
    question: str


ClarificationCallback = Callable[[List[Clarification]], Optional[List[str]]]


class ActiveRetrieval:
    def __init__(self, llm: BaseLLM, similarity_threshold: float = 0.6) -> None:
        self.llm = llm
        self.similarity_threshold = similarity_threshold

    def detect_conflicts(
        self,
        new_facts: List[Fact],
        existing_memories: List[MemorySearchResult],
    ) -> List[Clarification]:
        if not new_facts or not existing_memories:
            return []

        relevant = [
            m for m in existing_memories
            if m.similarity_score >= self.similarity_threshold
        ]
        if not relevant:
            return []

        mapper = IDMapper()
        for mem_result in relevant:
            mapper.add(mem_result.memory.id)

        existing_str = "\n".join(
            f"[{mapper.to_int(m.memory.id)}] {m.memory.content}"
            for m in relevant
        )

        all_clarifications: List[Clarification] = []

        for fact in new_facts:
            prompt = CONTRADICTION_DETECTION_PROMPT.format(
                new_fact=fact.content,
                existing_memories=existing_str,
            )

            try:
                result = self.llm.generate_json(prompt, system=CONTRADICTION_DETECTION_SYSTEM)
            except Exception:
                continue

            if not result.get("has_conflict", False):
                continue

            for conflict in result.get("conflicts", []):
                existing_id = conflict.get("existing_memory_id")
                real_id = mapper.to_uuid(int(existing_id)) if existing_id is not None else None

                all_clarifications.append(Clarification(
                    new_fact=fact.content,
                    existing_content=conflict.get("existing_content", ""),
                    existing_memory_id=real_id,
                    conflict_type=conflict.get("type", "ambiguity"),
                    question=conflict.get("question", ""),
                ))

        return all_clarifications
