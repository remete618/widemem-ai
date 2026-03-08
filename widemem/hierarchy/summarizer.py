from __future__ import annotations

from typing import List, Tuple

from widemem.core.types import Memory
from widemem.hierarchy.prompts import (
    GROUP_FACTS_PROMPT,
    GROUP_FACTS_SYSTEM,
    SUMMARIZE_FACTS_PROMPT,
    SUMMARIZE_FACTS_SYSTEM,
    SYNTHESIZE_THEME_PROMPT,
    SYNTHESIZE_THEME_SYSTEM,
)
from widemem.providers.llm.base import BaseLLM


class MemorySummarizer:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def group_facts(self, facts: List[Memory]) -> List[Tuple[str, List[Memory]]]:
        if len(facts) < 2:
            return []

        facts_str = "\n".join(f"[{i}] {f.content}" for i, f in enumerate(facts))
        prompt = GROUP_FACTS_PROMPT.format(facts=facts_str)
        result = self.llm.generate_json(prompt, system=GROUP_FACTS_SYSTEM)

        groups = []
        for group in result.get("groups", []):
            label = group.get("label", "unknown")
            indices = group.get("fact_indices", [])
            members = [facts[i] for i in indices if i < len(facts)]
            if len(members) >= 2:
                groups.append((label, members))

        return groups

    def summarize_group(self, facts: List[Memory]) -> Tuple[str, float]:
        facts_str = "\n".join(f"- {f.content}" for f in facts)
        prompt = SUMMARIZE_FACTS_PROMPT.format(facts=facts_str)
        result = self.llm.generate_json(prompt, system=SUMMARIZE_FACTS_SYSTEM)

        summary = result.get("summary", "")
        importance = float(result.get("importance", 7.0))
        return summary, importance

    def synthesize_theme(self, summaries: List[Memory]) -> Tuple[str, float]:
        summaries_str = "\n".join(f"- {s.content}" for s in summaries)
        prompt = SYNTHESIZE_THEME_PROMPT.format(summaries=summaries_str)
        result = self.llm.generate_json(prompt, system=SYNTHESIZE_THEME_SYSTEM)

        theme = result.get("theme", "")
        importance = float(result.get("importance", 8.0))
        return theme, importance
