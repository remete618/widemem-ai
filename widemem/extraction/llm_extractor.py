from __future__ import annotations

from typing import List, Optional

from widemem.core.exceptions import ExtractionError
from widemem.core.types import Fact, YMYLConfig
from widemem.extraction.base import BaseExtractor
from widemem.extraction.collector import ExtractionCollector
from widemem.extraction.prompts import FACT_EXTRACTION_PROMPT, build_extraction_system
from widemem.providers.llm.base import BaseLLM
from widemem.scoring.ymyl import classify_ymyl_detailed


class LLMExtractor(BaseExtractor):
    def __init__(
        self,
        llm: BaseLLM,
        collector: Optional[ExtractionCollector] = None,
        ymyl_config: Optional[YMYLConfig] = None,
        custom_topics: Optional[List[str]] = None,
    ) -> None:
        self.llm = llm
        self.collector = collector
        self.ymyl_config = ymyl_config or YMYLConfig()
        self.custom_topics = custom_topics or []

        self._system_prompt = build_extraction_system(
            ymyl_enabled=self.ymyl_config.enabled,
            custom_topics=self.custom_topics,
        )

    def extract(self, text: str) -> list[Fact]:
        prompt = FACT_EXTRACTION_PROMPT.format(text=text)
        try:
            result = self.llm.generate_json(prompt, system=self._system_prompt)
        except Exception as e:
            raise ExtractionError(f"Fact extraction failed: {e}") from e

        facts = []
        for item in result.get("facts", []):
            if isinstance(item, dict) and "content" in item:
                importance = float(item.get("importance", 5.0))
                content = item["content"]
                ymyl_category = None

                if self.ymyl_config.enabled:
                    # Stage 1: Regex for strong pattern matches (fast, definitive)
                    regex_result = classify_ymyl_detailed(content, self.ymyl_config)
                    if regex_result.is_strong:
                        ymyl_category = regex_result.category
                        importance = max(importance, self.ymyl_config.min_importance)
                    else:
                        # Stage 2: Use LLM's semantic classification
                        llm_ymyl = item.get("ymyl_category")
                        if llm_ymyl and llm_ymyl in [c for c in self.ymyl_config.categories]:
                            ymyl_category = llm_ymyl
                            importance = max(importance, self.ymyl_config.min_importance)

                facts.append(Fact(
                    content=content,
                    importance=importance,
                    ymyl_category=ymyl_category,
                ))

        if self.collector and facts:
            try:
                self.collector.log(text, facts, model=self.llm.config.model)
            except Exception:
                pass

        return facts
