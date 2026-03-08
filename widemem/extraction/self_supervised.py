from __future__ import annotations

import json
from typing import List, Optional

from widemem.core.types import Fact
from widemem.extraction.base import BaseExtractor


class SelfSupervisedExtractor(BaseExtractor):
    """Extraction using a fine-tuned small model with LLM fallback.

    The model is expected to take input text and return JSON:
    {"facts": [{"content": "...", "importance": 7.0, "confidence": 0.95}, ...]}

    When confidence is below threshold, falls back to the LLM extractor.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        fallback: Optional[BaseExtractor] = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.model_path = model_path
        self.fallback = fallback
        self.confidence_threshold = confidence_threshold
        self._model = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        try:
            from transformers import pipeline as hf_pipeline
            self._model = hf_pipeline(
                "text2text-generation",
                model=path,
                max_new_tokens=512,
            )
        except ImportError:
            pass
        except Exception:
            pass

    def extract(self, text: str) -> List[Fact]:
        if self._model is None:
            if self.fallback:
                return self.fallback.extract(text)
            return []

        try:
            result = self._model(text)[0]["generated_text"]
            parsed = json.loads(result)
        except (json.JSONDecodeError, KeyError, IndexError):
            if self.fallback:
                return self.fallback.extract(text)
            return []

        facts = []
        low_confidence = False

        for item in parsed.get("facts", []):
            confidence = float(item.get("confidence", 0.5))
            if confidence < self.confidence_threshold:
                low_confidence = True
                break
            facts.append(Fact(
                content=item["content"],
                importance=float(item.get("importance", 5.0)),
            ))

        if low_confidence and self.fallback:
            return self.fallback.extract(text)

        return facts

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
