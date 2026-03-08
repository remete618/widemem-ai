"""Tests for extraction collector, self-supervised extractor, and fallback chain."""

from __future__ import annotations

import json
import tempfile
from typing import List

import pytest

from widemem.core.types import Fact
from widemem.extraction.base import BaseExtractor
from widemem.extraction.collector import ExtractionCollector
from widemem.extraction.self_supervised import SelfSupervisedExtractor


class MockFallbackExtractor(BaseExtractor):
    def __init__(self):
        self.called = False

    def extract(self, text: str) -> List[Fact]:
        self.called = True
        return [Fact(content=f"fallback: {text}", importance=5.0)]


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# --- Collector Tests ---

class TestExtractionCollector:
    def test_log_and_count(self, tmp_dir):
        collector = ExtractionCollector(f"{tmp_dir}/extractions.db")
        assert collector.count() == 0

        facts = [Fact(content="Lives in Berlin", importance=8.0)]
        collector.log("I live in Berlin", facts, model="gpt-4")
        assert collector.count() == 1

        collector.log("I work at Google", [Fact(content="Works at Google", importance=7.0)])
        assert collector.count() == 2
        collector.close()

    def test_export_jsonl(self, tmp_dir):
        collector = ExtractionCollector(f"{tmp_dir}/extractions.db")
        collector.log("I live in Berlin", [Fact(content="Lives in Berlin", importance=8.0)])
        collector.log("I like coffee", [Fact(content="Likes coffee", importance=3.0)])

        output_path = f"{tmp_dir}/output.jsonl"
        count = collector.export(output_path)
        assert count == 2

        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["input"] == "I live in Berlin"
        assert first["output"][0]["content"] == "Lives in Berlin"
        collector.close()

    def test_export_with_limit(self, tmp_dir):
        collector = ExtractionCollector(f"{tmp_dir}/extractions.db")
        for i in range(5):
            collector.log(f"fact {i}", [Fact(content=f"fact {i}", importance=5.0)])

        output_path = f"{tmp_dir}/output.jsonl"
        count = collector.export(output_path, limit=3)
        assert count == 3
        collector.close()

    def test_multiple_facts_per_entry(self, tmp_dir):
        collector = ExtractionCollector(f"{tmp_dir}/extractions.db")
        facts = [
            Fact(content="Lives in Berlin", importance=8.0),
            Fact(content="Works at Google", importance=7.0),
        ]
        collector.log("I live in Berlin and work at Google", facts)

        output_path = f"{tmp_dir}/output.jsonl"
        collector.export(output_path)

        with open(output_path) as f:
            data = json.loads(f.readline())
        assert len(data["output"]) == 2
        collector.close()


# --- Self-Supervised Extractor Tests ---

class TestSelfSupervisedExtractor:
    def test_no_model_uses_fallback(self):
        fallback = MockFallbackExtractor()
        extractor = SelfSupervisedExtractor(model_path=None, fallback=fallback)

        assert not extractor.is_loaded
        facts = extractor.extract("hello")
        assert fallback.called
        assert len(facts) == 1
        assert facts[0].content == "fallback: hello"

    def test_no_model_no_fallback_returns_empty(self):
        extractor = SelfSupervisedExtractor(model_path=None, fallback=None)
        facts = extractor.extract("hello")
        assert facts == []

    def test_invalid_model_path_uses_fallback(self):
        fallback = MockFallbackExtractor()
        extractor = SelfSupervisedExtractor(
            model_path="/nonexistent/model",
            fallback=fallback,
        )
        assert not extractor.is_loaded
        extractor.extract("test")
        assert fallback.called

    def test_confidence_threshold(self):
        extractor = SelfSupervisedExtractor(confidence_threshold=0.8)
        # Can't test with real model, but verify the threshold is set
        assert extractor.confidence_threshold == 0.8


# --- Integration: Collector in LLM Extractor ---

class TestCollectorIntegration:
    def test_llm_extractor_logs_to_collector(self, tmp_dir):
        from widemem.core.types import LLMConfig
        from widemem.extraction.llm_extractor import LLMExtractor
        from widemem.providers.llm.base import BaseLLM

        class MockLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig())
            def generate(self, prompt, system=None):
                return '{"facts": [{"content": "Lives in Berlin", "importance": 8}]}'
            def generate_json(self, prompt, system=None):
                return {"facts": [{"content": "Lives in Berlin", "importance": 8}]}

        collector = ExtractionCollector(f"{tmp_dir}/extractions.db")
        extractor = LLMExtractor(MockLLM(), collector=collector)

        facts = extractor.extract("I live in Berlin")
        assert len(facts) == 1
        assert collector.count() == 1
        collector.close()

    def test_llm_extractor_without_collector(self, tmp_dir):
        from widemem.core.types import LLMConfig
        from widemem.extraction.llm_extractor import LLMExtractor
        from widemem.providers.llm.base import BaseLLM

        class MockLLM(BaseLLM):
            def __init__(self):
                super().__init__(LLMConfig())
            def generate(self, prompt, system=None):
                return '{"facts": []}'
            def generate_json(self, prompt, system=None):
                return {"facts": [{"content": "test", "importance": 5}]}

        extractor = LLMExtractor(MockLLM(), collector=None)
        facts = extractor.extract("test")
        assert len(facts) == 1  # works fine without collector
