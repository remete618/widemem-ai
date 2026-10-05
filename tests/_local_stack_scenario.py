"""Runs the default local stack end to end and prints a JSON report.

Executed in a subprocess by tests/test_local_stack_e2e.py, so a native crash
(the macOS faiss/torch libomp segfault) fails a test instead of killing pytest.
Everything is real except the LLM: sentence-transformers all-MiniLM-L6-v2,
FAISS on disk, SQLite history. The LLM is a deterministic fake because CI has
no Ollama.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from widemem.core.memory import WideMemory
from widemem.core.types import LLMConfig, MemoryConfig, VectorStoreConfig
from widemem.providers.llm.base import BaseLLM

USER = "alice"

FACTS = {
    "I live in Boston and work as a nurse at Mass General.": [
        {"content": "Alice lives in Boston", "importance": 8},
        {"content": "Alice works as a nurse at Mass General Hospital", "importance": 7},
    ],
    "My favorite food is sushi.": [
        {"content": "Alice's favorite food is sushi", "importance": 5},
    ],
    "I have a golden retriever named Max.": [
        {"content": "Alice has a golden retriever named Max", "importance": 6},
    ],
}

QUERY = "Where does Alice live?"
EXPECTED_TOP = "Alice lives in Boston"


class FakeLLM(BaseLLM):
    """Canned extraction keyed on the user text; every resolved fact is an ADD."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(provider="fake", model="fake"), max_retries=1, retry_delay=0)
        self.calls: list[str] = []

    def _generate(self, prompt: str, system: str | None = None) -> str:
        return json.dumps(self._generate_json(prompt, system))

    def _generate_json(self, prompt: str, system: str | None = None) -> dict:
        system = system or ""
        if system.startswith("You are a fact extraction engine"):
            self.calls.append("extract")
            for text, facts in FACTS.items():
                if text in prompt:
                    return {"facts": facts}
            raise AssertionError(f"no canned extraction for prompt: {prompt[:200]!r}")
        if system.startswith("You are a memory conflict resolution engine"):
            self.calls.append("resolve")
            new_facts = prompt.split("New facts to process:\n", 1)[1].split("\n\n", 1)[0]
            indices = [int(i) for i in re.findall(r"^\[(\d+)\]", new_facts, flags=re.M)]
            return {"actions": [{"fact_index": i, "action": "add", "target_id": None} for i in indices]}
        raise AssertionError(f"unexpected LLM call, system={system[:80]!r}")


def _search(mem: WideMemory) -> dict:
    result = mem.search(QUERY, user_id=USER)
    top = result.results[0]
    return {
        "top": top.memory.content,
        "top_id": top.memory.id,
        "similarity": top.raw_similarity_score if top.raw_similarity_score is not None else top.similarity_score,
        "confidence": result.confidence.value,
        "n": len(result.results),
    }


def main(workdir: str) -> None:
    root = Path(workdir)
    config = MemoryConfig(
        history_db_path=str(root / "history.db"),
        vector_store=VectorStoreConfig(path=str(root / "vectors")),
    )
    report: dict = {
        "config_llm": list(WideMemory._llm_config(config.llm).model_dump(include={"provider", "model"}).values()),
    }

    llm = FakeLLM()
    try:
        mem = WideMemory(config=config, llm=llm)
    except OSError as exc:  # huggingface_hub: not cached and no network
        print(f"MODEL_UNAVAILABLE {exc}")
        sys.exit(3)
    report["embedder"] = type(mem.embedder).__name__
    report["embedder_model"] = mem.embedder.config.model
    report["dimensions"] = mem.embedder.dimensions
    report["vector_store"] = type(mem.vector_store).__name__
    report["vector_store_dims"] = mem.vector_store.dimensions
    report["torch_loaded"] = "torch" in sys.modules

    # The sequence that segfaulted on macOS: torch is loaded, the second add()
    # searches a non-empty FAISS index for conflicts, then search() does again.
    added = []
    for text in FACTS:
        added.append(len(mem.add(text, user_id=USER).memories))
    report["added"] = added
    report["llm_calls"] = llm.calls
    report["first"] = _search(mem)
    report["count"] = mem.count(user_id=USER)
    report["history"] = len(mem.get_history(report["first"]["top_id"]))
    mem.close()

    reopened = WideMemory(config=config, llm=FakeLLM())
    report["reopened"] = _search(reopened)
    report["reopened_count"] = reopened.count(user_id=USER)
    report["reopened_history"] = len(reopened.get_history(report["reopened"]["top_id"]))
    reopened.close()

    print("REPORT " + json.dumps(report))


if __name__ == "__main__":
    main(sys.argv[1])
