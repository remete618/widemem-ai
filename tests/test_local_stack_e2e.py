"""End to end on the 2.0 default stack: real all-MiniLM-L6-v2, FAISS on disk,
SQLite history, with a fake LLM standing in for Ollama.

The stack runs once in a subprocess (tests/_local_stack_scenario.py). A
native crash there, like the macOS faiss/torch libomp segfault on the second
add(), shows up as a failed test with the exit code, not a dead pytest.
Torch is never imported into the pytest process itself.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from widemem.core.exceptions import StorageError
from widemem.core.memory import WideMemory
from widemem.core.types import EmbeddingConfig, MemoryConfig, VectorStoreConfig
from widemem.providers.embeddings.base import BaseEmbedder

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None
    or importlib.util.find_spec("faiss") is None,
    reason="local stack not installed: pip install -e '.[local]'",
)

REPO = Path(__file__).resolve().parent.parent
SCENARIO = Path(__file__).resolve().parent / "_local_stack_scenario.py"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("local_stack")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    # Users do not set this; pytest's conftest does. Run the stack as users do.
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    proc = subprocess.run(
        [sys.executable, str(SCENARIO), str(workdir)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    tail = (proc.stdout + proc.stderr)[-3000:]
    if proc.returncode == 3 and "MODEL_UNAVAILABLE" in proc.stdout and not os.environ.get("CI"):
        pytest.skip("all-MiniLM-L6-v2 not cached and not downloadable")
    assert proc.returncode == 0, f"local stack exited {proc.returncode} (-11/139 is a segfault)\n{tail}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("REPORT ")]
    assert lines, f"no report\n{tail}"
    return workdir, json.loads(lines[-1][len("REPORT "):])


def test_default_config_resolves_to_the_local_stack(run):
    _, r = run
    assert r["config_llm"] == ["ollama", "llama3.1:8b"]
    assert (r["embedder"], r["embedder_model"], r["dimensions"]) == (
        "SentenceTransformerEmbedder", "all-MiniLM-L6-v2", 384,
    )
    assert (r["vector_store"], r["vector_store_dims"]) == ("FAISSVectorStore", 384)
    assert r["torch_loaded"] is True


def test_add_add_search_after_torch_loads(run):
    _, r = run
    assert r["added"] == [2, 1, 1]
    # Resolution only runs once the index is non-empty: proof the later adds
    # searched FAISS with torch loaded.
    assert r["llm_calls"] == ["extract", "extract", "resolve", "extract", "resolve"]
    assert r["count"] == 4
    assert r["first"]["top"] == "Alice lives in Boston"


def test_known_match_is_not_low_confidence(run):
    _, r = run
    assert r["first"]["similarity"] >= 0.60
    assert r["first"]["confidence"] in ("high", "moderate")


def test_memories_and_history_survive_a_new_instance(run):
    _, r = run
    assert r["history"] >= 1
    assert r["reopened_count"] == r["count"]
    assert r["reopened"]["top_id"] == r["first"]["top_id"]
    assert r["reopened"]["similarity"] == pytest.approx(r["first"]["similarity"], abs=1e-5)
    assert r["reopened_history"] == r["history"]


class _OpenAISizedEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        super().__init__(EmbeddingConfig(provider="openai", model="text-embedding-3-small", dimensions=1536))

    def _embed(self, text: str) -> list[float]:
        return [0.0] * 1536

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


def test_a_384_index_refuses_a_1536_embedder(run):
    from tests._local_stack_scenario import FakeLLM

    workdir, _ = run
    config = MemoryConfig(
        history_db_path=str(workdir / "history.db"),
        vector_store=VectorStoreConfig(path=str(workdir / "vectors")),
    )
    with pytest.raises(StorageError, match=r"384.*1536"):
        WideMemory(config=config, llm=FakeLLM(), embedder=_OpenAISizedEmbedder())
