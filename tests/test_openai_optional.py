"""OpenAI is an optional extra: widemem must import and run local-only without
the openai package, and fail with an install hint only when OpenAI is chosen."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run_without_openai(body: str) -> subprocess.CompletedProcess:
    script = "import sys\nsys.modules['openai'] = None\n" + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=120
    )


def test_package_imports_without_openai():
    r = _run_without_openai("""
        import widemem
        from widemem import WideMemory, MemoryConfig
        import widemem.core.memory
        import widemem.providers.llm.openai
        import widemem.providers.embeddings.openai
    """)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "build",
    [
        "from widemem.providers.llm.openai import OpenAILLM as C; from widemem.core.types import LLMConfig as K",
        "from widemem.providers.embeddings.openai import OpenAIEmbedder as C; from widemem.core.types import EmbeddingConfig as K",
    ],
)
def test_choosing_openai_without_the_extra_names_the_extra(build):
    r = _run_without_openai(f"""
        {build}
        from widemem.core.exceptions import ProviderError
        try:
            C(K(provider="openai", api_key="sk-test"))
        except ProviderError as e:
            assert 'widemem-ai[openai]' in str(e), e
            print("hinted")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "hinted"


def test_local_stack_runs_without_openai():
    pytest.importorskip("faiss")
    r = _run_without_openai("""
        from widemem import WideMemory, MemoryConfig
        from widemem.core.types import LLMConfig, EmbeddingConfig, VectorStoreConfig
        from widemem.providers.llm.base import BaseLLM
        from widemem.providers.embeddings.base import BaseEmbedder

        class LocalLLM(BaseLLM):
            def _generate(self, prompt, system=None):
                return '{"facts": [{"content": "Alice lives in Vienna", "importance": 6}]}'
            def _generate_json(self, prompt, system=None):
                import json
                return json.loads(self._generate(prompt, system))

        class LocalEmbedder(BaseEmbedder):
            def _embed(self, text):
                return [1.0, 0.0, 0.0]
            def _embed_batch(self, texts):
                return [[1.0, 0.0, 0.0] for _ in texts]

        cfg = MemoryConfig(
            llm=LLMConfig(provider="ollama", model="llama3.2"),
            embedding=EmbeddingConfig(provider="sentence-transformers", model="x", dimensions=3),
            vector_store=VectorStoreConfig(provider="faiss"),
            history_db_path=":memory:",
        )
        WideMemory._create_llm = lambda self: LocalLLM(cfg.llm)
        WideMemory._create_embedder = lambda self: LocalEmbedder(cfg.embedding)
        m = WideMemory(cfg)
        m.add("Alice lives in Vienna", user_id="alice")
        print(len(m.search("where does alice live", user_id="alice")))
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] != "0"


def test_openai_is_an_extra_not_a_core_dependency():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert not any(d.startswith("openai") for d in project["dependencies"])
    extras = project["optional-dependencies"]
    assert any(d.startswith("openai") for d in extras["openai"])
    assert any(d.startswith("openai") for d in extras["all"])
