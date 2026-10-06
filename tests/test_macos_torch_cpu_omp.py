"""On macOS, faiss and torch each bundle libomp. If faiss is imported before
torch and the embedding model runs on CPU, torch's OpenMP thread pool
segfaults the process. The embedder pins torch to one thread in that case."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent

SCENARIO = r"""
import faiss  # imported before torch: the order that crashes
import numpy as np
import sentence_transformers  # noqa: F401
import torch
torch.backends.mps.is_available = lambda: False  # force the CPU path (Intel Mac, CI runner)
from widemem.core.types import EmbeddingConfig
from widemem.providers.embeddings.sentence_transformers import SentenceTransformerEmbedder
emb = SentenceTransformerEmbedder(EmbeddingConfig())
v = emb.embed_batch(["where does alice live"] * 64)
idx = faiss.IndexFlatIP(384)
idx.add(np.asarray(v, dtype="float32"))
idx.search(np.asarray(v[:1], dtype="float32"), 5)
for _ in range(5):
    emb.embed_batch(["where does alice live"] * 64)
print("OK")
"""


@pytest.mark.skipif(sys.platform != "darwin", reason="the libomp clash is macOS-only")
def test_faiss_before_torch_on_cpu_does_not_crash():
    for mod in ("faiss", "sentence_transformers"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} not installed")
    env = {k: v for k, v in os.environ.items() if k != "KMP_DUPLICATE_LIB_OK"}
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run([sys.executable, "-c", SCENARIO], capture_output=True, text=True, env=env, timeout=600)
    if "OSError" in proc.stderr and "all-MiniLM-L6-v2" in proc.stderr and not os.environ.get("CI"):
        pytest.skip("model unavailable")
    assert proc.returncode == 0 and "OK" in proc.stdout, f"exit {proc.returncode}: {proc.stderr[-2000:]}"


class _FakeModel:
    def __init__(self, device):
        self.device = SimpleNamespace(type=device)

    def get_sentence_embedding_dimension(self):
        return 384


class _FakeModelNewApi(_FakeModel):
    def get_embedding_dimension(self):
        return 384

    def get_sentence_embedding_dimension(self):
        raise AssertionError("deprecated method called although the new one exists")


@pytest.mark.parametrize("model_cls", [_FakeModel, _FakeModelNewApi], ids=["old-api", "new-api"])
@pytest.mark.parametrize(
    "platform, faiss_loaded, device, pinned",
    [
        ("darwin", True, "cpu", True),
        ("darwin", False, "cpu", False),
        ("darwin", True, "mps", False),
        ("darwin", False, "mps", False),
        ("linux", True, "cpu", False),
    ],
)
def test_torch_is_pinned_only_when_the_clash_can_happen(
    monkeypatch, platform, faiss_loaded, device, pinned, model_cls
):
    import widemem.providers.embeddings.sentence_transformers as mod
    from widemem.core.types import EmbeddingConfig

    calls = []
    fake_torch = SimpleNamespace(set_num_threads=calls.append)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda name: model_cls(device))
    )
    if faiss_loaded:
        monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace())
    else:
        monkeypatch.delitem(sys.modules, "faiss", raising=False)
    monkeypatch.setattr(mod.sys, "platform", platform)
    mod.SentenceTransformerEmbedder(EmbeddingConfig())
    assert calls == ([1] if pinned else [])
