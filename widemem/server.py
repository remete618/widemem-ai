from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    VectorStoreConfig,
)

logger = logging.getLogger("widemem.server")

_memory: Optional[WideMemory] = None


def _build_config() -> MemoryConfig:
    data_path = os.environ.get("WIDEMEM_DATA_PATH", "~/.widemem/data")
    data_path = str(Path(data_path).expanduser())

    llm_provider = os.environ.get("WIDEMEM_LLM_PROVIDER", "ollama")
    llm_model = os.environ.get("WIDEMEM_LLM_MODEL", "llama3.2")
    llm_base_url = os.environ.get("WIDEMEM_LLM_BASE_URL", "http://localhost:11434")
    embedding_provider = os.environ.get("WIDEMEM_EMBEDDING_PROVIDER", "sentence-transformers")

    llm_cfg = LLMConfig(provider=llm_provider, model=llm_model, base_url=llm_base_url)

    embedding_kwargs: dict = {"provider": embedding_provider}
    if embedding_provider == "sentence-transformers":
        embedding_kwargs["model"] = "all-MiniLM-L6-v2"
        embedding_kwargs["dimensions"] = 384
    emb_cfg = EmbeddingConfig(**embedding_kwargs)

    vs_cfg = VectorStoreConfig(provider="faiss", path=os.path.join(data_path, "faiss"))

    return MemoryConfig(
        llm=llm_cfg,
        embedding=emb_cfg,
        vector_store=vs_cfg,
        history_db_path=os.path.join(data_path, "history.db"),
    )


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _enforce_auth_policy() -> None:
    """Fail closed when bound to a non-local interface without an API key.

    The auth dependency disables itself when WIDEMEM_API_KEY is unset so
    that local development needs no setup. That is safe only while the
    server is bound to a loopback address. If the server is reachable
    from the network (WIDEMEM_HOST is anything other than localhost) and
    no API key is configured, /search and /add would be an open door to
    whatever LLM and embedding providers the deployment is configured
    with. Refuse to start in that case rather than silently exposing the
    endpoints.
    """
    host = os.environ.get("WIDEMEM_HOST", "127.0.0.1")
    has_key = bool(os.environ.get("WIDEMEM_API_KEY"))
    is_public = host not in _LOCAL_HOSTS

    if is_public and not has_key:
        raise RuntimeError(
            f"Refusing to start: WIDEMEM_HOST is bound to a non-local "
            f"interface ({host!r}) but WIDEMEM_API_KEY is not set. This "
            f"would expose /search and /add without authentication. Set "
            f"WIDEMEM_API_KEY to a strong secret, or bind to 127.0.0.1 "
            f"for local development."
        )
    if not has_key:
        logger.warning(
            "WIDEMEM_API_KEY not set; API authentication is disabled. "
            "Acceptable only for local development (host=%s).",
            host,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _memory
    _enforce_auth_policy()
    _memory = WideMemory(config=_build_config())
    yield
    _memory.close()
    _memory = None


app = FastAPI(title="widemem", lifespan=lifespan)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_auth(api_key: Optional[str] = Security(_api_key_header)) -> None:
    expected = os.environ.get("WIDEMEM_API_KEY")
    if not expected:
        return  # no key configured = auth disabled (local dev)
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class SearchRequest(BaseModel):
    query: str = Field(max_length=1000)
    user_id: Optional[str] = Field(default=None, max_length=256)
    top_k: int = Field(default=5, ge=1, le=100)


class MemoryItem(BaseModel):
    content: str
    importance: float
    score: float


class SearchResponse(BaseModel):
    memories: List[MemoryItem]


class AddRequest(BaseModel):
    text: str = Field(max_length=10000)
    user_id: Optional[str] = Field(default=None, max_length=256)


class AddResponse(BaseModel):
    added: int = 0
    updated: int = 0
    deleted: int = 0


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, _: None = Depends(_require_auth)):
    results = _memory.search(query=req.query, user_id=req.user_id, top_k=req.top_k)
    return SearchResponse(
        memories=[
            MemoryItem(
                content=r.memory.content,
                importance=r.memory.importance,
                score=r.final_score if r.final_score else r.similarity_score,
            )
            for r in results
        ]
    )


@app.post("/add", response_model=AddResponse)
def add(req: AddRequest, _: None = Depends(_require_auth)):
    result = _memory.add(text=req.text, user_id=req.user_id)
    return AddResponse(added=len(result.memories))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("WIDEMEM_PORT", os.environ.get("PORT", "11435")))
    host = os.environ.get("WIDEMEM_HOST", "127.0.0.1")
    # Fail fast with a clear message before uvicorn spawns the app, in
    # addition to the lifespan check (the authoritative guard for any
    # other ASGI entrypoint such as gunicorn).
    _enforce_auth_policy()
    uvicorn.run("widemem.server:app", host=host, port=port)
