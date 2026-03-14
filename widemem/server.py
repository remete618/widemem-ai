from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from widemem.core.memory import WideMemory
from widemem.core.types import (
    EmbeddingConfig,
    LLMConfig,
    MemoryConfig,
    VectorStoreConfig,
)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _memory
    _memory = WideMemory(config=_build_config())
    yield
    _memory.close()
    _memory = None


app = FastAPI(title="widemem", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=100)


class MemoryItem(BaseModel):
    content: str
    importance: float
    score: float


class SearchResponse(BaseModel):
    memories: List[MemoryItem]


class AddRequest(BaseModel):
    text: str
    user_id: Optional[str] = None


class AddResponse(BaseModel):
    added: int = 0
    updated: int = 0
    deleted: int = 0


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
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
def add(req: AddRequest):
    result = _memory.add(text=req.text, user_id=req.user_id)
    return AddResponse(added=len(result.memories))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("WIDEMEM_PORT", "11435"))
    host = os.environ.get("WIDEMEM_HOST", "127.0.0.1")
    uvicorn.run("widemem.server:app", host=host, port=port)
