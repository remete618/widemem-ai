"""A LangChain `BaseRetriever` backed by widemem search."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
    raise ImportError(
        'LangChain is not installed. Install it with: pip install "widemem-ai[langchain]"'
    ) from exc

from pydantic import ConfigDict, Field

from widemem.core.memory import WideMemory
from widemem.core.types import RetrievalConfidence, RetrievalMode

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )

# RetrievalConfidence is a str enum with no ordering of its own, and the
# comparison below is a threshold test, so the ranks live here explicitly.
_CONFIDENCE_RANK: dict[RetrievalConfidence, int] = {
    RetrievalConfidence.NONE: 0,
    RetrievalConfidence.LOW: 1,
    RetrievalConfidence.MODERATE: 2,
    RetrievalConfidence.HIGH: 3,
}


class WidememRetriever(BaseRetriever):
    """Retrieve widemem memories as LangChain documents.

        retriever = WidememRetriever(memory=memory, user_id="alice", top_k=5)
        chain = create_retrieval_chain(retriever, qa_chain)

    `min_confidence` is all-or-nothing, not a per-document filter. widemem
    reports confidence for the result set rather than per memory, so a set
    below the threshold yields no documents at all. That is the useful shape
    for a chain: an empty list is something a chain can branch on, whereas a
    thinned list of weak matches silently degrades the answer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory: WideMemory
    """The store to search."""

    user_id: Optional[str] = None
    """Scope the search to one user. Without it, every user's memories are in scope."""

    top_k: int = Field(default=5, ge=1)
    """Maximum documents to return."""

    retrieval_mode: RetrievalMode = RetrievalMode.BALANCED
    """widemem retrieval mode: wider candidate pools cost latency and tokens."""

    min_confidence: Optional[RetrievalConfidence] = None
    """Drop the whole result set when widemem's confidence falls below this."""

    def _search(self, query: str) -> list[Document]:
        result = self.memory.search(
            query=query,
            user_id=self.user_id,
            top_k=self.top_k,
            mode=self.retrieval_mode,
        )

        confidence = getattr(result, "confidence", None)
        if self.min_confidence is not None and confidence is not None:
            if _CONFIDENCE_RANK[confidence] < _CONFIDENCE_RANK[self.min_confidence]:
                return []

        return [self._to_document(item, confidence) for item in result]

    @staticmethod
    def _to_document(item: Any, confidence: Optional[RetrievalConfidence]) -> Document:
        memory = item.memory
        created_at = getattr(memory, "created_at", None)
        return Document(
            id=memory.id,
            page_content=memory.content,
            metadata={
                "memory_id": memory.id,
                "user_id": memory.user_id,
                "agent_id": memory.agent_id,
                "importance": memory.importance,
                "ymyl_category": memory.ymyl_category,
                "created_at": created_at.isoformat() if created_at else None,
                "similarity_score": item.similarity_score,
                "final_score": item.final_score,
                "retrieval_confidence": confidence.value if confidence else None,
            },
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._search(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        # `search()` is synchronous and does real work: embedding the query,
        # then a vector scan. Calling it directly would stall the event loop
        # for every other coroutine in the chain.
        return await asyncio.to_thread(self._search, query)
