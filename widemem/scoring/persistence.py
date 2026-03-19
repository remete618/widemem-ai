"""Persistence scoring: boost importance when the same fact is mentioned repeatedly."""

from __future__ import annotations

from widemem.core.types import MemoryAction
from widemem.storage.history import HistoryStore
from widemem.storage.vector.base import BaseVectorStore


def boost_on_repetition(
    content: str,
    user_id: str | None,
    history: HistoryStore,
    vector_store: BaseVectorStore,
    embedder,
    boost_amount: float = 1.0,
    max_importance: float = 10.0,
) -> str | None:
    """Check if similar content was added before. If so, boost the existing
    memory's importance. Returns the boosted memory_id, or None if no match.

    This implements natural reinforcement: things the user keeps mentioning
    automatically become harder to forget."""

    embedding = embedder.embed(content)
    filters = {"user_id": user_id} if user_id else None
    results = vector_store.search(vector=embedding, top_k=3, filters=filters)

    for mem_id, score, metadata in results:
        if score < 0.85:
            continue

        existing_content = metadata.get("content", "")
        # High similarity = likely the same fact repeated
        old_importance = metadata.get("importance", 5.0)
        new_importance = min(old_importance + boost_amount, max_importance)

        if new_importance > old_importance:
            metadata["importance"] = new_importance
            vector_store.update(id=mem_id, vector=embedding, metadata=metadata)
            history.log(
                mem_id, MemoryAction.UPDATE,
                old_content=f"importance:{old_importance:.1f}",
                new_content=f"importance:{new_importance:.1f} (repeated)",
            )
            return mem_id

    return None
