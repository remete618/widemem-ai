from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]
import numpy as np

from widemem.core.types import VectorStoreConfig
from widemem.storage.vector.base import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    """FAISS-backed vector store with thread-safe operations.

    All operations are serialized with a lock. FAISS index operations are
    fast (microseconds), so a simple mutex has negligible overhead compared
    to the LLM and embedding API calls that dominate real workloads.
    """

    def __init__(self, config: VectorStoreConfig, dimensions: int = 1536) -> None:
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required for the FAISS vector store. "
                "Install it with: pip install widemem-ai[faiss]"
            )
        super().__init__(config)
        self.dimensions = dimensions
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx = 0
        self._lock = threading.Lock()

        flat_index = faiss.IndexFlatIP(dimensions)
        self._index = faiss.IndexIDMap2(flat_index)

        if config.path:
            self._storage_path = Path(config.path).expanduser()
            self._load()
        else:
            self._storage_path = None

    def _validate_vector(self, vector: List[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimensions}, got {len(vector)}"
            )

    def insert(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        self._validate_vector(vector)
        vec = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vec)

        with self._lock:
            idx = self._next_idx
            ids = np.array([idx], dtype=np.int64)
            self._index.add_with_ids(vec, ids)
            self._id_to_idx[id] = idx
            self._idx_to_id[idx] = id
            self._metadata[id] = metadata
            self._next_idx += 1
            self._save()

    def search(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        self._validate_vector(vector)
        vec = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vec)

        with self._lock:
            if self._index.ntotal == 0:
                return []

            k = min(top_k * 3 if filters else top_k, self._index.ntotal)
            scores, indices = self._index.search(vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                id = self._idx_to_id.get(int(idx))
                if id is None:
                    continue
                meta = self._metadata.get(id, {})
                if filters and not self._matches_filters(meta, filters):
                    continue
                results.append((id, float(score), meta))
                if len(results) >= top_k:
                    break

        return results

    def update(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        self._validate_vector(vector)
        vec = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vec)

        with self._lock:
            old_idx = self._id_to_idx.get(id)
            if old_idx is not None:
                self._index.remove_ids(np.array([old_idx], dtype=np.int64))
                del self._idx_to_id[old_idx]

            idx = self._next_idx
            ids = np.array([idx], dtype=np.int64)
            self._index.add_with_ids(vec, ids)
            self._id_to_idx[id] = idx
            self._idx_to_id[idx] = id
            self._metadata[id] = metadata
            self._next_idx += 1
            self._save()

    def delete(self, id: str) -> None:
        with self._lock:
            idx = self._id_to_idx.get(id)
            if idx is None:
                return
            self._index.remove_ids(np.array([idx], dtype=np.int64))
            del self._id_to_idx[id]
            del self._idx_to_id[idx]
            self._metadata.pop(id, None)
            self._save()

    def get(self, id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        with self._lock:
            idx = self._id_to_idx.get(id)
            if idx is None:
                return None
            try:
                vec = self._index.reconstruct(int(idx))
                return vec.tolist(), self._metadata.get(id, {})
            except RuntimeError:
                return None

    def list_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 1000,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        with self._lock:
            results = []
            for id, meta in self._metadata.items():
                if filters and not self._matches_filters(meta, filters):
                    continue
                results.append((id, meta))
                if len(results) >= max_results:
                    break
            return results

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    def _save(self) -> None:
        if not self._storage_path:
            return
        self._storage_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._storage_path / "index.faiss"))
        state = {
            "metadata": self._metadata,
            "id_to_idx": self._id_to_idx,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
            "next_idx": self._next_idx,
        }
        with open(self._storage_path / "state.json", "w") as f:
            json.dump(state, f)

    def _load(self) -> None:
        if not self._storage_path:
            return
        index_path = self._storage_path / "index.faiss"
        state_path = self._storage_path / "state.json"
        if not index_path.exists() or not state_path.exists():
            return
        self._index = faiss.read_index(str(index_path))
        with open(state_path) as f:
            state = json.load(f)
        self._metadata = state["metadata"]
        self._id_to_idx = state["id_to_idx"]
        self._idx_to_id = {int(k): v for k, v in state["idx_to_id"].items()}
        self._next_idx = state["next_idx"]
