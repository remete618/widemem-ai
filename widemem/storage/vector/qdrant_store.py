from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from widemem.core.exceptions import StorageError
from widemem.core.types import VectorStoreConfig
from widemem.storage.vector.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    def __init__(
        self,
        config: VectorStoreConfig,
        dimensions: int = 1536,
        collection_name: str = "widemem",
    ) -> None:
        super().__init__(config)
        self.dimensions = dimensions
        self.collection_name = collection_name

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise StorageError("Install qdrant: pip install widemem[qdrant]")

        if config.path:
            self.client = QdrantClient(path=config.path)
        else:
            self.client = QdrantClient(url="localhost", port=6333)

        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimensions,
                    distance=Distance.COSINE,
                ),
            )

    def insert(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=self._to_qdrant_id(id),
                vector=vector,
                payload={**metadata, "_widemem_id": id},
            )],
        )

    def search(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        query_filter = None
        if filters:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            query_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        output = []
        for point in results.points:
            payload = point.payload or {}
            widemem_id = payload.pop("_widemem_id", str(point.id))
            output.append((widemem_id, point.score, payload))
        return output

    def update(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        self.insert(id, vector, metadata)

    def delete(self, id: str) -> None:
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[self._to_qdrant_id(id)]),
        )

    def get(self, id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[self._to_qdrant_id(id)],
            with_vectors=True,
            with_payload=True,
        )
        if not results:
            return None
        point = results[0]
        payload = point.payload or {}
        payload.pop("_widemem_id", None)
        vector = point.vector if isinstance(point.vector, list) else []
        return vector, payload

    def list_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 1000,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        query_filter = None
        if filters:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            query_filter = Filter(must=conditions)

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=max_results,
            with_payload=True,
            with_vectors=False,
        )

        output = []
        for point in results[0]:
            payload = point.payload or {}
            widemem_id = payload.pop("_widemem_id", str(point.id))
            output.append((widemem_id, payload))
        return output

    def _to_qdrant_id(self, id: str) -> str:
        try:
            uuid.UUID(id)
            return id
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, id))
