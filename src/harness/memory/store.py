from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from harness.memory.embeddings import EmbeddingProvider
from harness.schemas import MemoryRecord, MemoryScope
from harness.storage.base import StorageBackend
from harness.tools.redaction import redact


@dataclass(frozen=True)
class RetrievedMemory:
    memory: MemoryRecord
    score: float


class MemoryStore:
    def __init__(self, storage: StorageBackend, embeddings: EmbeddingProvider) -> None:
        self.storage = storage
        self.embeddings = embeddings

    async def add(
        self,
        namespace: str,
        text: str,
        metadata: dict | None = None,
        *,
        scope: MemoryScope = MemoryScope.AGENT,
        importance: float = 0.5,
        confidence: float = 1.0,
        source_session_id: str | None = None,
        source_turn_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryRecord:
        redacted_text = redact(text)
        [embedding] = await self.embeddings.embed([redacted_text])
        embedding_metadata = _embedding_metadata(self.embeddings, embedding)
        record = MemoryRecord(
            namespace=namespace,
            text=redacted_text,
            embedding=embedding,
            embedding_provider=embedding_metadata.provider,
            embedding_model=embedding_metadata.model,
            embedding_dimensions=embedding_metadata.dimensions,
            metadata=redact(metadata or {}),
            scope=scope,
            importance=importance,
            confidence=confidence,
            source_session_id=source_session_id,
            source_turn_id=source_turn_id,
            expires_at=expires_at,
        )
        await self.storage.save_memory(record)
        return record

    async def retrieve(
        self,
        namespace: str,
        query: str,
        limit: int | None = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_expired: bool = False,
    ) -> list[RetrievedMemory]:
        [query_embedding] = await self.embeddings.embed([redact(query)])
        query_metadata = _embedding_metadata(self.embeddings, query_embedding)
        memories = await self.storage.list_memories(
            namespace,
            scopes=scopes,
            include_expired=include_expired,
        )
        ranked = [
            RetrievedMemory(memory=memory, score=cosine(query_embedding, memory.embedding))
            for memory in memories
            if _memory_matches_embedding(memory, query_metadata)
        ]
        ranked.sort(
            key=lambda item: (
                item.score,
                item.memory.importance,
                item.memory.confidence,
                item.memory.updated_at,
            ),
            reverse=True,
        )
        return ranked if limit is None else ranked[:limit]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    denominator = left_norm * right_norm
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True)
class EmbeddingMetadata:
    provider: str
    model: str | None
    dimensions: int


def _embedding_metadata(provider: EmbeddingProvider, vector: list[float]) -> EmbeddingMetadata:
    provider_name = getattr(provider, "provider_name", None)
    model_name = getattr(provider, "model_name", None) or getattr(provider, "model", None)
    dimensions = getattr(provider, "dimensions", None) or len(vector)
    if not isinstance(provider_name, str) or not provider_name:
        provider_name = f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
    return EmbeddingMetadata(
        provider=provider_name,
        model=str(model_name) if model_name is not None else None,
        dimensions=int(dimensions),
    )


def _memory_matches_embedding(memory: MemoryRecord, metadata: EmbeddingMetadata) -> bool:
    return (
        memory.embedding_provider == metadata.provider
        and memory.embedding_model == metadata.model
        and memory.embedding_dimensions == metadata.dimensions
    )
