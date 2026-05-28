from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from harness.memory.embeddings import EmbeddingProvider
from harness.memory.store import MemoryStore, RetrievedMemory
from harness.observability.sink import EventSink
from harness.schemas import MemoryRecord, MemoryScope
from harness.storage.base import StorageBackend
from harness.tools.redaction import redact


class MemoryPolicy(BaseModel):
    enabled: bool = True
    namespace: str = "default"
    retrieval_limit: int = Field(default=5, ge=0)
    min_score: float = -1.0
    max_context_chars: int = Field(default=4_000, ge=1)
    scopes: list[MemoryScope] = Field(
        default_factory=lambda: [
            MemoryScope.SESSION,
            MemoryScope.AGENT,
            MemoryScope.GLOBAL,
        ]
    )
    auto_capture: bool = False


class MemoryContext(BaseModel):
    content: str
    memories: list[RetrievedMemory] = Field(default_factory=list)

    @property
    def memory_ids(self) -> list[str]:
        return [item.memory.id for item in self.memories]


class MemoryContextBuilder:
    def build(self, memories: list[RetrievedMemory], *, max_chars: int) -> MemoryContext:
        if not memories:
            return MemoryContext(content="", memories=[])

        lines = ["Relevant memory:"]
        included: list[RetrievedMemory] = []
        seen: set[str] = set()
        for item in memories:
            memory = item.memory
            text = _single_line(memory.text)
            if not text or text in seen:
                continue
            seen.add(text)
            prefix = f"- [{memory.scope.value} score={item.score:.3f}] "
            next_line = f"{prefix}{text}"
            next_content = "\n".join([*lines, next_line])
            if len(next_content) > max_chars:
                break
            lines.append(next_line)
            included.append(item)

        if not included:
            return MemoryContext(content="", memories=[])
        return MemoryContext(content="\n".join(lines), memories=included)


class MemoryCandidate(BaseModel):
    text: str
    scope: MemoryScope = MemoryScope.SESSION
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    expires_at: datetime | None = None


class MemoryExchange(BaseModel):
    session_id: str
    user_message: str
    assistant_message: str
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryExtractor(ABC):
    @abstractmethod
    async def extract(self, exchange: MemoryExchange) -> list[MemoryCandidate]:
        raise NotImplementedError


class NoopMemoryExtractor(MemoryExtractor):
    async def extract(self, exchange: MemoryExchange) -> list[MemoryCandidate]:
        _ = exchange
        return []


@dataclass
class MemoryManager:
    store: MemoryStore
    policy: MemoryPolicy = field(default_factory=MemoryPolicy)
    context_builder: MemoryContextBuilder = field(default_factory=MemoryContextBuilder)
    extractor: MemoryExtractor | None = None
    events: EventSink | None = None

    @property
    def storage(self) -> StorageBackend:
        return self.store.storage

    @property
    def embeddings(self) -> EmbeddingProvider:
        return self.store.embeddings

    async def remember(
        self,
        text: str,
        *,
        namespace: str | None = None,
        scope: MemoryScope = MemoryScope.AGENT,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
        source_session_id: str | None = None,
        source_turn_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryRecord:
        return await self.store.add(
            namespace or self.policy.namespace,
            text,
            metadata=metadata,
            scope=scope,
            confidence=confidence,
            source_session_id=source_session_id,
            source_turn_id=source_turn_id,
            expires_at=expires_at,
        )

    async def context_for(
        self,
        query: str,
        *,
        session_id: str,
        namespace: str | None = None,
    ) -> MemoryContext:
        if not self.policy.enabled or self.policy.retrieval_limit == 0:
            return MemoryContext(content="", memories=[])
        retrieved = await self.store.retrieve(
            namespace or self.policy.namespace,
            query,
            limit=None,
            scopes=self.policy.scopes,
        )
        filtered = [
            item
            for item in retrieved
            if item.score >= self.policy.min_score
            and (
                item.memory.scope != MemoryScope.SESSION
                or item.memory.source_session_id == session_id
            )
        ]
        context = self.context_builder.build(
            filtered[: self.policy.retrieval_limit],
            max_chars=self.policy.max_context_chars,
        )
        if context.memories:
            await self.storage.mark_memories_used(context.memory_ids)
            if self.events is not None:
                await self.events.emit(
                    "memory.context.loaded",
                    session_id=session_id,
                    memory_ids=context.memory_ids,
                    count=len(context.memories),
                    namespace=namespace or self.policy.namespace,
                )
        return context

    async def capture(
        self,
        exchange: MemoryExchange,
        *,
        namespace: str | None = None,
    ) -> list[MemoryRecord]:
        if not self.policy.enabled or not self.policy.auto_capture:
            return []
        extractor = self.extractor or NoopMemoryExtractor()
        candidates = await extractor.extract(exchange)
        records: list[MemoryRecord] = []
        for candidate in candidates:
            records.append(
                await self.remember(
                    candidate.text,
                    namespace=namespace,
                    scope=candidate.scope,
                    confidence=candidate.confidence,
                    metadata=candidate.metadata,
                    source_session_id=exchange.session_id,
                    source_turn_id=_optional_str(exchange.metadata.get("source_turn_id")),
                    expires_at=candidate.expires_at,
                )
            )
        if records and self.events is not None:
            await self.events.emit(
                "memory.capture.completed",
                session_id=exchange.session_id,
                memory_ids=[record.id for record in records],
                count=len(records),
                namespace=namespace or self.policy.namespace,
            )
        return records


def _single_line(text: str) -> str:
    return redact(" ".join(text.split()))


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
