from __future__ import annotations

from abc import ABC, abstractmethod

from harness.observability.events import Event
from harness.schemas import (
    ArtifactRecord,
    MemoryRecord,
    MemoryScope,
    Session,
    ToolCall,
    ToolCallRecord,
    ToolResult,
    TurnRecord,
)


class StorageBackend(ABC):
    async def close(self) -> None:
        return None

    @abstractmethod
    async def migrate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def create_session(self, session: Session) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_session(self, session_id: str) -> Session | None:
        raise NotImplementedError

    @abstractmethod
    async def list_sessions(self, limit: int | None = 100) -> list[Session]:
        raise NotImplementedError

    @abstractmethod
    async def try_acquire_session_lease(
        self,
        session_id: str,
        owner_id: str,
        *,
        ttl_seconds: float,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def refresh_session_lease(
        self,
        session_id: str,
        owner_id: str,
        *,
        ttl_seconds: float,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def release_session_lease(self, session_id: str, owner_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def save_turn(self, turn: TurnRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_turns(self, session_id: str, limit: int | None = 100) -> list[TurnRecord]:
        raise NotImplementedError

    @abstractmethod
    async def record_event(self, event: Event) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_events(
        self, session_id: str | None = None, limit: int | None = 100
    ) -> list[Event]:
        raise NotImplementedError

    @abstractmethod
    async def record_tool_call(self, call: ToolCall, result: ToolResult) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_tool_calls(
        self, session_id: str | None = None, limit: int | None = 100
    ) -> list[ToolCallRecord]:
        raise NotImplementedError

    @abstractmethod
    async def save_artifact(self, artifact: ArtifactRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_artifacts(
        self,
        session_id: str | None = None,
        tool_call_id: str | None = None,
        limit: int | None = 100,
    ) -> list[ArtifactRecord]:
        raise NotImplementedError

    @abstractmethod
    async def save_memory(self, memory: MemoryRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_memories(
        self,
        namespace: str,
        *,
        scopes: list[MemoryScope] | None = None,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    async def mark_memories_used(self, memory_ids: list[str]) -> None:
        raise NotImplementedError
