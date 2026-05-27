from __future__ import annotations

from harness.observability.sink import EventSink
from harness.schemas import Session, TurnRecord
from harness.storage.base import StorageBackend
from harness.tools.redaction import redact, redact_with_detected_secrets, sensitive_values


class AgentSessionManager:
    def __init__(self, storage: StorageBackend) -> None:
        self.storage = storage
        self.events = EventSink(storage)

    async def create(self, metadata: dict | None = None) -> Session:
        session = Session(metadata=redact_with_detected_secrets(metadata or {}))
        await self.storage.create_session(session)
        await self.events.emit("session.created", session_id=session.id, metadata=session.metadata)
        return session

    async def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> TurnRecord:
        metadata = metadata or {}
        turn_secrets = sensitive_values(metadata)
        turn = TurnRecord(
            session_id=session_id,
            role=role,
            content=redact(content, turn_secrets),
            metadata=redact(metadata, turn_secrets),
        )
        await self.storage.save_turn(turn)
        await self.events.emit(
            "session.turn.created",
            session_id=session_id,
            role=role,
            turn_id=turn.id,
        )
        return turn
