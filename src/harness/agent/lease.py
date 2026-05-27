from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from uuid import uuid4

from harness.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class SessionLeaseError(RuntimeError):
    pass


@dataclass
class SessionLease:
    session_id: str
    owner_id: str
    provider: SessionLeaseProvider
    released: bool = False

    async def release(self) -> None:
        if self.released:
            return
        await self.provider.release_session(self.session_id, self.owner_id)
        self.released = True

    async def wait_lost(self) -> None:
        await self.provider.wait_session_lost(self.session_id, self.owner_id)

    async def __aenter__(self) -> SessionLease:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


class SessionLeaseProvider:
    async def enter_session(self, session_id: str) -> SessionLease:
        raise NotImplementedError

    async def wait_session_lost(self, session_id: str, owner_id: str) -> None:
        _ = session_id, owner_id
        await asyncio.Event().wait()

    async def release_session(self, session_id: str, owner_id: str) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class StorageSessionLeaseProvider(SessionLeaseProvider):
    def __init__(
        self,
        storage: StorageBackend,
        *,
        owner_id: str | None = None,
        ttl_seconds: float = 300,
        heartbeat_seconds: float | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if heartbeat_seconds is not None and heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self.storage = storage
        self.owner_id = owner_id or str(uuid4())
        self.ttl_seconds = ttl_seconds
        self.heartbeat_seconds = heartbeat_seconds or max(1.0, ttl_seconds / 3)
        self._active: set[str] = set()
        self._heartbeat_tasks: dict[str, asyncio.Task[None]] = {}
        self._lost_events: dict[str, asyncio.Event] = {}
        self._lost_errors: dict[str, SessionLeaseError] = {}

    async def enter_session(self, session_id: str) -> SessionLease:
        if session_id in self._active:
            raise SessionLeaseError(f"Session is already active in this runtime: {session_id}")
        acquired = await self.storage.try_acquire_session_lease(
            session_id,
            self.owner_id,
            ttl_seconds=self.ttl_seconds,
        )
        if not acquired:
            raise SessionLeaseError(f"Session is already active: {session_id}")
        self._active.add(session_id)
        self._lost_events[session_id] = asyncio.Event()
        self._heartbeat_tasks[session_id] = asyncio.create_task(
            self._heartbeat_session(session_id)
        )
        return SessionLease(session_id=session_id, owner_id=self.owner_id, provider=self)

    async def wait_session_lost(self, session_id: str, owner_id: str) -> None:
        if owner_id != self.owner_id:
            await asyncio.Event().wait()
        event = self._lost_events.get(session_id)
        if event is None:
            await asyncio.Event().wait()
        assert event is not None
        await event.wait()
        error = self._lost_errors.get(session_id)
        if error is not None:
            raise error

    async def release_session(self, session_id: str, owner_id: str) -> None:
        if owner_id != self.owner_id:
            return
        task = self._heartbeat_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        try:
            await self.storage.release_session_lease(session_id, owner_id)
        finally:
            self._active.discard(session_id)
            self._lost_events.pop(session_id, None)
            self._lost_errors.pop(session_id, None)

    async def close(self) -> None:
        for session_id in list(self._active):
            await self.release_session(session_id, self.owner_id)

    async def _heartbeat_session(self, session_id: str) -> None:
        try:
            while session_id in self._active:
                await asyncio.sleep(self.heartbeat_seconds)
                if session_id not in self._active:
                    return
                refreshed = await self.storage.refresh_session_lease(
                    session_id,
                    self.owner_id,
                    ttl_seconds=self.ttl_seconds,
                )
                if not refreshed:
                    error = SessionLeaseError(f"Session lease lost ownership: {session_id}")
                    self._lost_errors[session_id] = error
                    self._lost_events[session_id].set()
                    logger.error("Session lease heartbeat lost ownership for %s", session_id)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = SessionLeaseError(f"Session lease heartbeat failed: {session_id}")
            error.__cause__ = exc
            self._lost_errors[session_id] = error
            self._lost_events[session_id].set()
            logger.exception("Session lease heartbeat failed for %s", session_id)
