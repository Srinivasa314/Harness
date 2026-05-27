from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from harness.observability.events import Event

if TYPE_CHECKING:
    from harness.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class EventSink:
    def __init__(self, storage: StorageBackend, *, fail_open: bool = True) -> None:
        self.storage = storage
        self.fail_open = fail_open

    async def emit(
        self,
        event_type: str,
        *,
        session_id: str | None = None,
        **payload: object,
    ) -> Event:
        from harness.tools.redaction import redact_with_detected_secrets

        event = Event(
            session_id=session_id,
            event_type=event_type,
            payload=redact_with_detected_secrets(dict(payload)),
        )
        try:
            await self.storage.record_event(event)
        except Exception:
            if not self.fail_open:
                raise
            logger.exception("Failed to persist observability event %s", event_type)
        return event
