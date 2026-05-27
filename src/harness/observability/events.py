from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from harness.schemas import utc_now


class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    event_type: str
    ts: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)
