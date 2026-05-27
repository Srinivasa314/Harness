from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def to_json(value: Any) -> str:
    return json.dumps(value, default=_default, sort_keys=True)


def from_json(value: str | bytes | None, default: Any = None) -> Any:
    if value is None:
        return default
    if not isinstance(value, str | bytes):
        return value
    return json.loads(value)


def _default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
