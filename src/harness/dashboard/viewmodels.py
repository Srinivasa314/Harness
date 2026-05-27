from __future__ import annotations

from harness.observability.events import Event
from harness.schemas import ArtifactRecord, ToolCallRecord, TurnRecord
from harness.tools.redaction import redact


def event_rows(events: list[Event]) -> list[dict]:
    return [
        {
            "ts": event.ts.isoformat(),
            "event_type": event.event_type,
            "session_id": event.session_id or "",
            "session_detail": f"/sessions/{event.session_id}" if event.session_id else "",
        }
        for event in events
    ]


def turn_rows(turns: list[TurnRecord]) -> list[dict]:
    return [redact(turn.model_dump(mode="json")) for turn in turns]


def artifact_rows(artifacts: list[ArtifactRecord]) -> list[dict]:
    return [redact(artifact.model_dump(mode="json")) for artifact in artifacts]


def tool_call_rows(
    tool_calls: list[ToolCallRecord] | list[dict],
    *,
    status: str | None = None,
) -> list[dict]:
    rows = [
        redact(row.model_dump(mode="json") if isinstance(row, ToolCallRecord) else row)
        for row in tool_calls
    ]
    if status:
        return [row for row in rows if row.get("status") == status]
    return rows
