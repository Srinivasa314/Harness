from __future__ import annotations

from collections import Counter

from harness.observability.events import Event
from harness.schemas import ArtifactRecord, Session, ToolCallRecord, TurnRecord
from harness.tools.redaction import redact


def session_options(sessions: list[Session]) -> dict[str, str]:
    return {"": "All sessions"} | {
        session.id: _session_label(session)
        for session in sessions
    }


def summary_cards(
    *,
    sessions: list[Session],
    events: list[Event],
    turns: list[TurnRecord],
    tool_calls: list[ToolCallRecord],
    artifacts: list[ArtifactRecord],
) -> list[dict[str, str]]:
    error_count = sum(1 for call in tool_calls if call.status in {"error", "denied", "timeout"})
    return [
        {"label": "Sessions", "value": str(len(sessions))},
        {"label": "Events", "value": str(len(events))},
        {"label": "Turns", "value": str(len(turns))},
        {"label": "Tool Calls", "value": str(len(tool_calls))},
        {"label": "Tool Issues", "value": str(error_count)},
        {"label": "Artifacts", "value": str(len(artifacts))},
    ]


def session_summary_rows(
    sessions: list[Session],
    *,
    events: list[Event],
    turns: list[TurnRecord],
    tool_calls: list[ToolCallRecord],
    artifacts: list[ArtifactRecord],
) -> list[dict]:
    events_by_session = Counter(event.session_id for event in events if event.session_id)
    turns_by_session = Counter(turn.session_id for turn in turns)
    calls_by_session = Counter(call.session_id for call in tool_calls)
    issues_by_session = Counter(
        call.session_id for call in tool_calls if call.status in {"error", "denied", "timeout"}
    )
    artifacts_by_session = Counter(artifact.session_id for artifact in artifacts)
    return [
        {
            "id": session.id,
            "created_at": session.created_at.isoformat(),
            "metadata": redact(session.metadata),
            "events": events_by_session[session.id],
            "turns": turns_by_session[session.id],
            "tool_calls": calls_by_session[session.id],
            "tool_issues": issues_by_session[session.id],
            "artifacts": artifacts_by_session[session.id],
            "detail": f"/sessions/{session.id}",
        }
        for session in sessions
    ]


def tool_status_rows(tool_calls: list[ToolCallRecord]) -> list[dict[str, str | int]]:
    counts = Counter(call.status for call in tool_calls)
    return [
        {"status": status, "count": counts[status]}
        for status in sorted(counts)
    ]


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


def _session_label(session: Session) -> str:
    metadata_name = session.metadata.get("name") or session.metadata.get("example")
    prefix = f"{metadata_name} " if isinstance(metadata_name, str) else ""
    return f"{prefix}{session.id[:8]} ({session.created_at.isoformat()})"
