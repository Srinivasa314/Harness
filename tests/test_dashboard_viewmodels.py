from __future__ import annotations

from datetime import UTC, datetime

import pytest

from harness.dashboard.viewmodels import (
    artifact_rows,
    event_rows,
    tool_call_rows,
    turn_rows,
)
from harness.observability.events import Event
from harness.schemas import ArtifactRecord, ToolCallRecord, TurnRecord


def test_dashboard_entrypoint_rejects_remote_host(monkeypatch):
    from harness.dashboard import app as dashboard_app

    monkeypatch.setattr("sys.argv", ["harness-dashboard", "--host", "0.0.0.0"])

    with pytest.raises(SystemExit, match="non-loopback"):
        dashboard_app.main()


def test_dashboard_registers_storage_shutdown(monkeypatch):
    from harness.dashboard import app as dashboard_app

    callbacks = []

    class FakeStorage:
        async def close(self) -> None:
            return None

    monkeypatch.setattr(dashboard_app, "create_storage", lambda _settings: FakeStorage())
    monkeypatch.setattr(dashboard_app.app, "on_shutdown", callbacks.append)

    dashboard_app.build_app()

    assert len(callbacks) == 1
    assert callbacks[0].__name__ == "close"


def test_tool_call_rows_filters_status():
    rows = tool_call_rows(
        [
            {"id": "1", "status": "ok"},
            {"id": "2", "status": "error"},
        ],
        status="error",
    )

    assert rows == [{"id": "2", "status": "error"}]


def test_event_rows_include_session_detail():
    rows = event_rows([Event(session_id="session-1", event_type="session.created")])

    assert rows[0]["session_detail"] == "/sessions/session-1"


def test_turn_rows_redact_secret_like_content():
    rows = turn_rows(
        [TurnRecord(session_id="session-1", role="user", content="api_key is sk-dashboard-secret")]
    )

    assert "sk-dashboard-secret" not in rows[0]["content"]
    assert "[REDACTED]" in rows[0]["content"]


def test_artifact_rows_redact_metadata():
    rows = artifact_rows(
        [ArtifactRecord(path="/tmp/raw.txt", metadata={"api_key": "sk-artifact-secret"})]
    )

    assert "sk-artifact-secret" not in str(rows)
    assert rows[0]["metadata"]["api_key"] == "[REDACTED]"


def test_tool_call_rows_redact_model_records():
    rows = tool_call_rows(
        [
            ToolCallRecord(
                id="call-1",
                session_id="session-1",
                tool_name="tool",
                status="denied",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                ended_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
                input={"api_key": "sk-tool-secret"},
            )
        ]
    )

    assert "sk-tool-secret" not in str(rows)
    assert rows[0]["input"]["api_key"] == "[REDACTED]"
