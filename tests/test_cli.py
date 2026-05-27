from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta

import pytest

from harness.agent import AgentSessionManager
from harness.cli import (
    doctor,
    export_session,
    run_agent,
)
from harness.dashboard.app import is_loopback_host
from harness.models import ModelMessage, ModelProvider, ModelResponse
from harness.observability.events import Event
from harness.schemas import (
    ArtifactRecord,
    MemoryRecord,
    MemoryScope,
    ToolCall,
    ToolResult,
    TurnRecord,
)
from harness.storage import SQLiteStorage

pytestmark = pytest.mark.anyio


class EchoAgentModel(ModelProvider):
    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        prompt = messages[-1].content
        return ModelResponse(content=json.dumps({"final": f"echo: {prompt}"}))


class UppercaseToolAgentModel(ModelProvider):
    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        tool_messages = [message for message in messages if message.role == "tool"]
        if not tool_messages:
            prompt = messages[-1].content
            return ModelResponse(
                content=json.dumps(
                    {
                        "tool_calls": [
                            {
                                "name": "text.uppercase",
                                "arguments": {"text": prompt},
                            }
                        ]
                    }
                )
            )
        result = json.loads(tool_messages[-1].content)[0]["output"]["value"]
        return ModelResponse(content=json.dumps({"final": result}))


async def test_export_session_writes_observability_payload(tmp_path, monkeypatch):
    db_path = tmp_path / "harness.sqlite3"
    output_path = tmp_path / "session.json"
    monkeypatch.setenv("HARNESS_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("HARNESS_SQLITE_PATH", str(db_path))
    storage = SQLiteStorage(db_path)
    await storage.migrate()
    manager = AgentSessionManager(storage)
    session = await manager.create(metadata={"api_key": "sk-session-meta-secret"})
    await manager.add_turn(session.id, "user", "hello")
    call = ToolCall(session_id=session.id, name="tool.echo", arguments={"message": "hello"})
    await storage.record_tool_call(
        call,
        ToolResult(call_id=call.call_id, name=call.name, status="ok", output={"ok": True}),
    )
    await storage.save_turn(
        TurnRecord(session_id=session.id, role="assistant", content="api_key is sk-export-secret")
    )
    await storage.record_event(
        Event(session_id=session.id, event_type="raw", payload={"api_key": "sk-event-secret"})
    )
    await storage.save_artifact(
        ArtifactRecord(
            session_id=session.id,
            path="/tmp/raw-artifact.txt",
            metadata={"api_key": "sk-artifact-secret"},
        )
    )
    await storage.save_memory(
        MemoryRecord(
            namespace="default",
            text="session api_key is sk-memory-secret",
            embedding=[1.0],
            scope=MemoryScope.SESSION,
            source_session_id=session.id,
        )
    )
    await storage.save_memory(
        MemoryRecord(
            namespace="default",
            text="other session memory",
            embedding=[1.0],
            scope=MemoryScope.SESSION,
            source_session_id="other-session",
        )
    )
    await storage.save_memory(
        MemoryRecord(
            namespace="default",
            text="global review preference",
            embedding=[1.0],
            scope=MemoryScope.GLOBAL,
        )
    )

    exit_code = await export_session(session.id, output_path)

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["session"]["id"] == session.id
    assert payload["turns"][0]["content"] == "hello"
    exported = output_path.read_text()
    assert "sk-session-meta-secret" not in exported
    assert "sk-export-secret" not in exported
    assert "sk-event-secret" not in exported
    assert "sk-artifact-secret" not in exported
    assert payload["tool_calls"][0]["input"] == {"message": "hello"}
    assert payload["tool_calls"][0]["output"] == {"ok": True}
    assert payload["session"]["metadata"]["api_key"] == "[REDACTED]"
    assert [memory["scope"] for memory in payload["memories"]] == ["global", "session"]
    assert payload["memories"][0]["text"] == "global review preference"
    assert "other session memory" not in exported
    assert "sk-memory-secret" not in exported


async def test_export_session_includes_all_turns(tmp_path, monkeypatch):
    db_path = tmp_path / "harness.sqlite3"
    output_path = tmp_path / "session.json"
    monkeypatch.setenv("HARNESS_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("HARNESS_SQLITE_PATH", str(db_path))
    storage = SQLiteStorage(db_path)
    await storage.migrate()
    session = await AgentSessionManager(storage).create()
    base_time = datetime(2030, 1, 1, tzinfo=UTC)
    for index in range(101):
        await storage.save_turn(
            TurnRecord(
                session_id=session.id,
                role="user",
                content=f"turn-{index:03d}",
                created_at=base_time + timedelta(seconds=index),
            )
        )

    exit_code = await export_session(session.id, output_path)

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert len(payload["turns"]) == 101
    assert payload["turns"][0]["content"] == "turn-000"
    assert payload["turns"][-1]["content"] == "turn-100"


async def test_run_agent_cli_with_injected_model(tmp_path, monkeypatch):
    db_path = tmp_path / "harness.sqlite3"
    output_path = tmp_path / "agent.json"
    monkeypatch.setenv("HARNESS_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("HARNESS_SQLITE_PATH", str(db_path))

    exit_code = await run_agent("hello", output_path, model=EchoAgentModel())

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["final"] == "echo: hello"


async def test_run_agent_cli_with_configured_tool(tmp_path, monkeypatch):
    db_path = tmp_path / "harness.sqlite3"
    output_path = tmp_path / "agent.json"
    tool_script = tmp_path / "upper_tool.py"
    tools_path = tmp_path / "tools.json"
    tool_script.write_text(
        """
import json
import sys

payload = json.load(sys.stdin)
print(json.dumps({"value": payload["arguments"]["text"].upper()}))
""".strip()
    )
    tools_path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "text.uppercase",
                        "description": "Uppercase text",
                        "required_capabilities": ["text:uppercase"],
                        "execution_mode": "subprocess",
                        "subprocess_command": [sys.executable, str(tool_script)],
                    }
                ]
            }
        )
    )
    monkeypatch.setenv("HARNESS_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("HARNESS_SQLITE_PATH", str(db_path))

    exit_code = await run_agent(
        "hello",
        output_path,
        tools_path,
        ["text:uppercase"],
        model=UppercaseToolAgentModel(),
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["final"] == "HELLO"


async def test_run_agent_cli_uses_configured_codex_provider_without_command(tmp_path, monkeypatch):
    db_path = tmp_path / "harness.sqlite3"
    output_path = tmp_path / "agent.json"
    codex_script = tmp_path / "codex.py"
    codex_script.write_text("print('{\"final\": \"from codex\"}')")
    monkeypatch.setenv("HARNESS_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("HARNESS_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("HARNESS_MODEL_PROVIDER", "codex")
    monkeypatch.setenv("HARNESS_CODEX_COMMAND", json.dumps([sys.executable, str(codex_script)]))

    exit_code = await run_agent("hello", output_path)

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["final"] == "from codex"


def test_doctor_cli_writes_report(tmp_path):
    output_path = tmp_path / "doctor.json"

    exit_code = doctor(output_path)

    assert exit_code in {0, 1}
    payload = json.loads(output_path.read_text())
    assert "checks" in payload


def test_dashboard_remote_host_guard():
    assert is_loopback_host("127.0.0.1") is True
    assert is_loopback_host("localhost") is True
    assert is_loopback_host("0.0.0.0") is False
