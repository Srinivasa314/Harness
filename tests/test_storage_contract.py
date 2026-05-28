from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import aiosqlite
import anyio
import pytest

from harness.observability.events import Event
from harness.schemas import (
    ArtifactRecord,
    MemoryRecord,
    MemoryScope,
    Session,
    ToolCall,
    ToolResult,
    TurnRecord,
)
from harness.storage import PostgresStorage, SQLiteStorage, StorageBackend

pytestmark = pytest.mark.anyio


async def _sqlite_storage(tmp_path) -> StorageBackend:
    storage = SQLiteStorage(tmp_path / "contract.sqlite3")
    await storage.migrate()
    return storage


async def _postgres_storage() -> StorageBackend:
    dsn = os.environ.get("HARNESS_POSTGRES_DSN")
    if not dsn:
        pytest.skip("HARNESS_POSTGRES_DSN is not set")
    storage = PostgresStorage(dsn)
    await storage.migrate()
    return storage


async def _exercise_storage_contract(storage: StorageBackend) -> None:
    session = Session(metadata={"suite": "contract"})
    await storage.create_session(session)
    assert (await storage.get_session(session.id)) == session
    assert await storage.try_acquire_session_lease(session.id, "owner-a", ttl_seconds=300)
    assert not await storage.try_acquire_session_lease(session.id, "owner-b", ttl_seconds=300)
    await storage.release_session_lease(session.id, "owner-b")
    assert not await storage.try_acquire_session_lease(session.id, "owner-b", ttl_seconds=300)
    await storage.release_session_lease(session.id, "owner-a")
    assert await storage.try_acquire_session_lease(session.id, "owner-b", ttl_seconds=300)
    assert await storage.refresh_session_lease(session.id, "owner-b", ttl_seconds=300)
    await storage.release_session_lease(session.id, "owner-b")
    assert await storage.try_acquire_session_lease(session.id, "owner-stale", ttl_seconds=0.01)
    await anyio.sleep(0.02)
    assert not await storage.refresh_session_lease(
        session.id,
        "owner-stale",
        ttl_seconds=0.01,
    )
    assert await storage.try_acquire_session_lease(
        session.id,
        "owner-recovered",
        ttl_seconds=0.01,
    )
    await storage.release_session_lease(session.id, "owner-recovered")

    secret_session = Session(metadata={"api_key": "sk-session-storage-secret"})
    await storage.create_session(secret_session)
    fetched_secret_session = await storage.get_session(secret_session.id)
    assert fetched_secret_session is not None
    assert fetched_secret_session.metadata["api_key"] == "[REDACTED]"

    await storage.save_turn(TurnRecord(session_id=session.id, role="user", content="hello"))
    await storage.save_turn(TurnRecord(session_id=session.id, role="assistant", content="hi"))
    turns = await storage.list_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant"]

    await storage.save_turn(
        TurnRecord(
            session_id=session.id,
            role="user",
            content="api_key is sk-storage-turn-secret",
            metadata={"api_key": "sk-storage-turn-metadata"},
        )
    )
    saved_turns = await storage.list_turns(session.id)
    secret_turn = saved_turns[-1]
    assert "sk-storage-turn-secret" not in secret_turn.content
    assert secret_turn.metadata["api_key"] == "[REDACTED]"

    await storage.record_event(
        Event(session_id=session.id, event_type="demo", payload={"ok": True})
    )
    events = await storage.list_events(session_id=session.id)
    assert events[0].payload == {"ok": True}

    await storage.record_event(
        Event(
            session_id=session.id,
            event_type="secret",
            payload={
                "api_key": "sk-storage-event-secret",
                "message": "sk-storage-event-secret",
            },
        )
    )
    [secret_event, *_] = await storage.list_events(session_id=session.id)
    assert secret_event.payload["api_key"] == "[REDACTED]"
    assert "sk-storage-event-secret" not in str(secret_event.model_dump(mode="json"))

    call = ToolCall(session_id=session.id, name="tool.echo", arguments={"message": "hello"})
    result = ToolResult(call_id=call.call_id, name=call.name, status="ok", output={"ok": True})
    await storage.record_tool_call(call, result)
    retry_result = ToolResult(
        call_id=call.call_id,
        name=call.name,
        status="error",
        output={"ok": False},
        error="retry failed",
    )
    await storage.record_tool_call(call, retry_result)
    tool_calls = await storage.list_tool_calls(session_id=session.id)
    assert tool_calls[0]["tool_name"] == "tool.echo"
    assert tool_calls[0]["status"] == "error"
    assert tool_calls[0]["input"] == {"message": "hello"}
    assert tool_calls[0]["output"] == {"ok": False}
    assert tool_calls[0]["error"] == "retry failed"

    secret_call = ToolCall(
        session_id=session.id,
        name="tool.secret",
        arguments={"clientSecret": "plain-secret", "message": "plain-secret"},
    )
    await storage.record_tool_call(
        secret_call,
        ToolResult(
            call_id=secret_call.call_id,
            name=secret_call.name,
            status="ok",
            output={"clientSecret": "generated-secret", "message": "generated-secret"},
        ),
    )
    [stored_secret_call, *_] = await storage.list_tool_calls(session_id=session.id, limit=1)
    assert "plain-secret" not in str(stored_secret_call.model_dump(mode="json"))
    assert "generated-secret" not in str(stored_secret_call.model_dump(mode="json"))

    artifact = ArtifactRecord(
        session_id=session.id,
        tool_call_id=call.call_id,
        path="/tmp/artifact.txt",
        media_type="text/plain",
        size_bytes=5,
        metadata={"api_key": "sk-storage-secret"},
    )
    await storage.save_artifact(artifact)
    artifacts = await storage.list_artifacts(session_id=session.id, tool_call_id=call.call_id)
    assert artifacts[0].path == "/tmp/artifact.txt"
    assert artifacts[0].metadata["api_key"] == "[REDACTED]"

    memory = MemoryRecord(
        namespace="project",
        text="hello",
        embedding=[1.0],
        metadata={"n": 1},
        scope=MemoryScope.SESSION,
        confidence=0.9,
        source_session_id=session.id,
    )
    await storage.save_memory(memory)
    memories = await storage.list_memories("project", scopes=[MemoryScope.SESSION])
    assert memories[0].metadata == {"n": 1}
    assert memories[0].scope == MemoryScope.SESSION
    assert memories[0].source_session_id == session.id
    await storage.mark_memories_used([memory.id])
    used_memories = await storage.list_memories("project", scopes=[MemoryScope.SESSION])
    assert used_memories[0].last_used_at is not None

    secret_memory = MemoryRecord(
        namespace="project",
        text="api_key is sk-storage-memory-secret",
        embedding=[1.0],
        metadata={"api_key": "sk-storage-memory-metadata"},
    )
    await storage.save_memory(secret_memory)
    [stored_secret_memory, *_] = await storage.list_memories("project")
    dumped_memory = stored_secret_memory.model_dump(mode="json")
    assert "sk-storage-memory-secret" not in str(dumped_memory)
    assert "sk-storage-memory-metadata" not in str(dumped_memory)
    assert stored_secret_memory.metadata["api_key"] == "[REDACTED]"

async def _exercise_implicit_session_creation(storage: StorageBackend) -> None:
    session_id = f"implicit-{uuid4().hex}"
    await storage.save_turn(TurnRecord(session_id=session_id, role="user", content="hello"))
    await storage.record_event(Event(session_id=session_id, event_type="demo", payload={}))
    call = ToolCall(session_id=session_id, name="tool.echo", arguments={})
    await storage.record_tool_call(
        call,
        ToolResult(call_id=call.call_id, name=call.name, status="ok"),
    )
    await storage.save_artifact(ArtifactRecord(session_id=session_id, path="/tmp/out.txt"))

    session = await storage.get_session(session_id)

    assert session is not None
    assert session.metadata == {"created_by": "implicit"}
    assert await storage.list_turns(session_id)
    assert await storage.list_events(session_id=session_id)
    assert await storage.list_tool_calls(session_id=session_id)
    assert await storage.list_artifacts(session_id=session_id)


async def _exercise_turn_window(storage: StorageBackend) -> None:
    session = Session()
    await storage.create_session(session)
    for index in range(105):
        await storage.save_turn(
            TurnRecord(
                session_id=session.id,
                role="user",
                content=f"turn-{index}",
                created_at=datetime(2030, 1, 1, tzinfo=UTC) - timedelta(seconds=index),
            )
        )

    turns = await storage.list_turns(session.id, limit=100)

    assert len(turns) == 100
    assert turns[0].content == "turn-5"
    assert turns[-1].content == "turn-104"

    all_turns = await storage.list_turns(session.id, limit=None)
    assert len(all_turns) == 105
    assert all_turns[0].content == "turn-0"
    assert all_turns[-1].content == "turn-104"


async def _exercise_turn_window_insertion_tie_breaker(storage: StorageBackend) -> None:
    session = Session()
    await storage.create_session(session)
    created_at = datetime(2030, 1, 1, tzinfo=UTC)
    for turn_id in ["turn-c", "turn-a", "turn-b"]:
        await storage.save_turn(
            TurnRecord(
                id=turn_id,
                session_id=session.id,
                role="user",
                content=turn_id,
                created_at=created_at,
            )
        )

    turns = await storage.list_turns(session.id, limit=2)

    assert [turn.id for turn in turns] == ["turn-a", "turn-b"]


async def _exercise_list_filter_order_and_limits(storage: StorageBackend) -> None:
    session_a = Session()
    session_b = Session()
    await storage.create_session(session_a)
    await storage.create_session(session_b)
    unique = uuid4().hex
    base_time = datetime(2100, 1, 2, tzinfo=UTC) + timedelta(
        microseconds=int(unique[:8], 16) % 1_000_000
    )
    for index, session in enumerate([session_a, session_b, session_a]):
        await storage.record_event(
            Event(
                session_id=session.id,
                event_type=f"event-{index}",
                ts=base_time + timedelta(seconds=index),
            )
        )

    events = await storage.list_events(session_id=session_a.id, limit=1)
    assert [event.event_type for event in events] == ["event-2"]

    for index, session in enumerate([session_a, session_b, session_a]):
        call = ToolCall(
            call_id=f"call-{session.id}-{index}",
            session_id=session.id,
            name="tool.echo",
            arguments={"index": index},
        )
        await storage.record_tool_call(
            call,
            ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="ok",
                output={"index": index},
                started_at=base_time + timedelta(seconds=index),
                ended_at=base_time + timedelta(seconds=index + 1),
            ),
        )

    tool_calls = await storage.list_tool_calls(session_id=session_a.id, limit=1)
    assert [call.input["index"] for call in tool_calls] == [2]

    for index, session in enumerate([session_a, session_b, session_a]):
        await storage.save_artifact(
            ArtifactRecord(
                session_id=session.id,
                tool_call_id=f"call-{session.id}-{index}",
                path=f"/tmp/artifact-{index}.txt",
                created_at=base_time + timedelta(seconds=index),
            )
        )

    artifacts = await storage.list_artifacts(session_id=session_a.id, limit=1)
    assert [artifact.path for artifact in artifacts] == ["/tmp/artifact-2.txt"]

async def test_sqlite_storage_contract(tmp_path):
    storage = await _sqlite_storage(tmp_path)
    try:
        await _exercise_storage_contract(storage)
    finally:
        await storage.close()


async def test_sqlite_list_turns_returns_latest_turns_in_chronological_order(tmp_path):
    storage = await _sqlite_storage(tmp_path)
    try:
        await _exercise_turn_window(storage)
    finally:
        await storage.close()


async def test_sqlite_list_turns_uses_insertion_tie_breaker(tmp_path):
    storage = await _sqlite_storage(tmp_path)
    try:
        await _exercise_turn_window_insertion_tie_breaker(storage)
    finally:
        await storage.close()


async def test_sqlite_list_filter_order_and_limits(tmp_path):
    storage = await _sqlite_storage(tmp_path)
    try:
        await _exercise_list_filter_order_and_limits(storage)
    finally:
        await storage.close()


async def test_sqlite_migrate_records_schema_version(tmp_path):
    storage = SQLiteStorage(tmp_path / "contract.sqlite3")
    try:
        await storage.migrate()
        async with aiosqlite.connect(storage.path) as db:
            rows = await db.execute_fetchall("select version from schema_migrations")
    finally:
        await storage.close()

    assert [row[0] for row in rows] == ["0001_initial"]


async def test_sqlite_migrate_adds_session_leases_to_existing_initial_schema(tmp_path):
    path = tmp_path / "contract.sqlite3"
    async with aiosqlite.connect(path) as db:
        await db.executescript(
            """
            create table schema_migrations (
              version text primary key,
              applied_at text not null default current_timestamp
            );
            insert into schema_migrations (version) values ('0001_initial');
            create table sessions (
              id text primary key,
              created_at text not null,
              metadata text not null
            );
            """
        )
        await db.commit()
    storage = SQLiteStorage(path)
    try:
        await storage.migrate()
        async with aiosqlite.connect(path) as db:
            rows = await db.execute_fetchall(
                "select name from sqlite_master where type = 'table' and name = 'session_leases'"
            )
    finally:
        await storage.close()

    assert rows


async def test_sqlite_implicitly_creates_sessions_for_session_scoped_records(tmp_path):
    storage = await _sqlite_storage(tmp_path)
    try:
        await _exercise_implicit_session_creation(storage)
    finally:
        await storage.close()


@pytest.mark.e2e
@pytest.mark.postgres
async def test_postgres_storage_contract():
    storage = await _postgres_storage()
    try:
        await _exercise_storage_contract(storage)
    finally:
        await storage.close()


@pytest.mark.e2e
@pytest.mark.postgres
async def test_postgres_list_turns_returns_latest_turns_in_chronological_order():
    storage = await _postgres_storage()
    try:
        await _exercise_turn_window(storage)
    finally:
        await storage.close()


@pytest.mark.e2e
@pytest.mark.postgres
async def test_postgres_list_turns_uses_insertion_tie_breaker():
    storage = await _postgres_storage()
    try:
        await _exercise_turn_window_insertion_tie_breaker(storage)
    finally:
        await storage.close()


@pytest.mark.e2e
@pytest.mark.postgres
async def test_postgres_list_filter_order_and_limits():
    storage = await _postgres_storage()
    try:
        await _exercise_list_filter_order_and_limits(storage)
    finally:
        await storage.close()


@pytest.mark.e2e
@pytest.mark.postgres
async def test_postgres_implicitly_creates_sessions_for_session_scoped_records():
    storage = await _postgres_storage()
    try:
        await _exercise_implicit_session_creation(storage)
    finally:
        await storage.close()
