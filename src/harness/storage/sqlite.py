from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from harness.observability.events import Event
from harness.schemas import (
    ArtifactRecord,
    MemoryRecord,
    MemoryScope,
    Session,
    ToolCall,
    ToolCallRecord,
    ToolResult,
    TurnRecord,
    utc_now,
)
from harness.storage.base import StorageBackend
from harness.storage.serde import from_json, to_json
from harness.tools.redaction import redact, redact_with_detected_secrets, sensitive_values

SCHEMA_VERSION = "0001_initial"


class SQLiteStorage(StorageBackend):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def migrate(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        async with aiosqlite.connect(self.path) as db:
            self.path.chmod(0o600)
            await db.execute("pragma foreign_keys = on")
            await db.execute("begin immediate")
            await db.execute(
                """
                create table if not exists schema_migrations (
                  version text primary key,
                  applied_at text not null default current_timestamp
                )
                """
            )
            rows = await db.execute_fetchall("select version from schema_migrations")
            applied = {row[0] for row in rows}
            if SCHEMA_VERSION not in applied:
                await db.executescript(
                    """
                create table if not exists sessions (
                  id text primary key,
                  created_at text not null,
                  metadata text not null
                );
                create table if not exists session_leases (
                  session_id text primary key,
                  owner_id text not null,
                  acquired_at text not null,
                  foreign key(session_id) references sessions(id)
                );
                create table if not exists events (
                  id text primary key,
                  session_id text,
                  event_type text not null,
                  ts text not null,
                  payload text not null,
                  foreign key(session_id) references sessions(id)
                );
                create index if not exists idx_events_session_ts on events(session_id, ts);
                create table if not exists turns (
                  turn_sequence integer primary key autoincrement,
                  id text not null unique,
                  session_id text not null,
                  role text not null,
                  content text not null,
                  metadata text not null,
                  created_at text not null,
                  foreign key(session_id) references sessions(id)
                );
                create index if not exists idx_turns_session_created
                  on turns(session_id, turn_sequence);
                create table if not exists tool_calls (
                  id text primary key,
                  session_id text not null,
                  tool_name text not null,
                  status text not null,
                  started_at text not null,
                  ended_at text not null,
                  input text not null,
                  output text,
                  artifacts text not null default '[]',
                  error text,
                  foreign key(session_id) references sessions(id)
                );
                create index if not exists idx_tool_calls_session on tool_calls(session_id);
                create table if not exists artifacts (
                  id text primary key,
                  session_id text,
                  tool_call_id text,
                  path text not null,
                  media_type text,
                  size_bytes integer,
                  metadata text not null,
                  created_at text not null,
                  foreign key(session_id) references sessions(id)
                );
                create index if not exists idx_artifacts_session on artifacts(session_id);
                create index if not exists idx_artifacts_tool_call on artifacts(tool_call_id);
                create table if not exists memories (
                  id text primary key,
                  namespace text not null,
                  text text not null,
                  embedding text not null,
                  embedding_provider text,
                  embedding_model text,
                  embedding_dimensions integer,
                  metadata text not null,
                  scope text not null,
                  importance real not null default 1.0,
                  confidence real not null,
                  source_session_id text,
                  source_turn_id text,
                  updated_at text not null,
                  last_used_at text,
                  expires_at text,
                  created_at text not null
                );
                create index if not exists idx_memories_namespace on memories(namespace);
                create index if not exists idx_memories_namespace_scope
                  on memories(namespace, scope);
                """
                )
                await db.execute(
                    "insert or ignore into schema_migrations (version) values (?)",
                    (SCHEMA_VERSION,),
                )
            await db.execute(
                """
                create table if not exists session_leases (
                  session_id text primary key,
                  owner_id text not null,
                  acquired_at text not null,
                  foreign key(session_id) references sessions(id)
                )
                """
            )
            await db.commit()

    async def create_session(self, session: Session) -> None:
        metadata = redact_with_detected_secrets(session.metadata)
        async with aiosqlite.connect(self.path) as db:
            await db.execute("pragma foreign_keys = on")
            await db.execute(
                "insert into sessions (id, created_at, metadata) values (?, ?, ?)",
                (session.id, session.created_at.isoformat(), to_json(metadata)),
            )
            await db.commit()

    async def _ensure_session(self, db: aiosqlite.Connection, session_id: str | None) -> None:
        if session_id is None:
            return
        session = Session(id=session_id, metadata={"created_by": "implicit"})
        await db.execute(
            "insert or ignore into sessions (id, created_at, metadata) values (?, ?, ?)",
            (session.id, session.created_at.isoformat(), to_json(redact(session.metadata))),
        )

    async def get_session(self, session_id: str) -> Session | None:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("select * from sessions where id = ?", (session_id,))
            row = await cursor.fetchone()
        if row is None:
            return None
        return Session(
            id=row["id"],
            created_at=row["created_at"],
            metadata=from_json(row["metadata"], {}),
        )

    async def list_sessions(self, limit: int | None = 100) -> list[Session]:
        query = "select * from sessions order by created_at desc"
        args: list[int] = []
        if limit is not None:
            query += " limit ?"
            args.append(limit)
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(query, args)
        return [
            Session(
                id=row["id"],
                created_at=row["created_at"],
                metadata=from_json(row["metadata"], {}),
            )
            for row in rows
        ]

    async def try_acquire_session_lease(
        self,
        session_id: str,
        owner_id: str,
        *,
        ttl_seconds: float,
    ) -> bool:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("pragma foreign_keys = on")
            await db.execute("begin immediate")
            await self._ensure_session(db, session_id)
            cutoff = (utc_now() - timedelta(seconds=ttl_seconds)).isoformat()
            await db.execute(
                """
                delete from session_leases
                where session_id = ? and acquired_at <= ?
                """,
                (session_id, cutoff),
            )
            cursor = await db.execute(
                """
                insert or ignore into session_leases (session_id, owner_id, acquired_at)
                values (?, ?, ?)
                """,
                (session_id, owner_id, utc_now().isoformat()),
            )
            await db.commit()
            return cursor.rowcount == 1

    async def refresh_session_lease(
        self,
        session_id: str,
        owner_id: str,
        *,
        ttl_seconds: float,
    ) -> bool:
        now = utc_now()
        cutoff = (now - timedelta(seconds=ttl_seconds)).isoformat()
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                """
                update session_leases
                set acquired_at = ?
                where session_id = ? and owner_id = ? and acquired_at > ?
                """,
                (now.isoformat(), session_id, owner_id, cutoff),
            )
            await db.commit()
            return cursor.rowcount == 1

    async def release_session_lease(self, session_id: str, owner_id: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "delete from session_leases where session_id = ? and owner_id = ?",
                (session_id, owner_id),
            )
            await db.commit()

    async def save_turn(self, turn: TurnRecord) -> None:
        turn_secrets = sensitive_values(turn.metadata)
        async with aiosqlite.connect(self.path) as db:
            await db.execute("pragma foreign_keys = on")
            await self._ensure_session(db, turn.session_id)
            await db.execute(
                """
                insert into turns (id, session_id, role, content, metadata, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.id,
                    turn.session_id,
                    turn.role,
                    redact(turn.content, turn_secrets),
                    to_json(redact(turn.metadata, turn_secrets)),
                    turn.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def list_turns(self, session_id: str, limit: int | None = 100) -> list[TurnRecord]:
        if limit is None:
            query = """
                select * from turns
                where session_id = ?
                order by turn_sequence asc
                """
            args: tuple[Any, ...] = (session_id,)
        else:
            query = """
                select * from (
                  select * from turns
                  where session_id = ?
                  order by turn_sequence desc
                  limit ?
                )
                order by turn_sequence asc
                """
            args = (session_id, limit)
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(query, args)
        return [
            TurnRecord(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                metadata=from_json(row["metadata"], {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def record_event(self, event: Event) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute("pragma foreign_keys = on")
            await self._ensure_session(db, event.session_id)
            await db.execute(
                """
                insert into events (id, session_id, event_type, ts, payload)
                values (?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.session_id,
                    event.event_type,
                    event.ts.isoformat(),
                    to_json(redact_with_detected_secrets(event.payload)),
                ),
            )
            await db.commit()

    async def list_events(
        self, session_id: str | None = None, limit: int | None = 100
    ) -> list[Event]:
        query = "select * from events"
        args: list[Any] = []
        if session_id is not None:
            query += " where session_id = ?"
            args.append(session_id)
        query += " order by ts desc"
        if limit is not None:
            query += " limit ?"
            args.append(limit)
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(query, tuple(args))
        return [
            Event(
                id=row["id"],
                session_id=row["session_id"],
                event_type=row["event_type"],
                ts=row["ts"],
                payload=from_json(row["payload"], {}),
            )
            for row in rows
        ]

    async def record_tool_call(self, call: ToolCall, result: ToolResult) -> None:
        redaction_secrets = {
            **sensitive_values(call.arguments),
            **sensitive_values(result.model_dump(mode="json")),
        }
        async with aiosqlite.connect(self.path) as db:
            await db.execute("pragma foreign_keys = on")
            await self._ensure_session(db, call.session_id)
            await db.execute(
                """
                insert into tool_calls
                (
                  id, session_id, tool_name, status, started_at, ended_at,
                  input, output, artifacts, error
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                  session_id = excluded.session_id,
                  tool_name = excluded.tool_name,
                  status = excluded.status,
                  started_at = excluded.started_at,
                  ended_at = excluded.ended_at,
                  input = excluded.input,
                  output = excluded.output,
                  artifacts = excluded.artifacts,
                  error = excluded.error
                """,
                (
                    call.call_id,
                    call.session_id,
                    call.name,
                    result.status,
                    result.started_at.isoformat(),
                    result.ended_at.isoformat(),
                    to_json(redact(call.arguments, redaction_secrets)),
                    to_json(redact(result.output, redaction_secrets)),
                    to_json(redact(result.artifacts, redaction_secrets)),
                    redact(result.error, redaction_secrets),
                ),
            )
            await db.commit()

    async def list_tool_calls(
        self, session_id: str | None = None, limit: int | None = 100
    ) -> list[ToolCallRecord]:
        query = "select * from tool_calls"
        args: list[Any] = []
        if session_id is not None:
            query += " where session_id = ?"
            args.append(session_id)
        query += " order by started_at desc"
        if limit is not None:
            query += " limit ?"
            args.append(limit)
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(query, tuple(args))
        return [_tool_call_row(row) for row in rows]

    async def save_artifact(self, artifact: ArtifactRecord) -> None:
        metadata = redact_with_detected_secrets(artifact.metadata)
        async with aiosqlite.connect(self.path) as db:
            await db.execute("pragma foreign_keys = on")
            await self._ensure_session(db, artifact.session_id)
            await db.execute(
                """
                insert into artifacts
                (id, session_id, tool_call_id, path, media_type, size_bytes, metadata, created_at)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.id,
                    artifact.session_id,
                    artifact.tool_call_id,
                    artifact.path,
                    artifact.media_type,
                    artifact.size_bytes,
                    to_json(metadata),
                    artifact.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def list_artifacts(
        self,
        session_id: str | None = None,
        tool_call_id: str | None = None,
        limit: int | None = 100,
    ) -> list[ArtifactRecord]:
        query = "select * from artifacts"
        filters = []
        args: list[Any] = []
        if session_id is not None:
            filters.append("session_id = ?")
            args.append(session_id)
        if tool_call_id is not None:
            filters.append("tool_call_id = ?")
            args.append(tool_call_id)
        if filters:
            query += " where " + " and ".join(filters)
        query += " order by created_at desc"
        if limit is not None:
            query += " limit ?"
            args.append(limit)
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(query, tuple(args))
        return [
            ArtifactRecord(
                id=row["id"],
                session_id=row["session_id"],
                tool_call_id=row["tool_call_id"],
                path=row["path"],
                media_type=row["media_type"],
                size_bytes=row["size_bytes"],
                metadata=from_json(row["metadata"], {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def save_memory(self, memory: MemoryRecord) -> None:
        memory_secrets = sensitive_values(memory.metadata)
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                insert into memories (
                  id, namespace, text, embedding, embedding_provider, embedding_model,
                  embedding_dimensions, metadata, scope,
                  importance, confidence, source_session_id, source_turn_id, updated_at,
                  last_used_at, expires_at, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.namespace,
                    redact(memory.text, memory_secrets),
                    to_json(memory.embedding),
                    memory.embedding_provider,
                    memory.embedding_model,
                    memory.embedding_dimensions,
                    to_json(redact(memory.metadata, memory_secrets)),
                    memory.scope.value,
                    1.0,
                    memory.confidence,
                    memory.source_session_id,
                    memory.source_turn_id,
                    memory.updated_at.isoformat(),
                    memory.last_used_at.isoformat() if memory.last_used_at else None,
                    memory.expires_at.isoformat() if memory.expires_at else None,
                    memory.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def list_memories(
        self,
        namespace: str,
        *,
        scopes: list[MemoryScope] | None = None,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        query = "select * from memories where namespace = ?"
        args: list[Any] = [namespace]
        if scopes:
            query += f" and scope in ({', '.join('?' for _ in scopes)})"
            args.extend(scope.value for scope in scopes)
        if not include_expired:
            query += " and (expires_at is null or expires_at > ?)"
            args.append(utc_now().isoformat())
        query += " order by created_at desc"
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(query, tuple(args))
        return [
            MemoryRecord(
                id=row["id"],
                namespace=row["namespace"],
                text=row["text"],
                embedding=from_json(row["embedding"], []),
                embedding_provider=row["embedding_provider"],
                embedding_model=row["embedding_model"],
                embedding_dimensions=row["embedding_dimensions"],
                metadata=from_json(row["metadata"], {}),
                scope=row["scope"],
                confidence=row["confidence"],
                source_session_id=row["source_session_id"],
                source_turn_id=row["source_turn_id"],
                updated_at=row["updated_at"],
                last_used_at=row["last_used_at"],
                expires_at=row["expires_at"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def mark_memories_used(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        placeholders = ", ".join("?" for _ in memory_ids)
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                f"update memories set last_used_at = ? where id in ({placeholders})",
                (utc_now().isoformat(), *memory_ids),
            )
            await db.commit()

def _tool_call_row(row: aiosqlite.Row) -> ToolCallRecord:
    return ToolCallRecord(
        id=row["id"],
        session_id=row["session_id"],
        tool_name=row["tool_name"],
        status=row["status"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        input=from_json(row["input"], {}),
        output=from_json(row["output"], None),
        artifacts=from_json(row["artifacts"], []),
        error=row["error"],
    )
