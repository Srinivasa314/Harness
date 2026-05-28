from __future__ import annotations

from datetime import timedelta

import anyio
import asyncpg

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


class PostgresStorage(StorageBackend):
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = anyio.Lock()

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(self.dsn)
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def migrate(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "select pg_advisory_xact_lock(hashtext('agentic-harness:migrations'))"
                )
                await conn.execute(
                    """
                create table if not exists schema_migrations (
                  version text primary key,
                  applied_at timestamptz not null default now()
                )
                """
                )
                rows = await conn.fetch("select version from schema_migrations")
                applied = {row["version"] for row in rows}
                if SCHEMA_VERSION not in applied:
                    await conn.execute(
                        """
                create table if not exists sessions (
                  id text primary key,
                  created_at timestamptz not null,
                  metadata jsonb not null
                );
                create table if not exists session_leases (
                  session_id text primary key,
                  owner_id text not null,
                  acquired_at timestamptz not null,
                  foreign key(session_id) references sessions(id)
                );
                create table if not exists events (
                  id text primary key,
                  session_id text,
                  event_type text not null,
                  ts timestamptz not null,
                  payload jsonb not null,
                  foreign key(session_id) references sessions(id)
                );
                create index if not exists idx_events_session_ts on events(session_id, ts);
                create table if not exists turns (
                  turn_sequence bigserial primary key,
                  id text not null unique,
                  session_id text not null,
                  role text not null,
                  content text not null,
                  metadata jsonb not null,
                  created_at timestamptz not null,
                  foreign key(session_id) references sessions(id)
                );
                create index if not exists idx_turns_session_created
                  on turns(session_id, turn_sequence);
                create table if not exists tool_calls (
                  id text primary key,
                  session_id text not null,
                  tool_name text not null,
                  status text not null,
                  started_at timestamptz not null,
                  ended_at timestamptz not null,
                  input jsonb not null,
                  output jsonb,
                  artifacts jsonb not null default '[]'::jsonb,
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
                  metadata jsonb not null,
                  created_at timestamptz not null,
                  foreign key(session_id) references sessions(id)
                );
                create index if not exists idx_artifacts_session on artifacts(session_id);
                create index if not exists idx_artifacts_tool_call on artifacts(tool_call_id);
                create table if not exists memories (
                  id text primary key,
                  namespace text not null,
                  text text not null,
                  embedding jsonb not null,
                  embedding_provider text,
                  embedding_model text,
                  embedding_dimensions integer,
                  metadata jsonb not null,
                  scope text not null,
                  confidence double precision not null,
                  source_session_id text,
                  source_turn_id text,
                  updated_at timestamptz not null,
                  last_used_at timestamptz,
                  expires_at timestamptz,
                  created_at timestamptz not null
                );
                create index if not exists idx_memories_namespace on memories(namespace);
                create index if not exists idx_memories_namespace_scope
                  on memories(namespace, scope);
                """
                    )
                    await conn.execute(
                        """
                        insert into schema_migrations (version)
                        values ($1)
                        on conflict do nothing
                        """,
                        SCHEMA_VERSION,
                    )
                await conn.execute(
                    """
                    create table if not exists session_leases (
                      session_id text primary key,
                      owner_id text not null,
                      acquired_at timestamptz not null,
                      foreign key(session_id) references sessions(id)
                    )
                    """
                )

    async def create_session(self, session: Session) -> None:
        metadata = redact_with_detected_secrets(session.metadata)
        pool = await self._get_pool()
        await pool.execute(
            "insert into sessions (id, created_at, metadata) values ($1, $2, $3::jsonb)",
            session.id,
            session.created_at,
            to_json(metadata),
        )

    async def _ensure_session(self, session_id: str | None) -> None:
        if session_id is None:
            return
        pool = await self._get_pool()
        session = Session(id=session_id, metadata={"created_by": "implicit"})
        await pool.execute(
            """
            insert into sessions (id, created_at, metadata)
            values ($1, $2, $3::jsonb)
            on conflict do nothing
            """,
            session.id,
            session.created_at,
            to_json(redact(session.metadata)),
        )

    async def get_session(self, session_id: str) -> Session | None:
        pool = await self._get_pool()
        row = await pool.fetchrow("select * from sessions where id = $1", session_id)
        if row is None:
            return None
        return Session(
            id=row["id"],
            created_at=row["created_at"],
            metadata=from_json(row["metadata"], {}),
        )

    async def try_acquire_session_lease(
        self,
        session_id: str,
        owner_id: str,
        *,
        ttl_seconds: float,
    ) -> bool:
        await self._ensure_session(session_id)
        pool = await self._get_pool()
        acquired_at = utc_now()
        row = await pool.fetchrow(
            """
            insert into session_leases (session_id, owner_id, acquired_at)
            values ($1, $2, $3)
            on conflict(session_id) do update set
              owner_id = excluded.owner_id,
              acquired_at = excluded.acquired_at
            where session_leases.acquired_at <= $4
            returning session_id
            """,
            session_id,
            owner_id,
            acquired_at,
            acquired_at - timedelta(seconds=ttl_seconds),
        )
        return row is not None

    async def refresh_session_lease(
        self,
        session_id: str,
        owner_id: str,
        *,
        ttl_seconds: float,
    ) -> bool:
        pool = await self._get_pool()
        refreshed_at = utc_now()
        result = await pool.execute(
            """
            update session_leases
            set acquired_at = $3
            where session_id = $1 and owner_id = $2 and acquired_at > $4
            """,
            session_id,
            owner_id,
            refreshed_at,
            refreshed_at - timedelta(seconds=ttl_seconds),
        )
        return result.endswith(" 1")

    async def release_session_lease(self, session_id: str, owner_id: str) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "delete from session_leases where session_id = $1 and owner_id = $2",
            session_id,
            owner_id,
        )

    async def save_turn(self, turn: TurnRecord) -> None:
        turn_secrets = sensitive_values(turn.metadata)
        await self._ensure_session(turn.session_id)
        pool = await self._get_pool()
        await pool.execute(
            """
            insert into turns (id, session_id, role, content, metadata, created_at)
            values ($1, $2, $3, $4, $5::jsonb, $6)
            """,
            turn.id,
            turn.session_id,
            turn.role,
            redact(turn.content, turn_secrets),
            to_json(redact(turn.metadata, turn_secrets)),
            turn.created_at,
        )

    async def list_turns(self, session_id: str, limit: int | None = 100) -> list[TurnRecord]:
        pool = await self._get_pool()
        if limit is None:
            rows = await pool.fetch(
                """
                select * from turns
                where session_id = $1
                order by turn_sequence asc
                """,
                session_id,
            )
        else:
            rows = await pool.fetch(
                """
                select * from (
                  select * from turns
                  where session_id = $1
                  order by turn_sequence desc
                  limit $2
                ) recent_turns
                order by turn_sequence asc
                """,
                session_id,
                limit,
            )
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
        await self._ensure_session(event.session_id)
        pool = await self._get_pool()
        await pool.execute(
            """
            insert into events (id, session_id, event_type, ts, payload)
            values ($1, $2, $3, $4, $5::jsonb)
            """,
            event.id,
            event.session_id,
            event.event_type,
            event.ts,
            to_json(redact_with_detected_secrets(event.payload)),
        )

    async def list_events(
        self, session_id: str | None = None, limit: int | None = 100
    ) -> list[Event]:
        pool = await self._get_pool()
        if session_id is None and limit is None:
            rows = await pool.fetch("select * from events order by ts desc")
        elif session_id is None:
            rows = await pool.fetch("select * from events order by ts desc limit $1", limit)
        elif limit is None:
            rows = await pool.fetch(
                "select * from events where session_id = $1 order by ts desc",
                session_id,
            )
        else:
            rows = await pool.fetch(
                "select * from events where session_id = $1 order by ts desc limit $2",
                session_id,
                limit,
            )
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
        await self._ensure_session(call.session_id)
        pool = await self._get_pool()
        redaction_secrets = {
            **sensitive_values(call.arguments),
            **sensitive_values(result.model_dump(mode="json")),
        }
        await pool.execute(
            """
            insert into tool_calls
            (
              id, session_id, tool_name, status, started_at, ended_at,
              input, output, artifacts, error
            )
            values ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::jsonb, $10)
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
            call.call_id,
            call.session_id,
            call.name,
            result.status,
            result.started_at,
            result.ended_at,
            to_json(redact(call.arguments, redaction_secrets)),
            to_json(redact(result.output, redaction_secrets)),
            to_json(redact(result.artifacts, redaction_secrets)),
            redact(result.error, redaction_secrets),
        )

    async def list_tool_calls(
        self, session_id: str | None = None, limit: int | None = 100
    ) -> list[ToolCallRecord]:
        pool = await self._get_pool()
        if session_id is None and limit is None:
            rows = await pool.fetch("select * from tool_calls order by started_at desc")
        elif session_id is None:
            rows = await pool.fetch(
                "select * from tool_calls order by started_at desc limit $1",
                limit,
            )
        elif limit is None:
            rows = await pool.fetch(
                "select * from tool_calls where session_id = $1 order by started_at desc",
                session_id,
            )
        else:
            rows = await pool.fetch(
                "select * from tool_calls where session_id = $1 order by started_at desc limit $2",
                session_id,
                limit,
            )
        return [_tool_call_row(row) for row in rows]

    async def save_artifact(self, artifact: ArtifactRecord) -> None:
        metadata = redact_with_detected_secrets(artifact.metadata)
        await self._ensure_session(artifact.session_id)
        pool = await self._get_pool()
        await pool.execute(
            """
            insert into artifacts
            (id, session_id, tool_call_id, path, media_type, size_bytes, metadata, created_at)
            values ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
            """,
            artifact.id,
            artifact.session_id,
            artifact.tool_call_id,
            artifact.path,
            artifact.media_type,
            artifact.size_bytes,
            to_json(metadata),
            artifact.created_at,
        )

    async def list_artifacts(
        self,
        session_id: str | None = None,
        tool_call_id: str | None = None,
        limit: int | None = 100,
    ) -> list[ArtifactRecord]:
        pool = await self._get_pool()
        limit_clause = "" if limit is None else " limit $3"
        if session_id is not None and tool_call_id is not None:
            rows = await pool.fetch(
                f"""
                select * from artifacts
                where session_id = $1 and tool_call_id = $2
                order by created_at desc{limit_clause}
                """,
                session_id,
                tool_call_id,
                *(() if limit is None else (limit,)),
            )
        elif session_id is not None:
            query = "select * from artifacts where session_id = $1 order by created_at desc"
            if limit is None:
                rows = await pool.fetch(query, session_id)
            else:
                rows = await pool.fetch(f"{query} limit $2", session_id, limit)
        elif tool_call_id is not None:
            query = "select * from artifacts where tool_call_id = $1 order by created_at desc"
            if limit is None:
                rows = await pool.fetch(query, tool_call_id)
            else:
                rows = await pool.fetch(f"{query} limit $2", tool_call_id, limit)
        else:
            query = "select * from artifacts order by created_at desc"
            if limit is None:
                rows = await pool.fetch(query)
            else:
                rows = await pool.fetch(f"{query} limit $1", limit)
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
        pool = await self._get_pool()
        await pool.execute(
            """
            insert into memories (
              id, namespace, text, embedding, embedding_provider, embedding_model,
              embedding_dimensions, metadata, scope,
              confidence, source_session_id, source_turn_id, updated_at, last_used_at,
              expires_at, created_at
            )
            values (
              $1, $2, $3, $4::jsonb, $5, $6, $7, $8::jsonb, $9, $10, $11,
              $12, $13, $14, $15, $16
            )
            """,
            memory.id,
            memory.namespace,
            redact(memory.text, memory_secrets),
            to_json(memory.embedding),
            memory.embedding_provider,
            memory.embedding_model,
            memory.embedding_dimensions,
            to_json(redact(memory.metadata, memory_secrets)),
            memory.scope.value,
            memory.confidence,
            memory.source_session_id,
            memory.source_turn_id,
            memory.updated_at,
            memory.last_used_at,
            memory.expires_at,
            memory.created_at,
        )

    async def list_memories(
        self,
        namespace: str,
        *,
        scopes: list[MemoryScope] | None = None,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        pool = await self._get_pool()
        filters = ["namespace = $1"]
        args: list[object] = [namespace]
        if scopes:
            args.append([scope.value for scope in scopes])
            filters.append(f"scope = any(${len(args)}::text[])")
        if not include_expired:
            args.append(utc_now())
            filters.append(f"(expires_at is null or expires_at > ${len(args)})")
        rows = await pool.fetch(
            f"select * from memories where {' and '.join(filters)} order by created_at desc",
            *args,
        )
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
        pool = await self._get_pool()
        await pool.execute(
            "update memories set last_used_at = now() where id = any($1::text[])",
            memory_ids,
        )

def _tool_call_row(row: asyncpg.Record) -> ToolCallRecord:
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
