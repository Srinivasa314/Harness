# Observability

Harness records structured runtime state in storage. The same records power
debugging, dashboard views, session exports, and external monitoring systems.

## Persisted Records

`StorageBackend` persists:

- sessions;
- turns;
- events;
- tool calls;
- artifacts;
- memories.

All public storage write paths redact secret-like values before durable
persistence. Redaction applies to direct storage calls and to writes routed
through the agent loop, memory manager, and tool gateway.

## Events

`EventSink` records structured events and redacts payloads before writing. It is
fail-open: an observability write failure should not stop agent execution.

Common event categories:

- agent run and iteration lifecycle;
- model/provider failure;
- tool-call start, finish, denial, and timeout;
- declared secret resolution by name;
- memory retrieval and capture;
- context compaction.

Events are operational breadcrumbs, not the only source of truth. Turns, tool
calls, artifacts, and memories are persisted as first-class records.

## Tool Calls

Tool call records store normalized execution results:

- tool name and execution mode;
- input arguments;
- output;
- status and error;
- artifact paths;
- timestamps.

The gateway redacts inputs and results before persistence, including declared
secrets and secret-like values discovered in payloads.

## Artifacts

`FileArtifactStore` writes bytes under a configured artifact root and persists an
artifact record. Path components are validated to prevent traversal. Duplicate
filenames receive unique suffixes instead of overwriting existing files.

## Dashboard

The NiceGUI dashboard is a local read-only view over persisted sessions, turns,
events, tool calls, and artifacts. Memory records are persisted and available
through storage APIs and session exports, but they do not currently have a
dedicated dashboard tab.

```bash
uv run harness dashboard
```

The dashboard is intended for local development and binds to localhost by
default. It is not a hosted multi-tenant UI.

## Export

For offline inspection:

```bash
uv run harness export-session SESSION_ID --output session.json
```

Exports use the same redacted storage records as the dashboard. External
evaluation, monitoring, or alerting systems can consume these exports or read
from the storage backend directly.
