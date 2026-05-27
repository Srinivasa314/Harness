# Architecture

Harness is organized around one running agent session. The core runtime stays
small and delegates model providers, storage, credentials, embeddings, and tool
execution through class-based interfaces.

## Runtime Flow

```text
caller
  -> AgentSessionManager
  -> AgentLoop
  -> MemoryManager
  -> ModelProvider
  -> protocol parser
  -> ToolExecutionGateway
  -> ToolExecutor
  -> StorageBackend and EventSink
  -> final response
```

The default model protocol is provider-neutral JSON:

```json
{"final": "answer"}
```

```json
{"tool_calls": [{"name": "tool.name", "arguments": {"key": "value"}}]}
```

Tool calls emitted in the same model response run concurrently.

## Core Boundaries

- `ModelProvider`: inference provider. Built-ins support OpenAI Responses and
  Codex CLI.
- `ToolExecutionGateway`: the mandatory standard path for tool calls.
- `ToolExecutor`: execution backend for in-process, subprocess, or Docker tools.
- `SecretResolver`: credential lookup for declared tool secrets.
- `StorageBackend`: durable records for sessions, turns, events, tool calls,
  artifacts, and memories.
- `EmbeddingProvider`: vectors for memory retrieval.
- `MemoryManager`: scoped retrieval, context formatting, and optional
  auto-capture.
- `ContextCompactor`: transcript summarization when the active context grows
  near its budget.
- `EventSink`: structured observability events created by runtime components
  from the configured storage backend.

Provider, storage, memory, credential, and executor boundaries can be injected
directly. Registries are only needed when an application wants settings-selected
named implementations.

## Package Map

- `harness.agent`: sessions, leases, loop, protocol parsing, and compaction.
- `harness.models`: model provider interfaces and built-in providers.
- `harness.tools`: registry, policy, secrets, gateway, and redaction.
- `harness.execution`: in-process, subprocess, and Docker executors.
- `harness.storage`: SQLite and Postgres persistence.
- `harness.memory`: embedding providers, memory store, and memory manager.
- `harness.observability`: structured event recording.
- `harness.dashboard`: local NiceGUI dashboard.
- `harness.runtime`: default composition for CLI and embedded use.

## Session Ownership

Every agent run executes inside a session. The agent loop acquires a
storage-backed lease before model or tool execution and releases it when the run
exits. This makes the session id the ownership boundary and prevents two
runtimes from executing the same session at the same time. Stale leases can be
recovered after their TTL expires.

## Non-Goals

- Multi-agent orchestration.
- Non-Docker container runtimes.
- Built-in local general-purpose inference.
- Provider-native tool-call adapters.
- Hosted dashboard authentication or multi-tenant operations.
- Built-in evaluation, regression, or alerting products.
