# Execution

Execution has two independent paths: model providers generate text responses,
and tool executors run explicit tool calls after the gateway enforces policy.

## Model Calls

Built-in providers:

- `OpenAIResponsesProvider`: OpenAI Responses API.
- `CodexCliProvider`: local Codex CLI.

The default agent loop expects provider responses to contain one of these JSON
shapes:

```json
{"final": "answer"}
```

```json
{"tool_calls": [{"name": "tool.name", "arguments": {"key": "value"}}]}
```

`CodexCliProvider` passes only an allowlisted process environment to
`codex exec`: path, home, temporary-directory, locale, and Codex configuration
variables needed for local CLI authentication. Harness secrets and unrelated
ambient credentials are not inherited by the subprocess.

## Tool Gateway

`ToolExecutionGateway` is the standard path for tool calls. It:

- looks up the tool definition;
- validates input against JSON Schema;
- checks required capabilities;
- preflights executor-specific secret eligibility;
- resolves declared secrets only after policy checks pass;
- dispatches to the configured executor;
- validates output schemas when present;
- redacts arguments, outputs, errors, metadata, and artifacts;
- persists the call and emits observability events.

Tool calls emitted in one model response run concurrently.

## Execution Modes

### In Process

In-process execution is for trusted async Python callables. Sync callables are
rejected because Python threads cannot be force-killed safely. Use subprocess or
container mode for blocking work or process-level cancellation.

### Subprocess

Subprocess execution is for trusted local commands. The protocol is JSON on
stdin and JSON or plain text on stdout. Subprocesses do not claim strong
sandboxing; use Docker for untrusted arbitrary code execution.

### Docker Container

Container execution is for tools that need an isolated runtime, especially tools
that execute arbitrary code. A tool references a named container schema, while
the schema defines the execution environment.

Schema responsibilities:

- image and working directory;
- network access;
- mount points;
- read-only root filesystem posture;
- Linux capabilities and privilege posture;
- memory, CPU, PID, and timeout limits;
- whether secrets are allowed;
- whether multiple tools share one session container.

Restrictive defaults include no network, dropped Linux capabilities,
`no-new-privileges`, resource limits, read-only root, temporary `/tmp`, and
validated mounts. Mounts are read-only by default; set
`mount_read_only=false` only when a tool needs write access to the mounted host
directory.

## Container Lifecycle

Every `ToolCall` must include a session id. Container ownership is scoped to the
session and schema:

- By default, a non-secret containerized tool gets a persistent container for
  `(session_id, container_schema, tool_name)`.
- If the schema sets `share_across_tools=true`, tools with the same session and
  schema share one container.
- When the session ends, cleanup is scheduled after
  `HARNESS_CONTAINER_CLEANUP_DELAY_MINUTES`, defaulting to 5 minutes.
- On timeout, cleanup waits for other active Docker calls in the same session to
  finish or time out, then removes that session's containers.

The agent loop owns the normal lifecycle: it enters the session, runs the loop,
schedules session cleanup, and closes runtime-owned executors when appropriate.
Applications that bypass `AgentLoop` and call `ToolExecutionGateway` directly
must make the same lifecycle explicit: call `gateway.end_session(session_id, ...)`
after the last tool call for that session, and call `gateway.aclose()` when the
gateway is no longer needed.

## Session Leases

The agent loop acquires a storage-backed lease before model or tool execution.
A second runtime trying to enter the same session is rejected before work starts.
Active leases are refreshed by heartbeat and can be recovered after their TTL if
a runtime exits without releasing the lease.

```bash
export HARNESS_SESSION_LEASE_TTL_SECONDS=300
export HARNESS_SESSION_LEASE_HEARTBEAT_SECONDS=100
```

The lease model applies to all sessions, not only sessions that use Docker
tools.

## Secrets In Tools

Tool definitions declare the secret names they need. The gateway resolves those
secrets after capability and executor preflight checks pass.

Container tools receive declared secrets in the JSON stdin payload for the
specific invocation. Secrets are not set as persistent container environment
variables.

Secret-enabled container schemas must explicitly set `allow_secrets=true`.
Credentialed calls use one-shot containers by default. A schema can set
`persistent_secrets=true` when iterative state is required; persistent
secret-enabled schemas must keep writable paths on tmpfs.

## Context Compaction

Before a model call, the agent loop can compact older transcript turns when the
active context approaches its budget. The built-in rolling compactor preserves
leading system messages, the current user request, and recent turns, then uses
the configured `ModelProvider` to create a bounded checkpoint summary.

Checkpoint summaries are persisted as committed turns. Later runs replay the
latest valid checkpoint plus raw turns after the checkpoint, so context rolls
forward without replaying already-summarized history.
