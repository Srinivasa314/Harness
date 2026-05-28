# Harness

Harness is a Python framework for building observable, tool-using agents with
real execution boundaries. It gives applications a small agent loop, pluggable
model and embedding providers, isolated tool execution, scoped memory, durable
storage, and a local observability dashboard.

## What It Provides

- **Agent runtime and providers:** provider-neutral JSON for responses and
  concurrent tool calls, built-in OpenAI Responses and Codex CLI inference, and
  class-based provider interfaces for application-specific implementations.
- **Secure tool execution:** capability checks, input and output schemas,
  isolated secret resolution, persistence-safe redaction, in-process,
  subprocess, and Docker execution, plus named container schemas for filesystem,
  network, resource, privilege, and lifecycle controls.
- **State and memory:** SQLite or Postgres persistence for sessions, turns,
  events, calls, artifacts, leases, and memories; scoped retrieval memory backed
  by MiniLM, OpenAI embeddings, or deterministic test embeddings.
- **Observability:** structured runtime events, tool traces, redacted exports
  and a local NiceGUI dashboard.

## Runtime Shape

```text
caller
  -> AgentSessionManager
  -> AgentLoop
  -> MemoryManager
  -> ModelProvider
  -> ToolExecutionGateway
  -> ToolExecutor
  -> StorageBackend and EventSink
  -> final response
```

Each boundary is replaceable in code. Registries are available when an
application wants settings-selected implementations, but providers can also be
injected directly.

## Security Model

Harness treats model output as a request for work, not as authority to perform
that work. Every tool call goes through `ToolExecutionGateway`, which validates
arguments, checks required capabilities, resolves declared secrets only after
policy passes, dispatches to the selected executor, validates declared outputs,
and redacts arguments, outputs, errors, metadata, artifacts, memory text, and
events before durable persistence.

Credential isolation is explicit. Tools declare the secret names they need, and
the default resolver reads only `HARNESS_SECRET_...` variables for those names.
The Codex CLI provider starts `codex exec` with an allowlisted environment so
Harness secrets and unrelated ambient credentials are not inherited by the
model subprocess. Docker tools receive declared secrets only in the per-call
JSON payload, not as long-lived container environment variables.

Docker sandboxing is controlled by container schemas rather than by individual
tool prompts. Schemas define the image, working directory, mounts, network
posture, Linux capabilities, privilege posture, resource limits, timeout, secret
eligibility, and whether tools share a persistent session container.

## Quick Start

Install development dependencies and local embedding support:

```bash
uv sync --extra dev --extra embeddings
```

Create local SQLite storage:

```bash
export HARNESS_STORAGE_BACKEND=sqlite
export HARNESS_SQLITE_PATH=data/harness.sqlite3
uv run harness migrate
```

Choose an inference provider. OpenAI uses the Responses API:

```bash
export HARNESS_MODEL_PROVIDER=openai
export HARNESS_OPENAI_API_KEY='...'
```

Or use a locally authenticated Codex CLI:

```bash
export HARNESS_MODEL_PROVIDER=codex
```

Run a minimal session:

```bash
uv run harness run-agent "Return a short status message."
```

Run with a capability-gated tool:

```bash
uv run harness run-agent \
  --tools examples/tools.json \
  --capability text:uppercase \
  "Use the text.uppercase tool to uppercase hello, then return the result."
```

Open the dashboard or export a session:

```bash
uv run harness dashboard
uv run harness export-session SESSION_ID --output session.json
```

For the full validation workflow, see [Testing](docs/testing.md) and
[AGENTS.md](AGENTS.md).

## Example Agent

The [GitHub PR review agent](examples/github_pr_review_agent/README.md) is the
main end-to-end example. It reviews a real pull request using:

- a GitHub App credential flow;
- real LLM inference through OpenAI or Codex CLI;
- embeddings-backed memory for review preferences;
- Docker-backed repository exploration;
- a credentialized GitHub comment tool;
- context compaction and dashboard observability.

It is intentionally generic: the agent copies git-tracked files from a local PR
checkout into a temporary review workspace, explores that workspace with shell
commands inside the configured container, reads directly addressed review
instructions from PR comments, stores durable preferences through the framework
memory tool, and posts its review through a tool call.

## Configuration

Harness is configured with `HARNESS_` environment variables and can also be
composed directly in Python. Common extension points include:

- `ModelProvider` for inference;
- `EmbeddingProvider`, `MemoryStore`, and `MemoryManager` for retrieval memory;
- `StorageBackend` for persistence;
- `SecretResolver` for credentials;
- `ToolExecutor` for execution backends;
- `ToolRegistry` for application tools.

See [Configuration](docs/configuration.md) for settings and injection examples.

## Storage, Memory, And Observability

Storage backends persist the operational record: sessions, turns, events, tool
calls, artifacts, leases, and memory records. SQLite is the local default, while
Postgres is available for shared deployments or service-style operation.

Memory is retrieval state, separate from transcript compaction. Harness supports
`session`, `agent`, and `global` scopes, stores embedding metadata with each
memory, and retrieves only memories compatible with the active embedding
configuration. Agents can write memories through the generic `memory.store` tool
when granted `memory:write`, while applications can also write through
`MemoryManager` directly.

The dashboard is a local read-only view over the stored runtime record. It shows
session summaries, aggregate counts, turns, events, tool calls, artifacts, and
tool status filters, and uses the same redacted records available through
session export.

## Documentation

- [Architecture](docs/architecture.md): runtime boundaries and package map.
- [Execution](docs/execution.md): model calls, tool calls, sandboxing, secrets,
  and session lifecycle.
- [Memory](docs/memory.md): scopes, retrieval, memory capture, and compaction.
- [Observability](docs/observability.md): persisted records, events, dashboard,
  and exports.
- [Testing](docs/testing.md): deterministic and optional e2e validation.
- [Development workflow](AGENTS.md): repository setup, local gates, and
  three-agent review workflow.
