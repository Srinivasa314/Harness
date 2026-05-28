# Harness

Harness is a Python framework for building observable, tool-using agents with
real execution boundaries. It gives applications a small agent loop, pluggable
model and embedding providers, capability-gated tools, durable memory, storage,
sandboxed execution, and a local observability dashboard.

The project is aimed at agents that need to run real tools safely enough to be
useful: code review agents, repository assistants, operational copilots, and
other workflows where model calls, tool calls, credentials, memory, and runtime
state need to be inspectable instead of hidden inside a script.

## What It Provides

- **Agent runtime:** provider-neutral JSON protocol for final answers and
  concurrent tool calls.
- **Model providers:** built-in OpenAI Responses and Codex CLI providers, plus a
  class-based `ModelProvider` interface for custom providers and tests.
- **Tool execution layer:** capability checks, tool schemas, secret resolution,
  output validation, and persistence-safe redaction.
- **Execution backends:** in-process, subprocess, and Docker executors, including
  named container schemas and session-scoped containers.
- **Storage:** SQLite for local filesystem-backed runs and Postgres for shared
  or service-style deployments.
- **Memory:** scoped session, agent, and global memories backed by MiniLM,
  OpenAI embeddings, or deterministic test embeddings.
- **Context compaction:** rolling transcript summarization through the active
  model provider when context approaches configured limits.
- **Session ownership:** storage-backed leases prevent two runtimes from
  executing the same session at the same time.
- **Observability:** structured events, model turns, tool traces, artifacts,
  session export, and a NiceGUI dashboard.
- **Testing support:** deterministic local tests plus opt-in provider,
  container, and Postgres e2e suites.

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

It is intentionally generic: the agent clones the PR, explores the repository
with shell commands inside the configured container, reads directly addressed
review instructions from PR comments, stores durable preferences through the
framework memory tool, and posts its review through a tool call.

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
