# Harness

Harness is a Python runtime for agent sessions. It gives the agent
loop explicit boundaries for model calls, tool execution, credentials, storage,
memory, sandboxing, and observability.

## What It Provides

- Provider-neutral JSON protocol for agent final answers and tool calls.
- Class-based provider interfaces for models, embeddings, storage,
  credentials, tools, and executors.
- OpenAI Responses and Codex CLI model providers.
- SQLite and Postgres storage for sessions, turns, events, tool calls,
  artifacts, leases, and memories.
- Capability-gated tool execution with secret resolution and redaction before
  persistence.
- In-process, subprocess, and Docker tool execution, including named container
  schemas and session-scoped containers.
- Concurrent execution for tool calls emitted in the same model response.
- Durable scoped memory with MiniLM, OpenAI, or deterministic test embeddings.
- Context-window compaction through a pluggable compactor backed by the active
  model provider.
- Storage-backed session leases that prevent concurrent runtimes from owning
  the same session.
- Structured observability events, tool traces, artifact records, session
  exports, and a local dashboard.

## Quick Start

Install the project with development tools and local embedding support:

```bash
uv sync --extra dev --extra embeddings
```

Pick a model provider. OpenAI uses the Responses API:

```bash
export HARNESS_MODEL_PROVIDER=openai
export HARNESS_OPENAI_API_KEY='...'
```

Or use an already-authenticated Codex CLI:

```bash
export HARNESS_MODEL_PROVIDER=codex
```

Use SQLite locally and create the schema:

```bash
export HARNESS_STORAGE_BACKEND=sqlite
export HARNESS_SQLITE_PATH=data/harness.sqlite3
uv run harness migrate
```

Run a minimal agent session:

```bash
uv run harness run-agent "Return a short status message."
```

Run with a capability-gated example tool:

```bash
uv run harness run-agent \
  --tools examples/tools.json \
  --capability text:uppercase \
  "Use the text.uppercase tool to uppercase hello, then return the tool result."
```

Inspect or export stored sessions:

```bash
uv run harness dashboard
uv run harness export-session SESSION_ID --output session.json
```

## Examples

- [GitHub PR review agent](examples/github_pr_review_agent/README.md): runs a
  real PR review using an LLM, embeddings-backed memory, a credentialized
  GitHub tool, Docker sandbox execution, context compaction, and observability.
  It intentionally uses a focused subset of the framework rather than every
  available feature.

## Documentation

- [Architecture](docs/architecture.md): runtime boundaries and package map.
- [Configuration](docs/configuration.md): environment settings and code-level
  pluggability.
- [Execution](docs/execution.md): model calls, tool calls, sandboxing, secrets,
  and session lifecycle.
- [Memory](docs/memory.md): scopes, retrieval, auto-capture, and compaction.
- [Observability](docs/observability.md): persisted records, events, dashboard,
  and exports.
- [Testing](docs/testing.md): deterministic and optional e2e validation.
- [Development workflow](AGENTS.md): repository setup, quality gates, and review
  workflow.
