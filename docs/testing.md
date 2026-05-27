# Testing And E2E Validation

Development workflow and final review expectations live in `AGENTS.md`. This
page explains the test slices and what each one needs.

## Supported Targets

Deterministic tests are intended to run on Linux and macOS with Python 3.11 and
3.12. Docker, Postgres, OpenAI, Codex CLI, and MiniLM checks are optional e2e
slices and run only when their external dependency is available. Windows is not
a supported target for this v1.

## Deterministic Gate

The local deterministic gate requires no credentials or external model calls:

```bash
uv run ruff check
uv run ty check
uv run pytest -m "not e2e and not container and not postgres and not provider"
```

It covers unit behavior, SQLite storage contracts, redaction boundaries, runtime
composition, CLI behavior, memory behavior, dashboard view models, and mocked
provider/container paths.

## Pytest Markers

- `e2e`: external services, providers, downloads, or runtimes.
- `container`: Docker-compatible runtime required.
- `postgres`: Postgres service required.
- `provider`: real model provider required.

Tests skip when required environment variables, credentials, CLIs, or services
are unavailable.

## Container Tests

```bash
uv run pytest -m container
```

Requires a Docker-compatible CLI and daemon.

## Postgres Tests

```bash
export HARNESS_POSTGRES_DSN='postgresql://user:password@localhost:5432/harness_test'
uv run pytest -m postgres
```

Use a disposable database. The storage contract covers sessions, turns, events,
tool calls, artifacts, and memories.

## Provider Tests

OpenAI:

```bash
export HARNESS_RUN_OPENAI_E2E=1
export HARNESS_OPENAI_API_KEY='...'
export HARNESS_OPENAI_MODEL=gpt-5.2
uv run pytest -m provider tests/test_e2e_providers.py::test_openai_provider_e2e
```

OpenAI embeddings:

```bash
export HARNESS_RUN_OPENAI_E2E=1
export HARNESS_OPENAI_API_KEY='...'
export HARNESS_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
uv run pytest -m provider tests/test_e2e_providers.py::test_openai_embedding_provider_e2e
```

Codex CLI:

```bash
export HARNESS_RUN_CODEX_E2E=1
export HARNESS_MODEL_PROVIDER=codex
uv run pytest -m provider tests/test_e2e_providers.py::test_codex_cli_provider_e2e
```

The Codex test requires a locally installed and authenticated Codex CLI. It uses
the CLI's authenticated default model so it works with ChatGPT subscription
accounts. Set `HARNESS_CODEX_MODEL` only when validating a specific model that
is available to that account. Because CI runners do not have local authenticated
CLI state, this slice is documented as a local e2e command rather than a GitHub
Actions workflow input.

MiniLM embeddings:

```bash
uv sync --extra dev --extra embeddings
export HARNESS_RUN_MINILM_E2E=1
uv run pytest -m e2e tests/test_e2e_providers.py::test_minilm_embedding_provider_e2e
```

The MiniLM test may download the configured sentence-transformers model on first
run.
