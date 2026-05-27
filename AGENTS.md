# Agent Development Guide

This file is the operating guide for humans and coding agents working in this
repository. Keep user-facing product docs in `README.md` and feature-specific
details in `docs/`; use this file for development setup, validation, and review
workflow.

## Workspace Rules

- Work inside the repository root only.
- Do not touch files outside the project directory unless the user explicitly
  approves it.
- Keep `.env`, `.env.*`, `.venv`, caches, local data, and generated validation
  outputs out of git.
- Use `uv` for dependency management and command execution.
- Use `apply_patch` for manual file edits.
- Before finalizing, stage intentional project changes and verify the staged
  tree.

## Setup

```bash
uv sync --extra dev --extra embeddings
```

Local settings may live in `.env.development`; it is loaded by the app and
ignored by git.

Common local settings:

```bash
export HARNESS_STORAGE_BACKEND=sqlite
export HARNESS_SQLITE_PATH=data/harness.sqlite3
export HARNESS_DOCKER_BIN=docker
```

If Docker is installed in a non-standard location or uses a named context, set
`HARNESS_DOCKER_BIN` and Docker's own environment variables for that machine.

## Required Local Gate

Run these before claiming the project is ready:

```bash
uv run ruff check
uv run ty check
uv run pytest -m "not e2e and not container and not postgres and not provider"
```

The deterministic suite should run without credentials or external model calls.

## Optional E2E Gates

Run these when the matching dependency is available:

```bash
uv run pytest -m container
uv run pytest -m postgres
uv run pytest -m provider
```

Container tests:

```bash
uv run pytest -m container
```

Disposable Postgres through Docker:

```bash
docker run --rm -d \
  --name harness-postgres-e2e \
  -e POSTGRES_PASSWORD=harness \
  -e POSTGRES_USER=harness \
  -e POSTGRES_DB=harness \
  -p 55432:5432 postgres:16-alpine

HARNESS_POSTGRES_DSN='postgresql://harness:harness@127.0.0.1:55432/harness' \
  uv run pytest -m postgres

docker stop harness-postgres-e2e
```

Provider e2e:

```bash
HARNESS_RUN_OPENAI_E2E=1 HARNESS_OPENAI_API_KEY='...' uv run pytest -m provider \
  tests/test_e2e_providers.py::test_openai_provider_e2e

HARNESS_RUN_OPENAI_E2E=1 HARNESS_OPENAI_API_KEY='...' uv run pytest -m provider \
  tests/test_e2e_providers.py::test_openai_embedding_provider_e2e

HARNESS_RUN_CODEX_E2E=1 HARNESS_MODEL_PROVIDER=codex \
  uv run pytest -m provider tests/test_e2e_providers.py::test_codex_cli_provider_e2e

uv sync --extra dev --extra embeddings
HARNESS_RUN_MINILM_E2E=1 HF_HOME=data/hf-cache SENTENCE_TRANSFORMERS_HOME=data/sentence-transformers \
  uv run pytest -m e2e tests/test_e2e_providers.py::test_minilm_embedding_provider_e2e
```

The Codex test requires a locally authenticated Codex CLI and uses the CLI's
default model, which keeps it compatible with ChatGPT subscription accounts. It
is not exposed as a GitHub Actions workflow input. The MiniLM test may download
model files; keep both opt-in.

## Staged Tree Checks

Before final handoff, verify the staged content, not only the working tree:

```bash
git diff --cached --check

tmpdir=".tmp-staged-check"
rm -rf "$tmpdir"
mkdir "$tmpdir"
git checkout-index -a --prefix="$tmpdir/"
cd "$tmpdir"
uv run python - <<'PY'
from harness.execution import DockerContainerExecutor, InProcessExecutor, SubprocessExecutor
from harness.models import CodexCliProvider, ModelProvider
from harness.memory import EmbeddingProvider, MemoryStore
print(
    DockerContainerExecutor.__name__,
    InProcessExecutor.__name__,
    SubprocessExecutor.__name__,
    ModelProvider.__name__,
    CodexCliProvider.__name__,
    EmbeddingProvider.__name__,
    MemoryStore.__name__,
)
PY
cd -
rm -rf "$tmpdir"
```

## Three-Agent Verification Pass

Use a final three-agent review when a change is broad, security-sensitive, or
intended as a release-ready handoff. Use fresh context and a strong current
coding/reasoning model. If the user specifies a model or reasoning level, use
that. Run the pass with three independent subagents. The three passes should be
read-only and independent:

1. Correctness/security: review the full project for correctness, security,
   data integrity, unsafe defaults, broken runtime paths, and regressions. Do
   not limit the review to a fixed feature checklist; the project surface may
   evolve over time.
2. Architecture/code quality: review pluggability, lifecycle/resource handling,
   API consistency, configuration ergonomics, module boundaries, docs alignment,
   and maintainability.
3. Test/release readiness: review deterministic tests, CI markers, e2e strategy,
   staged-tree completeness, docs/examples, release smoke, and supported OS
   assumptions.

The pass is only clean when all three reviewers report no high or medium
findings. Low-priority findings can remain if they are documented and do not
contradict the current release scope.

## Reporting Results

Final handoff should include:

- Commands run and pass/fail counts.
- Any skipped tests and what is needed to run them.
- Remaining low-priority findings.
- Current git staging state.
