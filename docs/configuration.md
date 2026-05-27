# Configuration

Harness settings use `HARNESS_` environment variables. The app reads the
process environment plus `.env` and `.env.development`; local `.env*` files stay
out of git.

## Storage

SQLite is the local default:

```bash
export HARNESS_STORAGE_BACKEND=sqlite
export HARNESS_SQLITE_PATH=data/harness.sqlite3
```

Postgres is selected with a DSN:

```bash
export HARNESS_STORAGE_BACKEND=postgres
export HARNESS_POSTGRES_DSN='postgresql://user:password@localhost:5432/harness'
```

`StorageBackend` implementations can also be injected directly, or registered
with `StorageBackendRegistry` for settings-based selection.

## Models

OpenAI Responses:

```bash
export HARNESS_MODEL_PROVIDER=openai
export HARNESS_OPENAI_API_KEY='...'
export HARNESS_OPENAI_MODEL=gpt-5.2
```

Codex CLI:

```bash
export HARNESS_MODEL_PROVIDER=codex
```

By default Harness lets the authenticated Codex CLI choose its configured
model. Set `HARNESS_CODEX_MODEL` only when you know that model is available for
the current Codex account.

Applications can inject any `ModelProvider` directly:

```python
from harness.models import ModelMessage, ModelProvider, ModelResponse
from harness.runtime import build_runtime_async


class StaticModel(ModelProvider):
    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        return ModelResponse(content='{"final": "ok"}')


runtime = await build_runtime_async(model=StaticModel())
```

Use `ModelProviderRegistry` only when a named provider should be selected from
settings.

## Tools And Capabilities

Runtime-created policies grant no capabilities by default. Grant only what a run
needs:

```bash
uv run harness run-agent \
  --tools examples/tools.json \
  --capability text:uppercase \
  "hello"
```

Capabilities can also come from settings:

```bash
export HARNESS_TOOL_CAPABILITIES='["text:uppercase"]'
```

Externally loaded tools must declare at least one required capability. Use an
explicit public capability such as `tool:public` when broad access is intended.

## Credentials

The default resolver reads environment variables with the `HARNESS_SECRET_`
prefix:

```bash
export HARNESS_SECRET_API_TOKEN='...'
```

Secret names are uppercased and non-alphanumeric characters become underscores,
so `api-token` resolves from `HARNESS_SECRET_API_TOKEN`.

Tool definitions declare the secret names they need. Secrets are resolved only
inside the tool execution path and redacted before persistence. Credential
resolution is pluggable:

```python
from harness.runtime import build_runtime
from harness.tools import SecretResolver


class VaultResolver(SecretResolver):
    async def resolve(self, names: list[str]) -> dict[str, str]:
        return {name: await read_secret_from_vault(name) for name in names}


runtime = build_runtime(secret_resolver=VaultResolver())
```

Use `SecretResolverRegistry` only when a named resolver should be selected from
settings.

## Embeddings And Memory

MiniLM is the default embedding provider:

```bash
uv sync --extra embeddings
export HARNESS_EMBEDDING_PROVIDER=minilm
export HARNESS_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

OpenAI embeddings use the OpenAI embeddings API:

```bash
export HARNESS_EMBEDDING_PROVIDER=openai
export HARNESS_OPENAI_API_KEY='...'
export HARNESS_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

For OpenAI `text-embedding-3` models, the returned vector size can be requested
with:

```bash
export HARNESS_OPENAI_EMBEDDING_DIMENSIONS=1536
```

Memories are retrieved only when their stored embedding provider, model, and
dimension match the active embedding configuration. Use a new
`HARNESS_MEMORY_NAMESPACE` when changing embedding configurations and you want
to avoid carrying old memory records.

The hash provider is deterministic and intended for tests:

```bash
export HARNESS_EMBEDDING_PROVIDER=hash
export HARNESS_EMBEDDING_DIMENSIONS=32
```

Memory controls:

```bash
export HARNESS_MEMORY_ENABLED=true
export HARNESS_MEMORY_NAMESPACE=default
export HARNESS_MEMORY_RETRIEVAL_LIMIT=5
export HARNESS_MEMORY_MAX_CONTEXT_CHARS=4000
export HARNESS_MEMORY_AUTO_CAPTURE=false
```

Applications can inject `EmbeddingProvider`, `MemoryStore`, `MemoryPolicy`, or
`MemoryManager` directly. `EmbeddingProviderRegistry` is optional and exists for
settings-selected named embedding providers.

## Context Compaction

Runtime-built agent loops enable rolling context compaction by default:

```bash
export HARNESS_CONTEXT_COMPACTION_ENABLED=true
export HARNESS_CONTEXT_MAX_CHARS=120000
export HARNESS_CONTEXT_COMPACTION_TRIGGER_RATIO=0.8
export HARNESS_CONTEXT_COMPACTION_PRESERVE_RECENT_MESSAGES=8
export HARNESS_CONTEXT_COMPACTION_SUMMARIZER_INPUT_MAX_CHARS=24000
export HARNESS_CONTEXT_COMPACTION_SUMMARY_MAX_CHARS=4000
```

The default compactor uses the configured `ModelProvider`. Applications can pass
`context_compactor=` to `RuntimeContext.agent_loop()` or directly to `AgentLoop`
for a no-op, token-aware, separate-model, or provider-specific implementation.

## Docker

Container tools use the Docker CLI:

```bash
export HARNESS_DOCKER_BIN=docker
export HARNESS_CONTAINER_SCHEMAS_PATH=container-schemas.json
export HARNESS_CONTAINER_CLEANUP_DELAY_MINUTES=5
```

Container schemas are separate from tool definitions. See
[execution.md](execution.md) for schema behavior and secret handling.

## Runtime Composition

Most runtime pieces can be swapped at composition time:

```python
from harness.runtime import build_runtime
from harness.schemas import ExecutionMode

runtime = build_runtime(
    model=my_model_provider,
    model_registry=my_model_registry,
    storage=my_storage_backend,
    secret_resolver=my_secret_resolver,
    embeddings=my_embedding_provider,
    memory=my_memory_manager,
    embedding_registry=my_embedding_registry,
    registry=my_tool_registry,
    executors={
        ExecutionMode.IN_PROCESS: my_in_process_executor,
        ExecutionMode.SUBPROCESS: my_subprocess_executor,
        ExecutionMode.CONTAINER: my_container_executor,
    },
)
```

Direct injection works without a registry for inference, persistence,
credentials, embeddings, tools, and executors. Registries are optional adapters
for settings-based selection.

`build_runtime()` is for synchronous callers. Use
`await build_runtime_async(...)` inside an existing event loop.

`RuntimeContext` closes resources it creates. Injected resources are not closed
unless ownership flags such as `own_storage=True`, `own_model=True`,
`own_embeddings=True`, `own_memory=True`, `own_secret_resolver=True`, or
`own_executors=True` are set.
