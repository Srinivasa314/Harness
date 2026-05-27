from __future__ import annotations

import asyncio
import json
import sys
from typing import cast

import httpx
import pytest
from pydantic import ValidationError

from harness.agent import NoopContextCompactor, RollingSummaryContextCompactor
from harness.config import HarnessSettings
from harness.execution import DockerContainerExecutor, load_container_schema_registry
from harness.memory import (
    EmbeddingProvider,
    EmbeddingProviderRegistry,
    HashEmbeddingProvider,
    MemoryManager,
    MemoryPolicy,
    MemoryStore,
    MiniLMEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from harness.models import (
    CodexCliProvider,
    ModelMessage,
    ModelProvider,
    ModelProviderRegistry,
    ModelResponse,
)
from harness.models import codex as codex_module
from harness.models.openai import OpenAIResponsesProvider, _extract_text, _to_response_input
from harness.runtime import build_runtime, build_runtime_async
from harness.schemas import ContainerSchema, ExecutionMode, ToolCall
from harness.storage import PostgresStorage, SQLiteStorage, StorageBackendRegistry, create_storage
from harness.tools import (
    EnvSecretResolver,
    SecretResolverRegistry,
    create_secret_resolver,
)


class StaticResponseModel(ModelProvider):
    def __init__(self, content: str, metadata: dict | None = None) -> None:
        self.content = content
        self.metadata = metadata or {}

    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        _ = messages
        return ModelResponse(content=self.content, metadata=self.metadata)


class StaticEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [1.0]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vector for _text in texts]


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs):  # noqa: ANN001
        self.calls.append({"texts": texts, "kwargs": kwargs})
        return [[float(len(text)), 1.0] for text in texts]


def test_tool_call_requires_session_id():
    with pytest.raises(ValidationError):
        ToolCall.model_validate({"name": "tool.echo"})


def test_storage_factory_sqlite(tmp_path):
    storage = create_storage(
        HarnessSettings(
            storage_backend="sqlite",
            sqlite_path=tmp_path / "harness.sqlite3",
        )
    )

    assert isinstance(storage, SQLiteStorage)


def test_storage_factory_postgres():
    storage = create_storage(
        HarnessSettings(
            storage_backend="postgres",
            postgres_dsn="postgresql://user:pass@localhost/db",
        )
    )

    assert isinstance(storage, PostgresStorage)


def test_custom_storage_backend_factory(tmp_path):
    registry = StorageBackendRegistry()
    registry.register("custom-sqlite", lambda settings: SQLiteStorage(settings.sqlite_path))

    storage = create_storage(
        HarnessSettings(
            storage_backend="custom-sqlite",
            sqlite_path=tmp_path / "custom.sqlite3",
        ),
        registry=registry,
    )

    assert isinstance(storage, SQLiteStorage)


def test_settings_loads_env_development(tmp_path, monkeypatch):
    env_path = tmp_path / ".env.development"
    env_path.write_text(
        "\n".join(
            [
                "HARNESS_OPENAI_API_KEY=test-key",
                "HARNESS_DOCKER_BIN=/usr/local/bin/docker",
                "HARNESS_EMBEDDING_PROVIDER=hash",
                "HARNESS_OPENAI_EMBEDDING_MODEL=text-embedding-test",
                "HARNESS_OPENAI_EMBEDDING_DIMENSIONS=64",
                "HARNESS_SECRET_ENV_PREFIX=APP_SECRET_",
                "HARNESS_CONTEXT_COMPACTION_SUMMARIZER_INPUT_MAX_CHARS=456",
            ]
        )
    )
    monkeypatch.delenv("HARNESS_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HARNESS_DOCKER_BIN", raising=False)
    monkeypatch.delenv("HARNESS_OPENAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("HARNESS_OPENAI_EMBEDDING_DIMENSIONS", raising=False)
    monkeypatch.delenv("HARNESS_SECRET_ENV_PREFIX", raising=False)
    monkeypatch.delenv("HARNESS_CONTEXT_COMPACTION_SUMMARIZER_INPUT_MAX_CHARS", raising=False)
    monkeypatch.delenv("HARNESS_DISABLE_ENV_FILES", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = HarnessSettings()

    assert settings.openai_api_key == "test-key"
    assert settings.docker_bin == "/usr/local/bin/docker"
    assert settings.embedding_provider == "hash"
    assert settings.openai_embedding_model == "text-embedding-test"
    assert settings.openai_embedding_dimensions == 64
    assert settings.secret_env_prefix == "APP_SECRET_"
    assert settings.context_compaction_summarizer_input_max_chars == 456


def test_secret_resolver_factory_env():
    settings = HarnessSettings(secret_backend="env", secret_env_prefix="APP_SECRET_")

    resolver = create_secret_resolver(settings)

    assert isinstance(resolver, EnvSecretResolver)


def test_env_secret_resolver_rejects_empty_prefix():
    with pytest.raises(ValueError, match="prefix"):
        EnvSecretResolver(prefix="")


@pytest.mark.anyio
async def test_env_secret_resolver_normalizes_secret_names(monkeypatch):
    monkeypatch.setenv("HARNESS_SECRET_API_TOKEN", "secret-value")
    resolver = EnvSecretResolver()

    resolved = await resolver.resolve(["api-token"])

    assert resolved == {"api-token": "secret-value"}


def test_secret_resolver_factory_none():
    settings = HarnessSettings(secret_backend="none")

    assert create_secret_resolver(settings) is None


def test_custom_secret_resolver_factory():
    registry = SecretResolverRegistry()
    registry.register(
        "custom-env",
        lambda settings: EnvSecretResolver(prefix=settings.secret_env_prefix),
    )

    resolver = create_secret_resolver(
        HarnessSettings(secret_backend="custom-env", secret_env_prefix="APP_SECRET_"),
        registry=registry,
    )

    assert isinstance(resolver, EnvSecretResolver)


def test_docker_container_executor_uses_configured_binary():
    executor = DockerContainerExecutor(docker_bin="/usr/local/bin/docker")

    assert executor.docker_bin == "/usr/local/bin/docker"


def test_container_schema_registry_loads_config(tmp_path):
    path = tmp_path / "container-schemas.json"
    path.write_text(
        json.dumps(
            {
                "schemas": [
                    {
                        "name": "python-dev",
                        "image": "python:3.12-alpine",
                        "network": True,
                        "allow_secrets": True,
                        "read_only_root": True,
                    }
                ]
            }
        )
    )

    registry = load_container_schema_registry(path)
    schema = registry.get("python-dev")

    assert schema.image == "python:3.12-alpine"
    assert schema.network is True
    assert schema.allow_secrets is True


def test_secret_enabled_container_schema_requires_read_only_root():
    with pytest.raises(ValueError, match="read_only_root"):
        ContainerSchema.model_validate(
            {
                "name": "secret",
                "image": "python:3.12-alpine",
                "allow_secrets": True,
                "read_only_root": False,
            }
        )


def test_secret_enabled_container_schema_rejects_host_mount(tmp_path):
    with pytest.raises(ValueError, match="host mounts"):
        ContainerSchema.model_validate(
            {
                "name": "secret",
                "image": "python:3.12-alpine",
                "allow_secrets": True,
                "read_only_root": True,
                "mount": tmp_path,
            }
        )


def test_persistent_secret_container_schema_requires_secret_schema_hardening():
    schema = ContainerSchema.model_validate(
        {
            "name": "secret",
            "image": "python:3.12-alpine",
            "allow_secrets": True,
            "persistent_secrets": True,
        }
    )

    assert schema.persistent_secrets is True
    assert schema.read_only_root is True
    assert schema.tmpfs_tmp is True
    assert schema.tmpfs_workdir is True


def test_persistent_secret_container_schema_requires_allow_secrets():
    with pytest.raises(ValueError, match="persistent_secrets"):
        ContainerSchema.model_validate(
            {
                "name": "bad",
                "image": "python:3.12-alpine",
                "persistent_secrets": True,
            }
        )


def test_persistent_secret_container_schema_requires_tmpfs_workspace():
    with pytest.raises(ValueError, match="tmpfs"):
        ContainerSchema.model_validate(
            {
                "name": "bad",
                "image": "python:3.12-alpine",
                "allow_secrets": True,
                "persistent_secrets": True,
                "tmpfs_workdir": False,
            }
        )


def test_container_schema_rejects_flag_like_image_name():
    with pytest.raises(ValueError, match="must not start"):
        ContainerSchema.model_validate(
            {
                "name": "bad",
                "image": "--privileged",
            }
        )


def test_runtime_loads_container_schema_config(tmp_path):
    path = tmp_path / "container-schemas.json"
    path.write_text(
        json.dumps(
            {
                "schemas": [
                    {
                        "name": "python-dev",
                        "image": "python:3.12-alpine",
                    }
                ]
            }
        )
    )

    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "harness.sqlite3",
            container_schemas_path=path,
        )
    )

    executor = runtime.gateway.executors[ExecutionMode.CONTAINER]
    assert isinstance(executor, DockerContainerExecutor)
    assert executor.schemas.get("python-dev").image == "python:3.12-alpine"


def test_runtime_uses_configured_capability_grants(tmp_path):
    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "harness.sqlite3",
            tool_capabilities=["text:uppercase"],
        )
    )

    assert runtime.gateway.policy.check(["text:uppercase"])
    assert not runtime.gateway.policy.check(["host:write"])


def test_runtime_default_capability_grant_is_empty(tmp_path):
    runtime = build_runtime(settings=HarnessSettings(sqlite_path=tmp_path / "harness.sqlite3"))

    assert not runtime.gateway.policy.check(["host:write"])


def test_runtime_builds_configured_openai_model(tmp_path):
    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "harness.sqlite3",
            model_provider="openai",
            openai_api_key="test-key",
            openai_model="gpt-test",
            openai_base_url="https://example.test/v1",
        )
    )

    assert isinstance(runtime.model, OpenAIResponsesProvider)
    assert runtime.model.model == "gpt-test"
    assert runtime.model.base_url == "https://example.test/v1"
    loop = runtime.agent_loop()
    assert isinstance(loop.model, OpenAIResponsesProvider)


def test_runtime_agent_loop_allows_context_compactor_override(tmp_path):
    compactor = NoopContextCompactor()
    runtime = build_runtime(
        settings=HarnessSettings(sqlite_path=tmp_path / "harness.sqlite3"),
        model=StaticResponseModel('{"final": "done"}'),
    )

    loop = runtime.agent_loop(context_compactor=compactor)

    assert loop.context_compactor is compactor


def test_runtime_agent_loop_wires_model_backed_context_compactor(tmp_path):
    model = StaticResponseModel('{"final": "done"}')
    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "harness.sqlite3",
            context_compaction_enabled=False,
            context_max_chars=1234,
            context_compaction_trigger_ratio=0.5,
            context_compaction_preserve_recent_messages=3,
            context_compaction_summarizer_input_max_chars=789,
            context_compaction_summary_max_chars=222,
        ),
        model=model,
    )

    loop = runtime.agent_loop()

    assert isinstance(loop.context_compactor, RollingSummaryContextCompactor)
    assert loop.context_compactor.model is model
    assert not loop.context_compactor.policy.enabled
    assert loop.context_compactor.policy.max_context_chars == 1234
    assert loop.context_compactor.policy.trigger_ratio == 0.5
    assert loop.context_compactor.policy.preserve_recent_messages == 3
    assert loop.context_compactor.policy.summarizer_input_max_chars == 789
    assert loop.context_compactor.policy.summary_max_chars == 222


def test_runtime_builds_configured_codex_model(tmp_path):
    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "harness.sqlite3",
            model_provider="codex",
            codex_command=["codex", "exec"],
            codex_model="gpt-test",
        )
    )

    assert isinstance(runtime.model, CodexCliProvider)
    assert runtime.model.model == "gpt-test"


def test_runtime_preserves_codex_safety_flags_for_absolute_cli_path(tmp_path):
    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "harness.sqlite3",
            model_provider="codex",
            codex_command=["/opt/bin/codex", "exec"],
        )
    )

    assert isinstance(runtime.model, CodexCliProvider)
    assert runtime.model.use_safe_defaults


def test_runtime_default_model_provider_is_none(tmp_path):
    runtime = build_runtime(settings=HarnessSettings(sqlite_path=tmp_path / "harness.sqlite3"))

    assert runtime.model is None
    with pytest.raises(ValueError, match="configured or injected model"):
        runtime.agent_loop()


def test_runtime_rejects_removed_command_model_provider(tmp_path):
    with pytest.raises(ValueError, match="Unknown model provider: command"):
        build_runtime(
            settings=HarnessSettings(
                sqlite_path=tmp_path / "harness.sqlite3",
                model_provider="command",
            )
        )


def test_runtime_builds_model_from_custom_registry(tmp_path):
    registry = ModelProviderRegistry()
    registry.register("fast", lambda _settings: StaticResponseModel("fast"))

    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "harness.sqlite3",
            model_provider="fast",
        ),
        model_registry=registry,
    )

    assert isinstance(runtime.model, StaticResponseModel)


def test_runtime_builds_default_memory_from_settings(tmp_path):
    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "harness.sqlite3",
            embedding_provider="hash",
            embedding_dimensions=8,
        )
    )

    assert isinstance(runtime.embeddings, HashEmbeddingProvider)
    assert runtime.embeddings.dimensions == 8
    assert runtime.memory is not None


def test_runtime_default_embedding_provider_is_minilm(tmp_path):
    runtime = build_runtime(settings=HarnessSettings(sqlite_path=tmp_path / "harness.sqlite3"))

    assert isinstance(runtime.embeddings, MiniLMEmbeddingProvider)
    assert runtime.embeddings.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_runtime_builds_embeddings_from_custom_registry(tmp_path):
    registry = EmbeddingProviderRegistry()
    registry.register("static", lambda _settings: StaticEmbeddingProvider([0.5]))

    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "harness.sqlite3",
            embedding_provider="static",
        ),
        embedding_registry=registry,
    )

    assert isinstance(runtime.embeddings, StaticEmbeddingProvider)
    assert runtime.memory is not None


def test_runtime_builds_configured_openai_embeddings(tmp_path):
    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "harness.sqlite3",
            embedding_provider="openai",
            openai_api_key="test-key",
            openai_base_url="https://example.test/v1",
            openai_embedding_model="text-embedding-test",
            openai_embedding_dimensions=64,
        )
    )

    assert isinstance(runtime.embeddings, OpenAIEmbeddingProvider)
    assert runtime.embeddings.model == "text-embedding-test"
    assert runtime.embeddings.base_url == "https://example.test/v1"
    assert runtime.embeddings.dimensions == 64
    assert runtime.memory is not None


def test_runtime_uses_direct_embedding_provider_without_registry(tmp_path):
    embeddings = StaticEmbeddingProvider([0.25])

    runtime = build_runtime(
        settings=HarnessSettings(sqlite_path=tmp_path / "harness.sqlite3"),
        embeddings=embeddings,
    )

    assert runtime.embeddings is embeddings
    assert runtime.memory is not None
    assert runtime.memory.embeddings is embeddings


def test_runtime_uses_direct_memory_manager_without_registry(tmp_path):
    storage = SQLiteStorage(tmp_path / "harness.sqlite3")
    embeddings = StaticEmbeddingProvider([0.25])
    memory = MemoryManager(
        MemoryStore(storage, embeddings),
        MemoryPolicy(namespace="custom"),
    )

    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "unused.sqlite3",
            memory_enabled=False,
        ),
        storage=storage,
        memory=memory,
    )

    assert runtime.memory is memory
    assert runtime.embeddings is None


def test_runtime_injected_memory_bypasses_embedding_provider_settings(tmp_path):
    storage = SQLiteStorage(tmp_path / "harness.sqlite3")
    memory = MemoryManager(
        MemoryStore(storage, StaticEmbeddingProvider([0.25])),
        MemoryPolicy(namespace="custom"),
    )

    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "unused.sqlite3",
            embedding_provider="missing",
        ),
        storage=storage,
        memory=memory,
    )

    assert runtime.memory is memory
    assert runtime.embeddings is None


def test_runtime_rejects_unknown_embedding_provider(tmp_path):
    with pytest.raises(ValueError, match="Unknown embedding provider: missing"):
        build_runtime(
            settings=HarnessSettings(
                sqlite_path=tmp_path / "harness.sqlite3",
                embedding_provider="missing",
            )
        )


def test_runtime_rejects_openai_embeddings_without_key(tmp_path):
    with pytest.raises(ValueError, match="HARNESS_OPENAI_API_KEY"):
        build_runtime(
            settings=HarnessSettings(
                sqlite_path=tmp_path / "harness.sqlite3",
                embedding_provider="openai",
                openai_api_key=None,
            )
        )


def test_runtime_does_not_build_embeddings_when_memory_disabled(tmp_path):
    runtime = build_runtime(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "harness.sqlite3",
            memory_enabled=False,
            embedding_provider="missing",
        )
    )

    assert runtime.embeddings is None
    assert runtime.memory is None


def test_runtime_allows_empty_executor_map(tmp_path):
    runtime = build_runtime(
        settings=HarnessSettings(sqlite_path=tmp_path / "harness.sqlite3"),
        executors={},
    )

    assert runtime.gateway.executors == {}
    assert ExecutionMode.IN_PROCESS not in runtime.gateway.executors


def test_runtime_rejects_missing_openai_key(tmp_path):
    with pytest.raises(ValueError, match="HARNESS_OPENAI_API_KEY"):
        build_runtime(
            settings=HarnessSettings(
                sqlite_path=tmp_path / "harness.sqlite3",
                model_provider="openai",
                openai_api_key=None,
            )
        )


def test_runtime_closes_created_resources_when_build_fails(tmp_path):
    created: list[SQLiteStorage] = []

    class ClosableStorage(SQLiteStorage):
        def __init__(self, path) -> None:  # noqa: ANN001
            super().__init__(path)
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    storage_registry = StorageBackendRegistry()
    storage_registry.register(
        "tracked",
        lambda settings: created.append(ClosableStorage(settings.sqlite_path)) or created[-1],
    )
    embedding_registry = EmbeddingProviderRegistry()
    embedding_registry.register(
        "broken",
        lambda _settings: (_ for _ in ()).throw(RuntimeError("embedding failed")),
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        build_runtime(
            settings=HarnessSettings(
                storage_backend="tracked",
                sqlite_path=tmp_path / "harness.sqlite3",
                embedding_provider="broken",
            ),
            storage_registry=storage_registry,
            embedding_registry=embedding_registry,
        )

    assert len(created) == 1
    assert cast(ClosableStorage, created[0]).closed


@pytest.mark.anyio
async def test_runtime_closes_created_resources_on_build_failure_inside_event_loop(tmp_path):
    created: list[SQLiteStorage] = []
    loop = asyncio.get_running_loop()

    class ClosableStorage(SQLiteStorage):
        def __init__(self, path) -> None:  # noqa: ANN001
            super().__init__(path)
            self.closed = False

        async def close(self) -> None:
            assert asyncio.get_running_loop() is loop
            self.closed = True

    storage_registry = StorageBackendRegistry()
    storage_registry.register(
        "tracked",
        lambda settings: created.append(ClosableStorage(settings.sqlite_path)) or created[-1],
    )
    embedding_registry = EmbeddingProviderRegistry()
    embedding_registry.register(
        "broken",
        lambda _settings: (_ for _ in ()).throw(RuntimeError("embedding failed")),
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        await build_runtime_async(
            settings=HarnessSettings(
                storage_backend="tracked",
                sqlite_path=tmp_path / "harness.sqlite3",
                embedding_provider="broken",
            ),
            storage_registry=storage_registry,
            embedding_registry=embedding_registry,
        )

    assert len(created) == 1
    assert cast(ClosableStorage, created[0]).closed


@pytest.mark.anyio
async def test_sync_runtime_factory_rejects_running_event_loop(tmp_path):
    with pytest.raises(RuntimeError, match="build_runtime_async"):
        build_runtime(settings=HarnessSettings(sqlite_path=tmp_path / "harness.sqlite3"))


@pytest.mark.anyio
async def test_runtime_close_closes_owned_resources(tmp_path):
    class ClosableModel(ModelProvider):
        def __init__(self) -> None:
            self.closed = False

        async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
            _ = messages
            return ModelResponse(content='{"final": "ok"}')

        async def close(self) -> None:
            self.closed = True

    model = ClosableModel()
    runtime = await build_runtime_async(
        settings=HarnessSettings(sqlite_path=tmp_path / "harness.sqlite3"),
        model=model,
        own_model=True,
    )

    await runtime.close()

    assert model.closed


@pytest.mark.anyio
async def test_runtime_close_closes_owned_memory_once(tmp_path):
    class ClosableMemory(MemoryManager):
        def __init__(self, storage: SQLiteStorage) -> None:
            super().__init__(
                MemoryStore(storage, StaticEmbeddingProvider([0.5])),
                MemoryPolicy(),
            )
            self.close_count = 0

        async def close(self) -> None:
            self.close_count += 1

    storage = SQLiteStorage(tmp_path / "harness.sqlite3")
    memory = ClosableMemory(storage)
    runtime = await build_runtime_async(
        settings=HarnessSettings(
            sqlite_path=tmp_path / "unused.sqlite3",
            memory_enabled=False,
        ),
        storage=storage,
        memory=memory,
        own_memory=True,
    )

    await runtime.close()
    await runtime.close()

    assert memory.close_count == 1


@pytest.mark.anyio
async def test_runtime_does_not_close_injected_resources_by_default(tmp_path):
    class ClosableModel(ModelProvider):
        def __init__(self) -> None:
            self.closed = False

        async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
            _ = messages
            return ModelResponse(content='{"final": "ok"}')

        async def close(self) -> None:
            self.closed = True

    class ClosableStorage(SQLiteStorage):
        def __init__(self, path) -> None:  # noqa: ANN001
            super().__init__(path)
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    model = ClosableModel()
    storage = ClosableStorage(tmp_path / "harness.sqlite3")
    runtime = await build_runtime_async(
        settings=HarnessSettings(sqlite_path=tmp_path / "harness.sqlite3"),
        storage=storage,
        model=model,
    )

    await runtime.close()

    assert not model.closed
    assert not storage.closed


@pytest.mark.anyio
async def test_runtime_close_continues_after_close_failure(tmp_path):
    class FailingModel(ModelProvider):
        async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
            _ = messages
            return ModelResponse(content='{"final": "ok"}')

        async def close(self) -> None:
            raise RuntimeError("close failed")

    class ClosableStorage(SQLiteStorage):
        def __init__(self, path) -> None:  # noqa: ANN001
            super().__init__(path)
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    storage = ClosableStorage(tmp_path / "harness.sqlite3")
    runtime = await build_runtime_async(
        settings=HarnessSettings(sqlite_path=tmp_path / "unused.sqlite3"),
        storage=storage,
        model=FailingModel(),
        own_storage=True,
        own_model=True,
    )

    with pytest.raises(RuntimeError, match="close failed"):
        await runtime.close()

    assert storage.closed


@pytest.mark.anyio
async def test_runtime_close_can_retry_after_close_failure(tmp_path):
    class FlakyModel(ModelProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
            _ = messages
            return ModelResponse(content='{"final": "ok"}')

        async def close(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("close failed once")

    model = FlakyModel()
    runtime = await build_runtime_async(
        settings=HarnessSettings(sqlite_path=tmp_path / "harness.sqlite3"),
        model=model,
        own_model=True,
    )

    with pytest.raises(RuntimeError, match="close failed once"):
        await runtime.close()
    await runtime.close()
    await runtime.close()

    assert model.calls == 2


def test_model_provider_registry_supports_multiple_models(tmp_path):
    _ = tmp_path
    registry = ModelProviderRegistry()
    registry.register("fast", lambda _settings: StaticResponseModel("fast"))
    registry.register("judge", lambda _settings: StaticResponseModel("judge"))

    assert registry.names() == ["fast", "judge"]
    assert isinstance(registry.create("fast", HarnessSettings()), StaticResponseModel)


def test_embedding_provider_registry_supports_multiple_implementations():
    registry = EmbeddingProviderRegistry()
    registry.register("small", lambda _settings: HashEmbeddingProvider(dimensions=8))
    registry.register("large", lambda _settings: HashEmbeddingProvider(dimensions=64))

    assert registry.names() == ["large", "small"]
    assert isinstance(registry.create("large", HarnessSettings()), HashEmbeddingProvider)


@pytest.mark.anyio
async def test_minilm_embedding_provider_uses_sentence_transformer_model():
    model = FakeSentenceTransformer()
    provider = MiniLMEmbeddingProvider(model=model)

    embeddings = await provider.embed(["a", "abcd"])

    assert embeddings == [[1.0, 1.0], [4.0, 1.0]]
    assert model.calls == [
        {
            "texts": ["a", "abcd"],
            "kwargs": {"normalize_embeddings": True, "convert_to_numpy": True},
        }
    ]


@pytest.mark.anyio
async def test_openai_embedding_provider_sends_embeddings_request():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ],
                "model": "text-embedding-test",
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
            headers={"x-request-id": "req_123"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-test",
            base_url="https://example.test/v1",
            dimensions=2,
            client=client,
        )
        vectors = await provider.embed(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    [request] = requests
    assert str(request.url) == "https://example.test/v1/embeddings"
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "text-embedding-test",
        "input": ["first", "second"],
        "encoding_format": "float",
        "dimensions": 2,
    }


@pytest.mark.anyio
async def test_openai_embedding_provider_returns_empty_for_empty_input():
    provider = OpenAIEmbeddingProvider(api_key="test-key")

    assert await provider.embed([]) == []


@pytest.mark.anyio
async def test_openai_embedding_provider_raises_for_incomplete_response():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIEmbeddingProvider(api_key="test-key", client=client)
        with pytest.raises(RuntimeError, match="one vector per input"):
            await provider.embed(["first", "second"])


@pytest.mark.anyio
async def test_openai_embedding_provider_rejects_unexpected_indexes():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.1]},
                    {"index": 2, "embedding": [0.2]},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIEmbeddingProvider(api_key="test-key", client=client)
        with pytest.raises(RuntimeError, match="one vector per input"):
            await provider.embed(["first", "second"])


@pytest.mark.anyio
async def test_openai_embedding_provider_raises_for_non_success_response():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIEmbeddingProvider(api_key="test-key", client=client)
        with pytest.raises(httpx.HTTPStatusError):
            await provider.embed(["hello"])


@pytest.mark.anyio
async def test_codex_cli_provider_passes_model_and_prompt(tmp_path, monkeypatch):
    script = tmp_path / "codex.py"
    script.write_text(
        """
import json
import os
import sys

prompt = sys.stdin.read()
print(json.dumps({
    "argv": sys.argv[1:],
    "prompt": prompt,
    "harness_secret": os.environ.get("HARNESS_SECRET_API_TOKEN"),
    "codex_home": os.environ.get("CODEX_HOME"),
}))
""".strip()
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("HARNESS_SECRET_API_TOKEN", "super-secret-value")
    provider = CodexCliProvider(command=[sys.executable, str(script)], model="gpt-test")

    response = await provider.complete([ModelMessage(role="user", content="hello")])

    payload = json.loads(response.content)
    assert payload["argv"][0:2] == ["--model", "gpt-test"]
    assert payload["argv"][-1] == "-"
    assert payload["prompt"] == "user: hello"
    assert payload["harness_secret"] is None
    assert payload["codex_home"] == str(tmp_path / "codex-home")
    assert response.metadata["provider"] == "codex_cli"


@pytest.mark.anyio
async def test_codex_cli_default_provider_uses_read_only_stdin_mode(monkeypatch):
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = b'{"final": "ok"}'
        stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        captured["command"] = command
        captured["input"] = kwargs["input"]
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs["cwd"]
        return Completed()

    monkeypatch.setenv("HARNESS_SECRET_API_TOKEN", "super-secret-value")
    monkeypatch.setattr(codex_module.anyio, "run_process", fake_run_process)
    provider = CodexCliProvider(model="gpt-test")

    response = await provider.complete([ModelMessage(role="user", content="hello")])

    command = cast(list[str], captured["command"])
    env = cast(dict[str, str], captured["env"])
    assert command[0:2] == ["codex", "exec"]
    assert "--sandbox" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command
    assert "--ignore-rules" in command
    assert command[-1] == "-"
    assert captured["input"] == b"user: hello"
    assert captured["cwd"] is not None
    assert "HARNESS_SECRET_API_TOKEN" not in env
    assert response.content == '{"final": "ok"}'
    assert response.metadata["safe_defaults"] is True


@pytest.mark.anyio
async def test_codex_cli_explicit_default_command_uses_safe_defaults(monkeypatch):
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = b'{"final": "ok"}'
        stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        return Completed()

    monkeypatch.setattr(codex_module.anyio, "run_process", fake_run_process)
    provider = CodexCliProvider(command=["codex", "exec"])

    response = await provider.complete([ModelMessage(role="user", content="hello")])

    command = cast(list[str], captured["command"])
    assert "--sandbox" in command
    assert "--ephemeral" in command
    assert captured["cwd"] is not None
    assert response.metadata["safe_defaults"] is True


@pytest.mark.anyio
async def test_codex_cli_provider_failure(tmp_path):
    script = tmp_path / "codex.py"
    script.write_text(
        "import sys; sys.stdin.read(); print('codex failed', file=sys.stderr); raise SystemExit(2)"
    )
    provider = CodexCliProvider(command=[sys.executable, str(script)])

    with pytest.raises(RuntimeError, match="codex failed"):
        await provider.complete([ModelMessage(role="user", content="hello")])


@pytest.mark.anyio
async def test_openai_provider_sends_responses_request_and_metadata():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_123",
                "output_text": "hello",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            headers={"x-request-id": "req_123"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider(
            api_key="test-key",
            model="gpt-test",
            base_url="https://example.test/v1",
            client=client,
        )
        response = await provider.complete([ModelMessage(role="user", content="hello")])

    assert response.content == "hello"
    assert response.metadata["response_id"] == "resp_123"
    assert response.metadata["request_id"] == "req_123"
    assert response.metadata["usage"] == {"input_tokens": 1, "output_tokens": 1}
    [request] = requests
    assert str(request.url) == "https://example.test/v1/responses"
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "gpt-test",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
    }


@pytest.mark.anyio
async def test_openai_provider_raises_for_non_success_response():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider(
            api_key="test-key",
            model="gpt-test",
            base_url="https://example.test/v1",
            client=client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await provider.complete([ModelMessage(role="user", content="hello")])


def test_openai_response_text_extraction_output_text():
    assert _extract_text({"output_text": "hello"}) == "hello"


def test_openai_response_text_extraction_output_items():
    assert (
        _extract_text(
            {
                "output": [
                    {
                        "content": [
                            {"text": "hello "},
                            {"text": "world"},
                        ]
                    }
                ]
            }
        )
        == "hello world"
    )


def test_openai_provider_normalizes_tool_role_messages():
    item = _to_response_input(ModelMessage(role="tool", content='[{"output": "ok"}]'))

    assert item["role"] == "user"
    assert item["content"][0]["text"].startswith("Tool result:")


def test_openai_provider_uses_output_text_for_assistant_history():
    item = _to_response_input(ModelMessage(role="assistant", content="prior answer"))

    assert item == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "prior answer"}],
    }
