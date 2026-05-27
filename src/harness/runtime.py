from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from harness.agent import (
    AgentLoop,
    ContextCompactionPolicy,
    ContextCompactor,
    RollingSummaryContextCompactor,
)
from harness.agent.lease import SessionLeaseProvider, StorageSessionLeaseProvider
from harness.config import HarnessSettings, load_settings
from harness.execution.base import ToolExecutor
from harness.execution.container import DockerContainerExecutor, load_container_schema_registry
from harness.execution.in_process import InProcessExecutor
from harness.execution.subprocess import SubprocessExecutor
from harness.memory import (
    EmbeddingProvider,
    EmbeddingProviderRegistry,
    MemoryManager,
    MemoryPolicy,
    MemoryStore,
    create_embedding_provider,
)
from harness.models import (
    ModelProvider,
    ModelProviderRegistry,
    create_model_provider,
)
from harness.schemas import ExecutionMode
from harness.storage import StorageBackend, StorageBackendRegistry, create_storage
from harness.tools import (
    CapabilityGrant,
    CapabilityPolicy,
    SecretResolver,
    SecretResolverRegistry,
    ToolExecutionGateway,
    ToolRegistry,
    create_secret_resolver,
    load_tool_registry,
)

if TYPE_CHECKING:
    from harness.tools.registry import ToolFunction


@dataclass
class RuntimeContext:
    settings: HarnessSettings
    storage: StorageBackend
    registry: ToolRegistry
    gateway: ToolExecutionGateway
    model: ModelProvider | None = None
    embeddings: EmbeddingProvider | None = None
    memory: MemoryManager | None = None
    lease_provider: SessionLeaseProvider | None = None
    closeables: list[object] | None = None
    _closed: bool = False

    def agent_loop(
        self,
        *,
        model: ModelProvider | None = None,
        context_compactor: ContextCompactor | None = None,
        max_iterations: int = 8,
        stop_after_tools: set[str] | frozenset[str] | None = None,
    ) -> AgentLoop:
        if model is None:
            model = self.model
            if model is None:
                raise ValueError("agent_loop requires a configured or injected model")
        return AgentLoop(
            model=model,
            tools=self.gateway,
            storage=self.storage,
            max_iterations=max_iterations,
            stop_after_tools=stop_after_tools,
            container_cleanup_delay_minutes=self.settings.container_cleanup_delay_minutes,
            lease_provider=self.lease_provider,
            memory=self.memory,
            context_compactor=context_compactor
            or RollingSummaryContextCompactor(
                model,
                ContextCompactionPolicy(
                    enabled=self.settings.context_compaction_enabled,
                    max_context_chars=self.settings.context_max_chars,
                    trigger_ratio=self.settings.context_compaction_trigger_ratio,
                    preserve_recent_messages=(
                        self.settings.context_compaction_preserve_recent_messages
                    ),
                    summarizer_input_max_chars=(
                        self.settings.context_compaction_summarizer_input_max_chars
                    ),
                    summary_max_chars=self.settings.context_compaction_summary_max_chars,
                ),
            ),
        )

    async def close(self) -> None:
        if self._closed:
            return
        first_error: BaseException | None = None
        resources = self.closeables if self.closeables is not None else [self.storage]
        for resource in reversed(resources):
            try:
                await _close_if_supported(resource)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        self._closed = True


def build_runtime(
    *,
    settings: HarnessSettings | None = None,
    tools_path: Path | None = None,
    policy: CapabilityPolicy | None = None,
    model: ModelProvider | None = None,
    model_registry: ModelProviderRegistry | None = None,
    embeddings: EmbeddingProvider | None = None,
    embedding_registry: EmbeddingProviderRegistry | None = None,
    memory: MemoryManager | None = None,
    storage: StorageBackend | None = None,
    storage_registry: StorageBackendRegistry | None = None,
    secret_resolver: SecretResolver | None = None,
    secret_resolver_registry: SecretResolverRegistry | None = None,
    executors: dict[ExecutionMode, ToolExecutor] | None = None,
    registry: ToolRegistry | None = None,
    tool_builtins: dict[str, ToolFunction] | None = None,
    own_storage: bool | None = None,
    own_model: bool | None = None,
    own_embeddings: bool | None = None,
    own_memory: bool | None = None,
    own_secret_resolver: bool | None = None,
    own_executors: bool | None = None,
) -> RuntimeContext:
    _raise_if_running_loop()
    cleanup_on_failure: list[object] = []
    try:
        return _compose_runtime(
            settings=settings,
            tools_path=tools_path,
            policy=policy,
            model=model,
            model_registry=model_registry,
            embeddings=embeddings,
            embedding_registry=embedding_registry,
            memory=memory,
            storage=storage,
            storage_registry=storage_registry,
            secret_resolver=secret_resolver,
            secret_resolver_registry=secret_resolver_registry,
            executors=executors,
            registry=registry,
            tool_builtins=tool_builtins,
            own_storage=own_storage,
            own_model=own_model,
            own_embeddings=own_embeddings,
            own_memory=own_memory,
            own_secret_resolver=own_secret_resolver,
            own_executors=own_executors,
            cleanup_on_failure=cleanup_on_failure,
        )
    except BaseException:
        try:
            _close_resources_sync(cleanup_on_failure)
        except BaseException:
            pass
        raise


async def build_runtime_async(
    *,
    settings: HarnessSettings | None = None,
    tools_path: Path | None = None,
    policy: CapabilityPolicy | None = None,
    model: ModelProvider | None = None,
    model_registry: ModelProviderRegistry | None = None,
    embeddings: EmbeddingProvider | None = None,
    embedding_registry: EmbeddingProviderRegistry | None = None,
    memory: MemoryManager | None = None,
    storage: StorageBackend | None = None,
    storage_registry: StorageBackendRegistry | None = None,
    secret_resolver: SecretResolver | None = None,
    secret_resolver_registry: SecretResolverRegistry | None = None,
    executors: dict[ExecutionMode, ToolExecutor] | None = None,
    registry: ToolRegistry | None = None,
    tool_builtins: dict[str, ToolFunction] | None = None,
    own_storage: bool | None = None,
    own_model: bool | None = None,
    own_embeddings: bool | None = None,
    own_memory: bool | None = None,
    own_secret_resolver: bool | None = None,
    own_executors: bool | None = None,
) -> RuntimeContext:
    cleanup_on_failure: list[object] = []
    try:
        return _compose_runtime(
            settings=settings,
            tools_path=tools_path,
            policy=policy,
            model=model,
            model_registry=model_registry,
            embeddings=embeddings,
            embedding_registry=embedding_registry,
            memory=memory,
            storage=storage,
            storage_registry=storage_registry,
            secret_resolver=secret_resolver,
            secret_resolver_registry=secret_resolver_registry,
            executors=executors,
            registry=registry,
            tool_builtins=tool_builtins,
            own_storage=own_storage,
            own_model=own_model,
            own_embeddings=own_embeddings,
            own_memory=own_memory,
            own_secret_resolver=own_secret_resolver,
            own_executors=own_executors,
            cleanup_on_failure=cleanup_on_failure,
        )
    except BaseException:
        try:
            await _close_resources(cleanup_on_failure)
        except BaseException:
            pass
        raise


def _compose_runtime(
    *,
    settings: HarnessSettings | None,
    tools_path: Path | None,
    policy: CapabilityPolicy | None,
    model: ModelProvider | None,
    model_registry: ModelProviderRegistry | None,
    embeddings: EmbeddingProvider | None,
    embedding_registry: EmbeddingProviderRegistry | None,
    memory: MemoryManager | None,
    storage: StorageBackend | None,
    storage_registry: StorageBackendRegistry | None,
    secret_resolver: SecretResolver | None,
    secret_resolver_registry: SecretResolverRegistry | None,
    executors: dict[ExecutionMode, ToolExecutor] | None,
    registry: ToolRegistry | None,
    tool_builtins: dict[str, ToolFunction] | None,
    own_storage: bool | None,
    own_model: bool | None,
    own_embeddings: bool | None,
    own_memory: bool | None,
    own_secret_resolver: bool | None,
    own_executors: bool | None,
    cleanup_on_failure: list[object],
) -> RuntimeContext:
    resolved_settings = settings or load_settings()
    created_storage = storage is None
    resolved_storage = storage or create_storage(resolved_settings, registry=storage_registry)
    if _owns_resource(own_storage, created_storage):
        cleanup_on_failure.append(resolved_storage)
    if registry is None:
        registry = (
            load_tool_registry(tools_path, builtins=tool_builtins)
            if tools_path is not None
            else ToolRegistry()
        )
    created_model = model is None
    resolved_model = model if model is not None else create_model_provider(
        resolved_settings,
        registry=model_registry,
    )
    if resolved_model is not None and _owns_resource(own_model, created_model):
        cleanup_on_failure.append(resolved_model)
    created_embeddings = embeddings is None and memory is None and resolved_settings.memory_enabled
    resolved_embeddings = (
        embeddings
        if embeddings is not None
        else (
            create_embedding_provider(
                resolved_settings,
                registry=embedding_registry,
            )
            if memory is None and resolved_settings.memory_enabled
            else None
        )
    )
    if resolved_embeddings is not None and _owns_resource(own_embeddings, created_embeddings):
        cleanup_on_failure.append(resolved_embeddings)
    created_memory = memory is None and resolved_settings.memory_enabled
    resolved_memory = memory or (
        MemoryManager(
            MemoryStore(resolved_storage, resolved_embeddings),
            MemoryPolicy(
                enabled=resolved_settings.memory_enabled,
                namespace=resolved_settings.memory_namespace,
                retrieval_limit=resolved_settings.memory_retrieval_limit,
                min_score=resolved_settings.memory_min_score,
                max_context_chars=resolved_settings.memory_max_context_chars,
                auto_capture=resolved_settings.memory_auto_capture,
            ),
        )
        if resolved_settings.memory_enabled and resolved_embeddings is not None
        else None
    )
    if resolved_memory is not None and _owns_resource(own_memory, created_memory):
        cleanup_on_failure.append(resolved_memory)
    lease_provider = StorageSessionLeaseProvider(
        resolved_storage,
        ttl_seconds=resolved_settings.session_lease_ttl_seconds,
        heartbeat_seconds=resolved_settings.session_lease_heartbeat_seconds,
    )
    cleanup_on_failure.append(lease_provider)
    created_secret_resolver = secret_resolver is None
    resolved_secret_resolver = (
        secret_resolver if secret_resolver is not None else create_secret_resolver(
            resolved_settings,
            registry=secret_resolver_registry,
        )
    )
    if resolved_secret_resolver is not None and _owns_resource(
        own_secret_resolver,
        created_secret_resolver,
    ):
        cleanup_on_failure.append(resolved_secret_resolver)
    created_executors = executors is None
    if executors is None:
        container_schemas = (
            load_container_schema_registry(resolved_settings.container_schemas_path)
            if resolved_settings.container_schemas_path is not None
            else None
        )
        resolved_executors: dict[ExecutionMode, ToolExecutor] = {
            ExecutionMode.IN_PROCESS: InProcessExecutor(registry.function_for),
            ExecutionMode.SUBPROCESS: SubprocessExecutor(),
            ExecutionMode.CONTAINER: DockerContainerExecutor(
                docker_bin=resolved_settings.docker_bin,
                schemas=container_schemas,
                allowed_mount_root=resolved_settings.docker_mount_root,
            ),
        }
    else:
        resolved_executors = executors
    if _owns_resource(own_executors, created_executors):
        cleanup_on_failure.extend(resolved_executors.values())
    grant = CapabilityGrant(frozenset(resolved_settings.tool_capabilities))
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=policy or CapabilityPolicy(grant),
        storage=resolved_storage,
        secret_resolver=resolved_secret_resolver,
        executors=resolved_executors,
    )
    return RuntimeContext(
        settings=resolved_settings,
        storage=resolved_storage,
        registry=registry,
        gateway=gateway,
        model=resolved_model,
        embeddings=resolved_embeddings,
        memory=resolved_memory,
        lease_provider=lease_provider,
        closeables=cleanup_on_failure,
    )


async def _close_if_supported(resource: object) -> None:
    close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _owns_resource(override: bool | None, created: bool) -> bool:
    return created if override is None else override


def _close_resources_sync(resources: list[object]) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_close_resources(resources))
        return

    raise RuntimeError("Use build_runtime_async() when an event loop is already running")


def _raise_if_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("Use build_runtime_async() when an event loop is already running")




async def _close_resources(resources: list[object]) -> None:
    first_error: BaseException | None = None
    for resource in reversed(resources):
        try:
            await _close_if_supported(resource)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
