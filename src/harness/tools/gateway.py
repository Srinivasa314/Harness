from __future__ import annotations

import inspect

import anyio
from jsonschema import ValidationError, validate

from harness.execution.base import ToolExecutor
from harness.execution.container import DockerContainerExecutor
from harness.execution.in_process import InProcessExecutor
from harness.execution.subprocess import SubprocessExecutor
from harness.observability.sink import EventSink
from harness.schemas import ArtifactRecord, ExecutionMode, ToolCall, ToolResult, utc_now
from harness.storage.base import StorageBackend
from harness.tools.policy import CapabilityPolicy
from harness.tools.redaction import redact, sensitive_values
from harness.tools.registry import ToolRegistry
from harness.tools.secrets import SecretResolver


class ToolExecutionGateway:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: CapabilityPolicy,
        storage: StorageBackend,
        secret_resolver: SecretResolver | None = None,
        executors: dict[ExecutionMode, ToolExecutor] | None = None,
        own_executors: bool | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.storage = storage
        self.secret_resolver = secret_resolver
        self.events = EventSink(storage)
        created_executors = executors is None
        if executors is None:
            executors = {
                ExecutionMode.IN_PROCESS: InProcessExecutor(registry.function_for),
                ExecutionMode.SUBPROCESS: SubprocessExecutor(),
                ExecutionMode.CONTAINER: DockerContainerExecutor(),
            }
        self.executors = executors
        self._own_executors = created_executors if own_executors is None else own_executors

    async def execute(self, call: ToolCall) -> ToolResult:
        started = utc_now()
        try:
            definition = self.registry.get(call.name)
        except KeyError as exc:
            return await self._finish_denied(call, str(exc), started=started)

        await self.events.emit(
            "tool.call.started",
            session_id=call.session_id,
            call_id=call.call_id,
            tool=call.name,
            execution_mode=definition.execution_mode.value,
        )

        try:
            validate(instance=call.arguments, schema=definition.input_schema)
        except ValidationError as exc:
            return await self._finish_denied(
                call,
                f"Invalid tool input: {exc.message}",
                started=started,
                emit_finished=True,
            )

        if not self.policy.check(definition.required_capabilities):
            return await self._finish_denied(
                call,
                "Missing required capability",
                started=started,
                emit_finished=True,
            )

        secrets: dict[str, str] = {}
        executor = self.executors.get(definition.execution_mode)
        if executor is None:
            result = ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                error=f"No executor configured for mode: {definition.execution_mode.value}",
                started_at=started,
                ended_at=utc_now(),
            )
        else:
            if definition.required_secrets:
                validate_secret_access = getattr(executor, "validate_secret_access", None)
                if validate_secret_access is not None:
                    denial = validate_secret_access(definition)
                    if denial is not None:
                        return await self._finish_denied(
                            call,
                            denial,
                            started=started,
                            emit_finished=True,
                        )
                if self.secret_resolver is None:
                    return await self._finish_denied(
                        call,
                        "No secret resolver configured",
                        started=started,
                        emit_finished=True,
                    )
                try:
                    secrets = await self.secret_resolver.resolve(definition.required_secrets)
                except KeyError as exc:
                    return await self._finish_denied(
                        call,
                        str(exc),
                        started=started,
                        emit_finished=True,
                    )
                await self.events.emit(
                    "tool.secret.resolved",
                    session_id=call.session_id,
                    call_id=call.call_id,
                    tool=call.name,
                    secret_names=list(secrets),
                )
            try:
                result = await executor.execute(definition, call, secrets)
            except Exception as exc:  # noqa: BLE001 - keep tool failures inside observability.
                result = ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    status="error",
                    error=f"Executor failed: {exc}",
                    started_at=started,
                    ended_at=utc_now(),
                )
        if result.status == "ok" and definition.output_schema is not None:
            try:
                validate(instance=result.output, schema=definition.output_schema)
            except ValidationError as exc:
                result = result.model_copy(
                    update={
                        "status": "error",
                        "error": f"Invalid tool output: {exc.message}",
                        "ended_at": utc_now(),
                    }
                )
        result_payload = {
            "output": result.output,
            "artifacts": result.artifacts,
            "error": result.error,
            "metadata": result.metadata,
        }
        redaction_secrets = {
            **sensitive_values(call.arguments),
            **sensitive_values(result_payload),
            **secrets,
        }
        result.output = redact(result.output, redaction_secrets)
        result.artifacts = redact(result.artifacts, redaction_secrets)
        result.error = redact(result.error, redaction_secrets)
        result.metadata = redact(result.metadata, redaction_secrets)
        persisted_call = call.model_copy(
            update={"arguments": redact(call.arguments, redaction_secrets)}
        )
        await self.storage.record_tool_call(persisted_call, result)
        await self._record_artifacts(call, result)
        await self.events.emit(
            "tool.call.finished",
            session_id=call.session_id,
            call_id=call.call_id,
            tool=call.name,
            status=result.status,
            error=result.error,
        )
        return result

    async def execute_many(
        self,
        calls: list[ToolCall],
        *,
        max_concurrency: int = 8,
    ) -> list[ToolResult]:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        limiter = anyio.Semaphore(max_concurrency)
        results: list[ToolResult | None] = [None] * len(calls)

        async def run(index: int, call: ToolCall) -> None:
            async with limiter:
                results[index] = await self.execute(call)

        async with anyio.create_task_group() as task_group:
            for index, call in enumerate(calls):
                task_group.start_soon(run, index, call)

        return [result for result in results if result is not None]

    async def end_session(self, session_id: str, *, cleanup_delay_minutes: float) -> None:
        for executor in self.executors.values():
            cleanup = getattr(executor, "schedule_session_cleanup", None)
            if cleanup is not None:
                await cleanup(session_id, cleanup_delay_minutes)

    async def aclose(self) -> None:
        if not self._own_executors:
            return
        first_error: BaseException | None = None
        for executor in reversed(list(self.executors.values())):
            close = getattr(executor, "aclose", None) or getattr(executor, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    async def _record_artifacts(self, call: ToolCall, result: ToolResult) -> None:
        for artifact_path in result.artifacts:
            await self.storage.save_artifact(
                ArtifactRecord(
                    session_id=call.session_id,
                    tool_call_id=call.call_id,
                    path=artifact_path,
                )
            )

    async def _finish_denied(
        self,
        call: ToolCall,
        error: str,
        *,
        started,
        emit_finished: bool = False,
    ) -> ToolResult:
        redaction_secrets = sensitive_values(call.arguments)
        redacted_error = redact(error, redaction_secrets)
        result = ToolResult(
            call_id=call.call_id,
            name=call.name,
            status="denied",
            error=redacted_error,
            started_at=started,
            ended_at=utc_now(),
        )
        persisted_call = call.model_copy(
            update={"arguments": redact(call.arguments, redaction_secrets)}
        )
        await self.storage.record_tool_call(persisted_call, result)
        await self.events.emit(
            "tool.call.denied",
            session_id=call.session_id,
            call_id=call.call_id,
            tool=call.name,
            error=redacted_error,
        )
        if emit_finished:
            await self.events.emit(
                "tool.call.finished",
                session_id=call.session_id,
                call_id=call.call_id,
                tool=call.name,
                status=result.status,
                error=redacted_error,
            )
        return result
