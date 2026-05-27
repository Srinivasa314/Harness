from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio

from harness.execution.base import ToolExecutor
from harness.execution.process import ProcessOutputLimitExceeded, run_limited_process
from harness.process_env import DEFAULT_ENV_ALLOWLIST, filtered_env
from harness.schemas import ContainerSchema, ToolCall, ToolDefinition, ToolResult, utc_now

DOCKER_ENV_ALLOWLIST = (
    *DEFAULT_ENV_ALLOWLIST,
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)
LABEL_MANAGED = "com.agentic-harness.managed"
LABEL_SESSION = "com.agentic-harness.session"
LABEL_SCHEMA = "com.agentic-harness.schema"
LABEL_KEY_SCOPE = "com.agentic-harness.key-scope"


class ContainerSchemaRegistry:
    def __init__(self, schemas: list[ContainerSchema] | None = None) -> None:
        self._schemas: dict[str, ContainerSchema] = {}
        for schema in schemas or []:
            self.register(schema)

    def register(self, schema: ContainerSchema, *, replace: bool = False) -> None:
        if not replace and schema.name in self._schemas:
            raise ValueError(f"Container schema already registered: {schema.name}")
        self._schemas[schema.name] = schema

    def get(self, name: str) -> ContainerSchema:
        try:
            return self._schemas[name]
        except KeyError as exc:
            raise KeyError(f"Unknown container schema: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._schemas)


def default_container_schema_registry() -> ContainerSchemaRegistry:
    return ContainerSchemaRegistry(
        [
            ContainerSchema(
                name="default",
                image="python:3.12-alpine",
                read_only_root=True,
                network=False,
            )
        ]
    )


def load_container_schema_registry(path: str | Path) -> ContainerSchemaRegistry:
    payload = json.loads(Path(path).read_text())
    raw_schemas = payload["schemas"] if isinstance(payload, dict) else payload
    return ContainerSchemaRegistry([ContainerSchema.model_validate(raw) for raw in raw_schemas])


@dataclass(frozen=True)
class _ContainerState:
    session_id: str
    schema_name: str
    key_scope: str
    container_name: str
    remove_volumes: bool = False


class DockerContainerExecutor(ToolExecutor):
    def __init__(
        self,
        docker_bin: str = "docker",
        *,
        schemas: ContainerSchemaRegistry | None = None,
        allowed_mount_root: str | Path | None = None,
        cleanup_timeout_seconds: float = 5,
    ) -> None:
        self.docker_bin = docker_bin
        self.cleanup_timeout_seconds = cleanup_timeout_seconds
        self.schemas = schemas or default_container_schema_registry()
        self.allowed_mount_root = (
            Path(allowed_mount_root).expanduser().resolve()
            if allowed_mount_root is not None
            else None
        )
        self._containers: dict[tuple[str, str, str], _ContainerState] = {}
        self._container_locks: dict[tuple[str, str, str], anyio.Lock] = {}
        self._exec_locks: dict[tuple[str, str, str], anyio.Lock] = {}
        self._cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_activity_lock = anyio.Lock()
        self._session_lifecycle_locks: dict[str, anyio.Lock] = {}
        self._session_activity: dict[str, int] = {}
        self._session_activity_events: dict[str, asyncio.Event] = {}

    async def execute(
        self,
        definition: ToolDefinition,
        call: ToolCall,
        secrets: dict[str, str],
    ) -> ToolResult:
        if not definition.container_command:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                error="Container tool requires a command",
            )
        try:
            schema = self.schemas.get(definition.container_schema)
        except KeyError as exc:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                error=str(exc),
            )
        if secrets and not schema.allow_secrets:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="denied",
                error=f"Container schema {schema.name!r} does not allow secrets",
            )
        if schema.allow_secrets and not definition.required_secrets:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="denied",
                error=f"Container schema {schema.name!r} is reserved for credentialed tools",
            )

        started = utc_now()
        persistent = not schema.allow_secrets or schema.persistent_secrets
        one_shot = not persistent
        key = _container_key(call.session_id, schema, definition)
        state: _ContainerState | None = None
        timed_out = False
        cleanup_failed = False
        activity_open = True
        await self._begin_session_activity(call.session_id)
        try:
            with anyio.fail_after(definition.timeout_seconds):
                exec_lock = self._exec_locks.setdefault(key, anyio.Lock())
                async with exec_lock:
                    state = await self._ensure_container(
                        key,
                        schema,
                        definition,
                        allow_existing=persistent,
                    )
                    completed = await self._exec_tool(definition, call, secrets, state)
        except TimeoutError:
            timed_out = True
            await self._end_session_activity(call.session_id)
            activity_open = False
            await self.cleanup_session(call.session_id)
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="timeout",
                error=f"Tool timed out after {definition.timeout_seconds} seconds",
                started_at=started,
                ended_at=utc_now(),
            )
        except ProcessOutputLimitExceeded as exc:
            if state is not None:
                if await self._cleanup_container(state):
                    await self._forget_container(state)
                else:
                    if one_shot:
                        await self._quarantine_container(state)
                    if schema.allow_secrets:
                        cleanup_failed = True
                state = None
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                output={"stdout": exc.stdout.decode(errors="replace")},
                error=str(exc),
                metadata={
                    "runtime": "docker",
                    "container_schema": schema.name,
                    **({"cleanup_failed": True} if cleanup_failed else {}),
                },
                started_at=started,
                ended_at=utc_now(),
            )
        except Exception as exc:  # noqa: BLE001 - keep executor errors inside tool results.
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                error=f"Container executor failed: {exc}",
                started_at=started,
                ended_at=utc_now(),
            )
        finally:
            if state is not None and not persistent and not timed_out:
                if await self._cleanup_container(state):
                    await self._forget_container(state)
                else:
                    if one_shot:
                        await self._quarantine_container(state)
                    if schema.allow_secrets:
                        cleanup_failed = True
            if activity_open:
                await self._end_session_activity(call.session_id)

        assert state is not None
        stdout_text = completed.stdout.decode(errors="replace")
        stderr_text = completed.stderr.decode(errors="replace")
        metadata: dict[str, Any] = {
            "runtime": "docker",
            "container": state.container_name,
            "container_schema": state.schema_name,
        }
        if stderr_text:
            metadata["stderr"] = stderr_text
        if cleanup_failed:
            metadata["cleanup_failed"] = True
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                output={"stdout": stdout_text},
                error=(
                    "Secret-enabled container cleanup failed; "
                    "credentialed tool result was not accepted"
                ),
                metadata=metadata,
                started_at=started,
                ended_at=utc_now(),
            )
        if completed.returncode != 0:
            if _is_dead_container_error(stderr_text):
                await self._forget_container(state)
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                output={"stdout": stdout_text},
                error=stderr_text or f"Container exec exited with {completed.returncode}",
                metadata=metadata,
                started_at=started,
                ended_at=utc_now(),
            )
        try:
            output = json.loads(stdout_text) if stdout_text.strip() else None
        except json.JSONDecodeError:
            output = {"stdout": stdout_text}
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            status="ok",
            output=output,
            metadata=metadata,
            started_at=started,
            ended_at=utc_now(),
        )

    def validate_secret_access(self, definition: ToolDefinition) -> str | None:
        try:
            schema = self.schemas.get(definition.container_schema)
        except KeyError as exc:
            return str(exc)
        if definition.required_secrets and not schema.allow_secrets:
            return f"Container schema {schema.name!r} does not allow secrets"
        if schema.allow_secrets and not definition.required_secrets:
            return f"Container schema {schema.name!r} is reserved for credentialed tools"
        return None

    async def schedule_session_cleanup(self, session_id: str, delay_minutes: float) -> None:
        existing = self._cleanup_tasks.pop(session_id, None)
        if existing is not None:
            existing.cancel()
            await asyncio.gather(existing, return_exceptions=True)
        if not any(
            _state_belongs_to_session(key, state, session_id)
            for key, state in self._containers.items()
        ):
            self._prune_session_state(session_id)
            return
        self._cleanup_tasks[session_id] = asyncio.create_task(
            self._cleanup_after_delay(session_id, delay_minutes)
        )

    async def cleanup_session(self, session_id: str, *, wait_for_active: bool = True) -> None:
        lock = self._session_lifecycle_locks.setdefault(session_id, anyio.Lock())
        async with lock:
            if wait_for_active:
                await self._wait_for_session_activity_at_most(session_id, 0)
            task = self._cleanup_tasks.pop(session_id, None)
            current_task = asyncio.current_task()
            if task is not None and task is not current_task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            states = [
                state
                for key, state in list(self._containers.items())
                if _state_belongs_to_session(key, state, session_id)
            ]
            for state in states:
                if await self._cleanup_container(state):
                    await self._forget_container(state)
            await self._cleanup_containers_by_labels({LABEL_SESSION: session_id})
            self._prune_session_state(session_id)

    async def cleanup_all(self) -> None:
        tasks = list(self._cleanup_tasks.values())
        for task in tasks:
            task.cancel()
        self._cleanup_tasks.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for state in list(self._containers.values()):
            if await self._cleanup_container(state):
                await self._forget_container(state)

    async def aclose(self) -> None:
        await self.cleanup_all()

    async def _begin_session_activity(self, session_id: str) -> None:
        lock = self._session_lifecycle_locks.setdefault(session_id, anyio.Lock())
        async with lock:
            async with self._session_activity_lock:
                self._session_activity[session_id] = (
                    self._session_activity.get(session_id, 0) + 1
                )
                self._notify_session_activity_locked(session_id)

    async def _end_session_activity(self, session_id: str) -> None:
        async with self._session_activity_lock:
            active = self._session_activity.get(session_id, 0)
            if active <= 1:
                self._session_activity.pop(session_id, None)
            else:
                self._session_activity[session_id] = active - 1
            self._notify_session_activity_locked(session_id)

    async def _wait_for_session_activity_at_most(
        self, session_id: str, max_active: int
    ) -> None:
        while True:
            async with self._session_activity_lock:
                if self._session_activity.get(session_id, 0) <= max_active:
                    return
                event = self._session_activity_events.setdefault(session_id, asyncio.Event())
            await event.wait()

    def _notify_session_activity_locked(self, session_id: str) -> None:
        event = self._session_activity_events.get(session_id)
        if event is not None:
            event.set()
        self._session_activity_events[session_id] = asyncio.Event()

    def _prune_session_state(self, session_id: str) -> None:
        active = self._session_activity.get(session_id, 0)
        has_containers = any(
            _state_belongs_to_session(key, state, session_id)
            for key, state in self._containers.items()
        )
        if active > 0 or has_containers:
            return
        self._session_activity_events.pop(session_id, None)
        for key in list(self._container_locks):
            if key[0] == session_id:
                self._container_locks.pop(key, None)
        for key in list(self._exec_locks):
            if key[0] == session_id:
                self._exec_locks.pop(key, None)
        self._session_lifecycle_locks.pop(session_id, None)

    async def _cleanup_after_delay(self, session_id: str, delay_minutes: float) -> None:
        try:
            await anyio.sleep(max(delay_minutes, 0) * 60)
            await self.cleanup_session(session_id)
        except asyncio.CancelledError:
            return
        finally:
            current_task = asyncio.current_task()
            if self._cleanup_tasks.get(session_id) is current_task:
                self._cleanup_tasks.pop(session_id, None)

    async def _ensure_container(
        self,
        key: tuple[str, str, str],
        schema: ContainerSchema,
        definition: ToolDefinition,
        *,
        allow_existing: bool,
    ) -> _ContainerState:
        session_id, schema_name, key_scope = key
        cleanup_task = self._cleanup_tasks.pop(session_id, None)
        if cleanup_task is not None:
            cleanup_task.cancel()
            await asyncio.gather(cleanup_task, return_exceptions=True)
        existing = self._containers.get(key)
        if existing is not None and allow_existing:
            return existing

        lock = self._container_locks.setdefault(key, anyio.Lock())
        async with lock:
            cleanup_task = self._cleanup_tasks.pop(session_id, None)
            if cleanup_task is not None:
                cleanup_task.cancel()
                await asyncio.gather(cleanup_task, return_exceptions=True)
            existing = self._containers.get(key)
            if existing is not None and allow_existing:
                return existing

            await self._cleanup_containers_by_labels(
                {
                    LABEL_SESSION: session_id,
                    LABEL_SCHEMA: schema_name,
                    LABEL_KEY_SCOPE: key_scope,
                },
                remove_volumes=schema.allow_secrets,
            )
            container_name = (
                f"harness-{_container_name_part(session_id)}-"
                f"{_container_name_part(schema_name)}-"
                f"{_container_name_part(definition.name)}-{uuid4().hex[:12]}"
            )
            if schema.allow_secrets:
                await self._assert_secret_image_has_no_volumes(schema)
            command = self._docker_run_command(
                container_name,
                schema,
                session_id=session_id,
                schema_name=schema_name,
                key_scope=key_scope,
            )
            env = self._docker_env()
            try:
                completed = await anyio.run_process(command, check=False, env=env)
            except BaseException:
                await self._cleanup_container(container_name, remove_volumes=schema.allow_secrets)
                raise
            if completed.returncode != 0:
                stderr = completed.stderr.decode(errors="replace")
                await self._cleanup_container(container_name, remove_volumes=schema.allow_secrets)
                raise RuntimeError(stderr or f"Docker run exited with {completed.returncode}")
            state = _ContainerState(
                session_id=session_id,
                schema_name=schema_name,
                key_scope=key_scope,
                container_name=container_name,
                remove_volumes=schema.allow_secrets,
            )
            self._containers[key] = state
            return state

    async def _exec_tool(
        self,
        definition: ToolDefinition,
        call: ToolCall,
        secrets: dict[str, str],
        state: _ContainerState,
    ):
        payload: dict[str, object] = {"arguments": call.arguments}
        if secrets:
            payload["secrets"] = secrets
        tool_command = definition.container_command
        if tool_command is None:
            raise ValueError("Container tool requires a command")
        return await run_limited_process(
            [self.docker_bin, "exec", "-i", state.container_name, *tool_command],
            input=json.dumps(payload).encode(),
            env=self._docker_env(),
            timeout_seconds=definition.timeout_seconds,
            max_output_bytes=definition.max_output_bytes,
        )

    def _docker_run_command(
        self,
        container_name: str,
        schema: ContainerSchema,
        *,
        session_id: str,
        schema_name: str,
        key_scope: str,
    ) -> list[str]:
        command = [
            self.docker_bin,
            "run",
            "-d",
            "--name",
            container_name,
            "--pids-limit",
            str(schema.pids_limit),
            "--memory",
            schema.memory,
            "--cpus",
            schema.cpus,
            "--network",
            "bridge" if schema.network else "none",
            "-w",
            schema.workdir,
            "--label",
            f"{LABEL_MANAGED}=true",
            "--label",
            f"{LABEL_SESSION}={session_id}",
            "--label",
            f"{LABEL_SCHEMA}={schema_name}",
            "--label",
            f"{LABEL_KEY_SCOPE}={key_scope}",
        ]
        if schema.cap_drop_all:
            command.extend(["--cap-drop", "ALL"])
        if schema.no_new_privileges:
            command.extend(["--security-opt", "no-new-privileges"])
        if schema.read_only_root:
            command.append("--read-only")
        if schema.tmpfs_tmp:
            command.extend(["--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=64m"])
        if schema.tmpfs_workdir and schema.mount is None:
            command.extend(["--tmpfs", f"{schema.workdir}:rw,nosuid,nodev,size=256m"])
        if schema.mount is not None:
            mount = schema.mount.resolve()
            if self.allowed_mount_root is None:
                raise ValueError("Container mounts require an allowed mount root")
            if not mount.is_relative_to(self.allowed_mount_root):
                raise ValueError("Container mount is outside the allowed mount root")
            command.extend(["-v", f"{mount}:{schema.workdir}:rw"])
        command.append(schema.image)
        command.extend(schema.keepalive_command)
        return command

    async def _cleanup_containers_by_labels(
        self,
        labels: dict[str, str],
        *,
        remove_volumes: bool = True,
    ) -> bool:
        command = [self.docker_bin, "ps", "-aq", "--filter", f"label={LABEL_MANAGED}=true"]
        for key, value in labels.items():
            command.extend(["--filter", f"label={key}={value}"])
        try:
            with anyio.move_on_after(self.cleanup_timeout_seconds, shield=True):
                completed = await anyio.run_process(
                    command,
                    check=False,
                    env=self._docker_env(),
                )
                if completed.returncode != 0:
                    return False
                container_names = completed.stdout.decode(errors="replace").split()
                if not container_names:
                    return True
                remove_flag = "-fv" if remove_volumes else "-f"
                removed = await anyio.run_process(
                    [self.docker_bin, "rm", remove_flag, *container_names],
                    check=False,
                    env=self._docker_env(),
                )
                return removed.returncode == 0
        except Exception:  # noqa: BLE001 - cleanup must not mask caller behavior.
            return False
        return False

    async def _cleanup_container(
        self,
        state_or_name: _ContainerState | str,
        *,
        remove_volumes: bool | None = None,
    ) -> bool:
        if isinstance(state_or_name, _ContainerState):
            container_name = state_or_name.container_name
            should_remove_volumes = state_or_name.remove_volumes
        else:
            container_name = state_or_name
            should_remove_volumes = bool(remove_volumes)
        remove_flag = "-fv" if should_remove_volumes else "-f"
        try:
            with anyio.move_on_after(self.cleanup_timeout_seconds, shield=True):
                completed = await anyio.run_process(
                    [self.docker_bin, "rm", remove_flag, container_name],
                    check=False,
                    env=self._docker_env(),
                )
                return completed.returncode == 0
        except Exception:  # noqa: BLE001 - cleanup must not mask tool results.
            return False
        return False

    async def _assert_secret_image_has_no_volumes(self, schema: ContainerSchema) -> None:
        volumes = await self._image_volumes(schema.image)
        if volumes:
            joined = ", ".join(sorted(volumes))
            raise ValueError(
                f"Secret-enabled container image {schema.image!r} declares Docker volumes: "
                f"{joined}"
            )

    async def _image_volumes(self, image: str) -> set[str]:
        inspect = await anyio.run_process(
            [
                self.docker_bin,
                "image",
                "inspect",
                image,
                "--format",
                "{{json .Config.Volumes}}",
            ],
            check=False,
            env=self._docker_env(),
        )
        if inspect.returncode != 0:
            pull = await anyio.run_process(
                [self.docker_bin, "pull", image],
                check=False,
                env=self._docker_env(),
            )
            if pull.returncode != 0:
                stderr = pull.stderr.decode(errors="replace")
                raise RuntimeError(stderr or f"Docker pull exited with {pull.returncode}")
            inspect = await anyio.run_process(
                [
                    self.docker_bin,
                    "image",
                    "inspect",
                    image,
                    "--format",
                    "{{json .Config.Volumes}}",
                ],
                check=False,
                env=self._docker_env(),
            )
        if inspect.returncode != 0:
            stderr = inspect.stderr.decode(errors="replace")
            raise RuntimeError(stderr or f"Docker image inspect exited with {inspect.returncode}")
        raw = inspect.stdout.decode(errors="replace").strip()
        if raw in {"", "null", "{}"}:
            return set()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return set()
        return {str(path) for path in payload}

    def _docker_env(self) -> dict[str, str]:
        env = filtered_env(inherit=DOCKER_ENV_ALLOWLIST)
        docker_parent = str(Path(self.docker_bin).expanduser().parent)
        if "/" in self.docker_bin and docker_parent not in env.get("PATH", "").split(":"):
            env["PATH"] = f"{docker_parent}:{env.get('PATH', '')}"
        return env

    async def _forget_container(self, state: _ContainerState) -> None:
        keys = [
            key
            for key, existing in list(self._containers.items())
            if existing.container_name == state.container_name
        ]
        keys.append((state.session_id, state.schema_name, state.key_scope))
        for key in keys:
            self._containers.pop(key, None)

    async def _quarantine_container(self, state: _ContainerState) -> None:
        await self._forget_container(state)
        self._containers[
            (f"__cleanup__:{state.container_name}", state.schema_name, state.key_scope)
        ] = state


def _is_dead_container_error(stderr_text: str) -> bool:
    lowered = stderr_text.lower()
    return (
        "no such container" in lowered
        or "is not running" in lowered
        or "container is not running" in lowered
    )


def _state_belongs_to_session(
    key: tuple[str, str, str],
    state: _ContainerState,
    session_id: str,
) -> bool:
    return key[0] == session_id or state.session_id == session_id


def _container_name_part(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    if not sanitized or not sanitized[0].isalnum():
        sanitized = "x"
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{sanitized[:24]}-{digest}"


def _container_key(
    session_id: str,
    schema: ContainerSchema,
    definition: ToolDefinition,
) -> tuple[str, str, str]:
    key_scope = (
        "shared" if schema.share_across_tools and not schema.allow_secrets else definition.name
    )
    return (session_id, schema.name, key_scope)
