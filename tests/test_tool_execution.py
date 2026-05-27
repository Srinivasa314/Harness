from __future__ import annotations

import json
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from harness.agent import AgentSessionManager
from harness.config import HarnessSettings
from harness.execution import ContainerSchemaRegistry, DockerContainerExecutor
from harness.execution.base import ToolExecutor
from harness.execution.process import run_limited_process
from harness.execution.subprocess import SubprocessExecutor
from harness.schemas import ContainerSchema, ExecutionMode, ToolCall, ToolDefinition, ToolResult
from harness.storage import FileArtifactStore, SQLiteStorage
from harness.tools import (
    CapabilityGrant,
    CapabilityPolicy,
    EnvSecretResolver,
    SecretResolver,
    ToolExecutionGateway,
)
from harness.tools.registry import ToolRegistry


async def _require_docker_available(docker_bin: str) -> None:
    if shutil.which(docker_bin) is None and not Path(docker_bin).exists():
        pytest.skip("docker is not installed")
    try:
        with anyio.fail_after(10):
            completed = await anyio.run_process(
                [docker_bin, "info", "--format", "{{json .ServerVersion}}"],
                check=False,
            )
    except TimeoutError:
        pytest.skip("docker daemon did not respond")
    except OSError as exc:
        pytest.skip(f"docker is not available: {exc}")
    if completed.returncode != 0:
        stderr = completed.stderr.decode(errors="replace").strip()
        pytest.skip(stderr or "docker daemon is not available")


def test_public_execution_package_imports_in_fresh_process():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from harness.execution import SubprocessExecutor; print(SubprocessExecutor.__name__)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "SubprocessExecutor"


async def echo_tool(arguments: dict, secrets: dict[str, str]) -> dict:
    return {"arguments": arguments, "secret_seen": bool(secrets)}


@pytest.fixture
async def storage(tmp_path):
    storage = SQLiteStorage(tmp_path / "harness.sqlite3")
    await storage.migrate()
    return storage


@pytest.mark.anyio
async def test_in_process_tool_records_trace(storage):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo arguments",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            required_capabilities=["tool:echo"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        echo_tool,
    )
    session = await AgentSessionManager(storage).create()
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:echo"}))),
        storage=storage,
    )

    result = await gateway.execute(
        ToolCall(session_id=session.id, name="echo", arguments={"message": "hello"})
    )

    assert result.status == "ok"
    assert result.output == {"arguments": {"message": "hello"}, "secret_seen": False}
    calls = await storage.list_tool_calls(session_id=session.id)
    assert calls[0]["tool_name"] == "echo"


@pytest.mark.anyio
async def test_session_turns_are_persisted(storage):
    manager = AgentSessionManager(storage)
    session = await manager.create()

    await manager.add_turn(session.id, "user", "hello")
    await manager.add_turn(session.id, "assistant", "hi")

    turns = await storage.list_turns(session.id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "hello"


@pytest.mark.anyio
async def test_session_manager_redacts_turn_content_and_metadata(storage):
    manager = AgentSessionManager(storage)
    session = await manager.create()

    await manager.add_turn(
        session.id,
        "user",
        "api_key is sk-session-secret",
        metadata={"api_key": "sk-session-metadata"},
    )

    [turn] = await storage.list_turns(session.id)
    assert "sk-session-secret" not in turn.content
    assert turn.metadata["api_key"] == "[REDACTED]"


@pytest.mark.anyio
async def test_session_manager_redacts_session_metadata(storage):
    manager = AgentSessionManager(storage)

    session = await manager.create(metadata={"api_key": "sk-session-metadata"})

    assert session.metadata["api_key"] == "[REDACTED]"
    stored = await storage.get_session(session.id)
    assert stored is not None
    assert stored.metadata["api_key"] == "[REDACTED]"


@pytest.mark.anyio
async def test_missing_capability_denies_tool(storage):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo arguments",
            required_capabilities=["tool:echo"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        echo_tool,
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset())),
        storage=storage,
    )

    result = await gateway.execute(ToolCall(session_id="session-1", name="echo", arguments={}))

    assert result.status == "denied"
    assert "capability" in (result.error or "")


@pytest.mark.anyio
async def test_unknown_tool_denial_is_persisted_and_observed(storage):
    session = await AgentSessionManager(storage).create()
    gateway = ToolExecutionGateway(
        registry=ToolRegistry(),
        policy=CapabilityPolicy(CapabilityGrant.all()),
        storage=storage,
    )

    result = await gateway.execute(
        ToolCall(
            session_id=session.id,
            name="missing",
            arguments={"api_key": "sk-secret"},
        )
    )

    assert result.status == "denied"
    [stored] = await storage.list_tool_calls(session_id=session.id)
    assert stored["input"]["api_key"] == "[REDACTED]"
    events = await storage.list_events(session_id=session.id)
    assert events[0].event_type == "tool.call.denied"


@pytest.mark.anyio
async def test_invalid_tool_input_denial_is_persisted_and_observed(storage):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="typed",
            description="Requires a string",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            required_capabilities=["tool:typed"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        echo_tool,
    )
    session = await AgentSessionManager(storage).create()
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:typed"}))),
        storage=storage,
    )

    result = await gateway.execute(
        ToolCall(session_id=session.id, name="typed", arguments={"message": 3})
    )

    assert result.status == "denied"
    assert "Invalid tool input" in (result.error or "")
    events = await storage.list_events(session_id=session.id)
    assert any(event.event_type == "tool.call.denied" for event in events)
    assert any(
        event.event_type == "tool.call.finished" and event.payload["status"] == "denied"
        for event in events
    )


@pytest.mark.anyio
async def test_invalid_tool_input_error_is_redacted(storage):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="typed",
            description="Rejects unknown API keys",
            input_schema={
                "type": "object",
                "properties": {"api_key": {"type": "string", "enum": ["allowed"]}},
                "required": ["api_key"],
            },
            required_capabilities=["tool:typed"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        echo_tool,
    )
    session = await AgentSessionManager(storage).create()
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:typed"}))),
        storage=storage,
    )

    result = await gateway.execute(
        ToolCall(
            session_id=session.id,
            name="typed",
            arguments={"api_key": "sk-live-secret", "message": "sk-live-secret"},
        )
    )

    assert result.status == "denied"
    assert "sk-live-secret" not in (result.error or "")
    [stored] = await storage.list_tool_calls(session_id=session.id)
    assert "sk-live-secret" not in str(stored.model_dump(mode="json"))
    events = await storage.list_events(session_id=session.id)
    assert "sk-live-secret" not in str([event.model_dump(mode="json") for event in events])


@pytest.mark.anyio
async def test_secret_is_injected_and_redacted(storage, monkeypatch):
    monkeypatch.setenv("HARNESS_SECRET_API_TOKEN", "super-secret-value")
    registry = ToolRegistry()

    async def credentialed_tool(_arguments: dict, secrets: dict[str, str]) -> dict:
        return {"message": f"token={secrets['api_token']}"}

    registry.register(
        ToolDefinition(
            name="credentialed",
            description="Uses a secret",
            required_capabilities=["secret:use"],
            required_secrets=["api_token"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        credentialed_tool,
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"secret:use"}))),
        storage=storage,
        secret_resolver=EnvSecretResolver(),
    )

    result = await gateway.execute(
        ToolCall(session_id="session-1", name="credentialed", arguments={})
    )

    assert result.status == "ok"
    assert result.output == {"message": "token=[REDACTED]"}


@pytest.mark.anyio
async def test_generated_tool_output_secrets_are_redacted(storage):
    registry = ToolRegistry()

    async def generated_secret_tool(_arguments: dict, _secrets: dict[str, str]) -> dict:
        return {
            "clientSecret": "generated-secret",
            "message": "generated-secret",
        }

    registry.register(
        ToolDefinition(
            name="generated-secret",
            description="Generates a secret",
            required_capabilities=["secret:generate"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        generated_secret_tool,
    )
    session = await AgentSessionManager(storage).create()
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"secret:generate"}))),
        storage=storage,
    )

    result = await gateway.execute(ToolCall(session_id=session.id, name="generated-secret"))

    assert result.status == "ok"
    assert "generated-secret" not in str(result.output)
    [stored] = await storage.list_tool_calls(session_id=session.id)
    assert "generated-secret" not in str(stored.output)


@pytest.mark.anyio
async def test_missing_secret_resolver_denies_tool(storage):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="needs-secret",
            description="Requires a secret resolver",
            required_capabilities=["secret:use"],
            required_secrets=["api_token"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        echo_tool,
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"secret:use"}))),
        storage=storage,
    )

    result = await gateway.execute(ToolCall(session_id="session-1", name="needs-secret"))

    assert result.status == "denied"
    assert "No secret resolver" in (result.error or "")


@pytest.mark.anyio
async def test_missing_env_secret_denies_tool(storage, monkeypatch):
    monkeypatch.delenv("HARNESS_SECRET_API_TOKEN", raising=False)
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="needs-env-secret",
            description="Requires an env secret",
            required_capabilities=["secret:use"],
            required_secrets=["api_token"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        echo_tool,
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"secret:use"}))),
        storage=storage,
        secret_resolver=EnvSecretResolver(),
    )

    result = await gateway.execute(ToolCall(session_id="session-1", name="needs-env-secret"))

    assert result.status == "denied"
    assert "Missing secret" in (result.error or "")


@pytest.mark.anyio
async def test_tool_inputs_are_redacted_before_persistence(storage):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo arguments",
            required_capabilities=["tool:echo"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        echo_tool,
    )
    session = await AgentSessionManager(storage).create()
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:echo"}))),
        storage=storage,
    )

    await gateway.execute(
        ToolCall(
            session_id=session.id,
            name="echo",
            arguments={"api_key": "sk-secret", "message": "hello"},
        )
    )

    [stored] = await storage.list_tool_calls(session_id=session.id)
    assert stored["input"]["api_key"] == "[REDACTED]"
    assert "sk-secret" not in str(stored["input"])


@pytest.mark.anyio
async def test_common_secret_key_aliases_are_redacted(storage):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo arguments",
            required_capabilities=["tool:echo"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        echo_tool,
    )
    session = await AgentSessionManager(storage).create()
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:echo"}))),
        storage=storage,
    )

    await gateway.execute(
        ToolCall(
            session_id=session.id,
            name="echo",
            arguments={
                "x-api-key": "secret-1",
                "openai_api_key": "secret-2",
                "clientSecret": "plain-client-secret",
                "accessToken": "plain-access-token",
                "privateKey": "plain-private-key",
                "secret_seen": True,
            },
        )
    )

    [stored] = await storage.list_tool_calls(session_id=session.id)
    assert stored["input"]["x-api-key"] == "[REDACTED]"
    assert stored["input"]["openai_api_key"] == "[REDACTED]"
    assert stored["input"]["clientSecret"] == "[REDACTED]"
    assert stored["input"]["accessToken"] == "[REDACTED]"
    assert stored["input"]["privateKey"] == "[REDACTED]"
    assert stored["input"]["secret_seen"] is True


@pytest.mark.anyio
async def test_sensitive_tool_arguments_are_redacted_from_outputs(storage):
    registry = ToolRegistry()

    async def leaks_argument(arguments: dict, _secrets: dict[str, str]) -> dict:
        return {"message": arguments["clientSecret"]}

    registry.register(
        ToolDefinition(
            name="leaky",
            description="Echo a sensitive argument",
            required_capabilities=["tool:leaky"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        leaks_argument,
    )
    session = await AgentSessionManager(storage).create()
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:leaky"}))),
        storage=storage,
    )

    result = await gateway.execute(
        ToolCall(
            session_id=session.id,
            name="leaky",
            arguments={"clientSecret": "plain-client-secret"},
        )
    )

    assert result.output == {"message": "[REDACTED]"}
    [stored] = await storage.list_tool_calls(session_id=session.id)
    dumped = stored.model_dump(mode="json")
    assert "plain-client-secret" not in str(dumped)
    assert dumped["output"] == {"message": "[REDACTED]"}


@pytest.mark.anyio
async def test_storage_redacts_direct_tool_call_records(storage):
    call = ToolCall(
        session_id="session-1",
        name="direct",
        arguments={"clientSecret": "plain-client-secret", "message": "hello"},
    )
    result = ToolResult(
        call_id=call.call_id,
        name=call.name,
        status="error",
        output={"accessToken": "plain-access-token"},
        artifacts=["/tmp/sk-direct-artifact-secret.txt"],
        error="failed with sk-direct-error-secret",
    )

    await storage.record_tool_call(call, result)

    [stored] = await storage.list_tool_calls()
    dumped = stored.model_dump(mode="json")
    assert "plain-client-secret" not in str(dumped)
    assert "plain-access-token" not in str(dumped)
    assert "sk-direct-error-secret" not in str(dumped)
    assert "sk-direct-artifact-secret" not in str(dumped)
    assert dumped["input"]["clientSecret"] == "[REDACTED]"
    assert dumped["output"]["accessToken"] == "[REDACTED]"


@pytest.mark.anyio
async def test_gateway_does_not_stat_tool_returned_artifact_paths(storage, monkeypatch):
    registry = ToolRegistry()

    class ArtifactExecutor(ToolExecutor):
        async def execute(
            self,
            definition: ToolDefinition,
            call: ToolCall,
            secrets: dict[str, str],
        ) -> ToolResult:
            _ = definition, secrets
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="ok",
                artifacts=["/host/path/that/should/not/be/statted"],
            )

    registry.register(
        ToolDefinition(
            name="artifact-tool",
            description="Returns an artifact path",
            required_capabilities=["tool:artifact"],
            execution_mode=ExecutionMode.SUBPROCESS,
        )
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:artifact"}))),
        storage=storage,
        executors={ExecutionMode.SUBPROCESS: ArtifactExecutor()},
    )

    def fail_stat(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("host stat should not be called for tool-returned artifacts")

    monkeypatch.setattr(Path, "stat", fail_stat)

    result = await gateway.execute(ToolCall(session_id="session-1", name="artifact-tool"))

    assert result.status == "ok"
    [artifact] = await storage.list_artifacts()
    assert artifact.path == "/host/path/that/should/not/be/statted"
    assert artifact.size_bytes is None


@pytest.mark.anyio
async def test_denied_tool_inputs_are_redacted_before_persistence(storage):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo arguments",
            required_capabilities=["tool:echo"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        echo_tool,
    )
    session = await AgentSessionManager(storage).create()
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset())),
        storage=storage,
    )

    result = await gateway.execute(
        ToolCall(
            session_id=session.id,
            name="echo",
            arguments={"openai_api_key": "sk-secret", "message": "hello"},
        )
    )

    assert result.status == "denied"
    [stored] = await storage.list_tool_calls(session_id=session.id)
    assert stored["input"]["openai_api_key"] == "[REDACTED]"
    assert "sk-secret" not in str(stored)


@pytest.mark.anyio
async def test_tool_output_schema_is_enforced(storage):
    registry = ToolRegistry()

    async def bad_output_tool(_arguments: dict, _secrets: dict[str, str]) -> dict:
        return {"value": 123}

    registry.register(
        ToolDefinition(
            name="bad-output",
            description="Returns invalid output",
            output_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            required_capabilities=["tool:bad-output"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        bad_output_tool,
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:bad-output"}))),
        storage=storage,
    )

    result = await gateway.execute(ToolCall(session_id="session-1", name="bad-output"))

    assert result.status == "error"
    assert "Invalid tool output" in (result.error or "")


class StaticSecretResolver(SecretResolver):
    async def resolve(self, names: list[str]) -> dict[str, str]:
        return {name: f"static-{name}" for name in names}


class FailingSecretResolver(SecretResolver):
    async def resolve(self, names: list[str]) -> dict[str, str]:
        _ = names
        raise AssertionError("secret resolver should not have been called")


def _schemas(*schemas: ContainerSchema) -> ContainerSchemaRegistry:
    return ContainerSchemaRegistry(list(schemas))


def _schema(**overrides) -> ContainerSchema:  # noqa: ANN003
    values = {
        "name": "default",
        "image": "python:3.12-alpine",
    }
    values.update(overrides)
    return ContainerSchema.model_validate(values)


@pytest.mark.anyio
async def test_custom_secret_resolver_can_be_injected(storage):
    registry = ToolRegistry()

    async def custom_secret_tool(_arguments: dict, secrets: dict[str, str]) -> dict:
        return {"message": f"token={secrets['api_token']}"}

    registry.register(
        ToolDefinition(
            name="custom-secret",
            description="Uses a pluggable secret resolver",
            required_capabilities=["secret:use"],
            required_secrets=["api_token"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        custom_secret_tool,
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"secret:use"}))),
        storage=storage,
        secret_resolver=StaticSecretResolver(),
    )

    result = await gateway.execute(
        ToolCall(session_id="session-1", name="custom-secret", arguments={})
    )

    assert result.status == "ok"
    assert result.output == {"message": "token=[REDACTED]"}


@pytest.mark.anyio
async def test_container_tool_requires_secret_enabled_schema(storage):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="container-secret",
            description="Invalid credentialed container tool",
            required_capabilities=["secret:use"],
            required_secrets=["api_token"],
            execution_mode=ExecutionMode.CONTAINER,
            container_command=["python", "-c", "print('{}')"],
        )
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"secret:use"}))),
        storage=storage,
        secret_resolver=FailingSecretResolver(),
        executors={
            ExecutionMode.CONTAINER: DockerContainerExecutor(
                docker_bin="docker",
                schemas=_schemas(_schema(allow_secrets=False)),
            )
        },
    )

    result = await gateway.execute(
        ToolCall(session_id="session-1", name="container-secret", arguments={})
    )

    assert result.status == "denied"
    assert "does not allow secrets" in (result.error or "")


@pytest.mark.anyio
async def test_gateway_closes_owned_default_executors(storage):
    gateway = ToolExecutionGateway(
        registry=ToolRegistry(),
        policy=CapabilityPolicy(CapabilityGrant(frozenset())),
        storage=storage,
    )
    closed = False

    async def fake_close() -> None:
        nonlocal closed
        closed = True

    cast(Any, gateway.executors[ExecutionMode.CONTAINER]).aclose = fake_close

    await gateway.aclose()

    assert closed


@pytest.mark.anyio
async def test_gateway_does_not_close_injected_executors_by_default(storage):
    class ClosableExecutor(ToolExecutor):
        def __init__(self) -> None:
            self.closed = False

        async def execute(
            self,
            definition: ToolDefinition,
            call: ToolCall,
            secrets: dict[str, str],
        ) -> ToolResult:
            _ = definition, call, secrets
            return ToolResult(call_id=call.call_id, name=call.name, status="ok")

        async def aclose(self) -> None:
            self.closed = True

    executor = ClosableExecutor()
    gateway = ToolExecutionGateway(
        registry=ToolRegistry(),
        policy=CapabilityPolicy(CapabilityGrant(frozenset())),
        storage=storage,
        executors={ExecutionMode.SUBPROCESS: executor},
    )

    await gateway.aclose()

    assert not executor.closed


@pytest.mark.anyio
async def test_subprocess_json_protocol(storage, tmp_path):
    tool_script = tmp_path / "tool.py"
    tool_script.write_text(
        """
import json
import sys

payload = json.load(sys.stdin)
print(json.dumps({"echo": payload["arguments"]["message"]}))
""".strip()
    )
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="subprocess-echo",
            description="Echo via subprocess",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            required_capabilities=["tool:echo"],
            execution_mode=ExecutionMode.SUBPROCESS,
            subprocess_command=[sys.executable, str(tool_script)],
        )
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:echo"}))),
        storage=storage,
    )

    result = await gateway.execute(
        ToolCall(session_id="session-1", name="subprocess-echo", arguments={"message": "hello"})
    )

    assert result.status == "ok"
    assert result.output == {"echo": "hello"}


@pytest.mark.anyio
async def test_executor_launch_failure_is_persisted(storage):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="missing-command",
            description="Missing subprocess command",
            required_capabilities=["tool:missing"],
            execution_mode=ExecutionMode.SUBPROCESS,
            subprocess_command=["/definitely/not/here"],
        )
    )
    session = await AgentSessionManager(storage).create()
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:missing"}))),
        storage=storage,
    )

    result = await gateway.execute(ToolCall(session_id=session.id, name="missing-command"))

    assert result.status == "error"
    assert "Subprocess executor failed" in (result.error or "")
    calls = await storage.list_tool_calls(session_id=session.id)
    assert calls[0]["status"] == "error"


@pytest.mark.anyio
async def test_missing_executor_mapping_returns_tool_error(storage):
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="container-tool",
            description="No container executor configured",
            required_capabilities=["tool:container"],
            execution_mode=ExecutionMode.CONTAINER,
            container_command=["python", "-c", "print('{}')"],
        )
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:container"}))),
        storage=storage,
        executors={},
    )

    result = await gateway.execute(ToolCall(session_id="session-1", name="container-tool"))

    assert result.status == "error"
    assert "No executor configured" in (result.error or "")


@pytest.mark.anyio
async def test_subprocess_executor_requires_command():
    executor = SubprocessExecutor()
    definition = ToolDefinition(name="no-command", description="No command")

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="no-command"), {}
    )

    assert result.status == "error"
    assert "no command" in (result.error or "").lower()


@pytest.mark.anyio
async def test_subprocess_executor_normalizes_launch_failure():
    executor = SubprocessExecutor()
    definition = ToolDefinition(
        name="missing-command",
        description="Missing command",
        execution_mode=ExecutionMode.SUBPROCESS,
        subprocess_command=["/definitely/not/here"],
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="missing-command"), {}
    )

    assert result.status == "error"
    assert "Subprocess executor failed" in (result.error or "")


@pytest.mark.anyio
async def test_subprocess_executor_handles_nonzero_and_stderr(tmp_path):
    script = tmp_path / "bad_tool.py"
    script.write_text("import sys; print('partial'); print('bad stderr', file=sys.stderr); exit(7)")
    executor = SubprocessExecutor()
    definition = ToolDefinition(
        name="bad-subprocess",
        description="Fails",
        execution_mode=ExecutionMode.SUBPROCESS,
        subprocess_command=[sys.executable, str(script)],
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="bad-subprocess"), {}
    )

    assert result.status == "error"
    assert result.output == {"stdout": "partial\n"}
    assert "bad stderr" in (result.error or "")


@pytest.mark.anyio
async def test_subprocess_executor_handles_non_json_stdout(tmp_path):
    script = tmp_path / "text_tool.py"
    script.write_text("print('plain text')")
    executor = SubprocessExecutor()
    definition = ToolDefinition(
        name="text-subprocess",
        description="Text stdout",
        execution_mode=ExecutionMode.SUBPROCESS,
        subprocess_command=[sys.executable, str(script)],
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="text-subprocess"), {}
    )

    assert result.status == "ok"
    assert result.output == {"stdout": "plain text\n"}


@pytest.mark.anyio
async def test_subprocess_executor_bounds_captured_output(tmp_path):
    script = tmp_path / "noisy_tool.py"
    script.write_text("print('x' * 1024)")
    executor = SubprocessExecutor()
    definition = ToolDefinition(
        name="noisy-subprocess",
        description="Noisy stdout",
        execution_mode=ExecutionMode.SUBPROCESS,
        subprocess_command=[sys.executable, str(script)],
        max_output_bytes=64,
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="noisy-subprocess"), {}
    )

    assert result.status == "error"
    assert "output exceeded 64 bytes" in (result.error or "")
    assert result.output == {"stdout": "x" * 64}


@pytest.mark.anyio
async def test_subprocess_executor_bounds_combined_output(tmp_path):
    script = tmp_path / "noisy_tool.py"
    script.write_text("import sys; print('x' * 40); print('y' * 40, file=sys.stderr)")
    executor = SubprocessExecutor()
    definition = ToolDefinition(
        name="noisy-subprocess",
        description="Noisy stdout and stderr",
        execution_mode=ExecutionMode.SUBPROCESS,
        subprocess_command=[sys.executable, str(script)],
        max_output_bytes=64,
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="noisy-subprocess"), {}
    )

    assert result.status == "error"
    assert "output exceeded 64 bytes" in (result.error or "")


@pytest.mark.anyio
async def test_limited_process_timeout_kills_child_process_group(tmp_path):
    marker = tmp_path / "child-survived.txt"
    script = tmp_path / "spawn_child.py"
    script.write_text(
        """
import subprocess
import sys
import time

marker = sys.argv[1]
subprocess.Popen([
    sys.executable,
    "-c",
    "import pathlib, sys, time; time.sleep(0.8); pathlib.Path(sys.argv[1]).write_text('alive')",
    marker,
])
time.sleep(10)
""".strip()
    )

    with pytest.raises(TimeoutError):
        await run_limited_process(
            [sys.executable, str(script), str(marker)],
            timeout_seconds=0.2,
            max_output_bytes=1024,
        )
    await anyio.sleep(1)

    assert not marker.exists()


@pytest.mark.anyio
async def test_subprocess_executor_does_not_inherit_parent_secrets(tmp_path, monkeypatch):
    script = tmp_path / "env_tool.py"
    script.write_text(
        """
import json
import os

print(json.dumps({
    "harness_secret": os.environ.get("HARNESS_SECRET_API_TOKEN"),
    "openai_key": os.environ.get("HARNESS_OPENAI_API_KEY"),
}))
""".strip()
    )
    monkeypatch.setenv("HARNESS_SECRET_API_TOKEN", "super-secret-value")
    monkeypatch.setenv("HARNESS_OPENAI_API_KEY", "sk-parent-provider-secret")
    executor = SubprocessExecutor()
    definition = ToolDefinition(
        name="env-subprocess",
        description="Inspect env",
        execution_mode=ExecutionMode.SUBPROCESS,
        subprocess_command=[sys.executable, str(script)],
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="env-subprocess"), {}
    )

    assert result.status == "ok"
    assert result.output == {"harness_secret": None, "openai_key": None}


@pytest.mark.anyio
async def test_concurrent_tool_calls(storage):
    registry = ToolRegistry()

    async def slow_tool(arguments: dict, _secrets: dict[str, str]) -> dict:
        import anyio

        await anyio.sleep(0.01)
        return {"value": arguments["value"]}

    registry.register(
        ToolDefinition(
            name="slow",
            description="Slow echo",
            required_capabilities=["tool:slow"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        slow_tool,
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:slow"}))),
        storage=storage,
    )

    results = await gateway.execute_many(
        [
            ToolCall(session_id="session-1", name="slow", arguments={"value": value})
            for value in range(5)
        ],
        max_concurrency=2,
    )

    assert sorted(result.output["value"] for result in results) == [0, 1, 2, 3, 4]


@pytest.mark.anyio
async def test_execute_many_respects_max_concurrency(storage):
    registry = ToolRegistry()
    active = 0
    max_active = 0

    async def counted_tool(arguments: dict, _secrets: dict[str, str]) -> dict:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            import anyio

            await anyio.sleep(0.02)
            return {"value": arguments["value"]}
        finally:
            active -= 1

    registry.register(
        ToolDefinition(
            name="counted",
            description="Count active calls",
            required_capabilities=["tool:counted"],
            execution_mode=ExecutionMode.IN_PROCESS,
        ),
        counted_tool,
    )
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:counted"}))),
        storage=storage,
    )

    results = await gateway.execute_many(
        [
            ToolCall(session_id="session-1", name="counted", arguments={"value": value})
            for value in range(6)
        ],
        max_concurrency=2,
    )

    assert [result.output["value"] for result in results] == list(range(6))
    assert max_active == 2


@pytest.mark.anyio
async def test_execute_many_rejects_invalid_concurrency(storage):
    gateway = ToolExecutionGateway(
        registry=ToolRegistry(),
        policy=CapabilityPolicy(CapabilityGrant.all()),
        storage=storage,
    )

    with pytest.raises(ValueError, match="max_concurrency"):
        await gateway.execute_many([], max_concurrency=0)


@pytest.mark.anyio
async def test_sync_in_process_tool_registration_is_rejected():
    registry = ToolRegistry()

    def blocking_tool(_arguments: dict, _secrets: dict[str, str]) -> dict:
        return {"done": True}

    with pytest.raises(ValueError, match="async Python functions"):
        registry.register(
            ToolDefinition(
                name="blocking",
                description="Blocking sync tool",
                required_capabilities=["tool:blocking"],
                execution_mode=ExecutionMode.IN_PROCESS,
            ),
            blocking_tool,
        )


@pytest.mark.anyio
async def test_artifact_store_persists_file_and_metadata(storage, tmp_path):
    artifact_store = FileArtifactStore(tmp_path / "artifacts", storage)

    artifact = await artifact_store.save_bytes(
        b"hello",
        filename="output.txt",
        session_id="session-1",
        tool_call_id="call-1",
        media_type="text/plain",
    )

    saved = await storage.list_artifacts(session_id="session-1", tool_call_id="call-1")
    assert saved[0].id == artifact.id
    assert saved[0].size_bytes == 5
    assert stat.S_IMODE(Path(artifact.path).stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(artifact.path).parent.stat().st_mode) == 0o700


@pytest.mark.anyio
async def test_sqlite_storage_creates_private_database_file(tmp_path):
    storage = SQLiteStorage(tmp_path / "private" / "harness.sqlite3")

    await storage.migrate()

    assert stat.S_IMODE(storage.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(storage.path.parent.stat().st_mode) == 0o700


@pytest.mark.anyio
async def test_artifact_store_redacts_metadata(storage, tmp_path):
    artifact_store = FileArtifactStore(tmp_path / "artifacts", storage)

    await artifact_store.save_bytes(
        b"hello",
        filename="output.txt",
        session_id="session-1",
        tool_call_id="call-1",
        metadata={"api_key": "sk-artifact-secret"},
    )

    [saved] = await storage.list_artifacts(session_id="session-1", tool_call_id="call-1")
    assert saved.metadata["api_key"] == "[REDACTED]"
    assert "sk-artifact-secret" not in str(saved.model_dump(mode="json"))


@pytest.mark.anyio
async def test_artifact_store_does_not_overwrite_duplicate_filenames(storage, tmp_path):
    artifact_store = FileArtifactStore(tmp_path / "artifacts", storage)

    first = await artifact_store.save_bytes(
        b"first",
        filename="output.txt",
        session_id="session-1",
        tool_call_id="call-1",
    )
    second = await artifact_store.save_bytes(
        b"second",
        filename="output.txt",
        session_id="session-1",
        tool_call_id="call-1",
    )

    assert first.path != second.path
    assert Path(first.path).read_bytes() == b"first"
    assert Path(second.path).read_bytes() == b"second"


@pytest.mark.anyio
async def test_artifact_store_rejects_path_traversal(storage, tmp_path):
    artifact_store = FileArtifactStore(tmp_path / "artifacts", storage)

    with pytest.raises(ValueError, match="Unsafe artifact path component"):
        await artifact_store.save_bytes(
            b"hello",
            filename="../../outside.txt",
            session_id="session-1",
            tool_call_id="call-1",
        )

    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.anyio
async def test_docker_executor_uses_hardened_defaults(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []
    inputs: list[bytes | None] = []

    class Completed:
        def __init__(self, stdout: bytes = b"") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        commands.append(command)
        inputs.append(kwargs.get("input"))
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="container"), {}
    )

    assert result.status == "ok"
    run_command = next(command for command in commands if command[1] == "run")
    exec_command = next(command for command in commands if command[1] == "exec")
    assert run_command[0:3] == ["docker", "run", "-d"]
    assert "--cap-drop" in run_command
    assert "--security-opt" in run_command
    assert "no-new-privileges" in run_command
    assert "--name" in run_command
    assert run_command[run_command.index("--name") + 1].startswith("harness-")
    assert "--read-only" in run_command
    assert "--tmpfs" in run_command
    assert "/work:rw,nosuid,nodev,size=256m" in run_command
    assert run_command[run_command.index("--network") + 1] == "none"
    assert exec_command[0:3] == ["docker", "exec", "-i"]
    stdin = cast(bytes, inputs[commands.index(exec_command)])
    assert b"secrets" not in stdin


@pytest.mark.anyio
async def test_docker_executor_does_not_pass_parent_secrets_to_docker_cli(monkeypatch):
    from harness.execution import container as container_module

    captured: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = b'{"ok": true}'
        stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = command
        captured["env"] = kwargs["env"]
        return Completed()

    monkeypatch.setenv("HARNESS_SECRET_API_TOKEN", "super-secret-value")
    monkeypatch.setenv("HARNESS_OPENAI_API_KEY", "sk-parent-provider-secret")
    monkeypatch.setenv("DOCKER_CONTEXT", "desktop-linux")
    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="container"), {}
    )

    assert result.status == "ok"
    env = cast(dict[str, str], captured["env"])
    assert "HARNESS_SECRET_API_TOKEN" not in env
    assert "HARNESS_OPENAI_API_KEY" not in env
    assert env["DOCKER_CONTEXT"] == "desktop-linux"


@pytest.mark.anyio
async def test_docker_executor_reuses_container_for_session_and_schema(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: bytes = b"") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    first = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="container"),
        {},
    )
    second = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="container"),
        {},
    )

    assert first.status == "ok"
    assert second.status == "ok"
    assert [command[1] for command in commands if command[1] != "ps"] == [
        "run",
        "exec",
        "exec",
    ]
    exec_commands = [command for command in commands if command[1] == "exec"]
    assert exec_commands[0][3] == exec_commands[1][3]


@pytest.mark.anyio
async def test_docker_executor_isolates_non_secret_tools_by_default(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: bytes = b"") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")

    for tool_name in ("first", "second"):
        await executor.execute(
            ToolDefinition(
                name=tool_name,
                description=tool_name,
                execution_mode=ExecutionMode.CONTAINER,
                container_command=["python", "-c", "print('{}')"],
            ),
            ToolCall(session_id="session-1", name=tool_name),
            {},
        )

    run_commands = [command for command in commands if command[1] == "run"]
    exec_commands = [command for command in commands if command[1] == "exec"]
    assert len(run_commands) == 2
    assert exec_commands[0][3] != exec_commands[1][3]


@pytest.mark.anyio
async def test_docker_executor_serializes_concurrent_container_startup(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: bytes = b"") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "run":
            await container_module.anyio.sleep(0.01)
            return Completed()
        return Completed(b'{"ok": true}')

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )
    results: list[ToolResult] = []

    async def execute_once() -> None:
        results.append(
            await executor.execute(
                definition,
                ToolCall(session_id="session-1", name="container"),
                {},
            )
        )

    async with container_module.anyio.create_task_group() as task_group:
        task_group.start_soon(execute_once)
        task_group.start_soon(execute_once)

    assert [result.status for result in results] == ["ok", "ok"]
    run_commands = [command for command in commands if command[1] == "run"]
    exec_commands = [command for command in commands if command[1] == "exec"]
    assert len(run_commands) == 1
    assert len(exec_commands) == 2
    assert exec_commands[0][3] == exec_commands[1][3]


@pytest.mark.anyio
async def test_docker_executor_evicts_dead_cached_container(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []
    exec_count = 0

    class Completed:
        def __init__(
            self,
            stdout: bytes = b"",
            stderr: bytes = b"",
            returncode: int = 0,
        ) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        nonlocal exec_count
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            exec_count += 1
            if exec_count == 1:
                return Completed(stderr=b"Error: No such container", returncode=1)
            return Completed(stdout=b'{"ok": true}')
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    first = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="container"),
        {},
    )
    second = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="container"),
        {},
    )

    assert first.status == "error"
    assert second.status == "ok"
    assert [command[1] for command in commands if command[1] != "ps"] == [
        "run",
        "exec",
        "run",
        "exec",
    ]


@pytest.mark.anyio
async def test_docker_executor_uses_separate_containers_for_schemas(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: bytes = b"") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(
            _schema(name="python", image="python:3.12-alpine"),
            _schema(name="node", image="node:22-alpine"),
        ),
    )

    await executor.execute(
        ToolDefinition(
            name="python-tool",
            description="Python",
            execution_mode=ExecutionMode.CONTAINER,
            container_schema="python",
            container_command=["python", "-c", "print('{}')"],
        ),
        ToolCall(session_id="session-1", name="python-tool"),
        {},
    )
    await executor.execute(
        ToolDefinition(
            name="node-tool",
            description="Node",
            execution_mode=ExecutionMode.CONTAINER,
            container_schema="node",
            container_command=["node", "-e", "console.log('{}')"],
        ),
        ToolCall(session_id="session-1", name="node-tool"),
        {},
    )

    run_commands = [command for command in commands if command[1] == "run"]
    assert len(run_commands) == 2
    assert (
        run_commands[0][run_commands[0].index("--name") + 1]
        != run_commands[1][run_commands[1].index("--name") + 1]
    )


@pytest.mark.anyio
async def test_docker_executor_passes_secrets_only_when_schema_allows(monkeypatch):
    from harness.execution import container as container_module

    inputs: list[bytes | None] = []

    class Completed:
        def __init__(self, stdout: bytes = b"") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = command
        inputs.append(kwargs.get("input"))
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(_schema(allow_secrets=True, read_only_root=True)),
    )
    definition = ToolDefinition(
        name="container",
        description="Container",
        required_secrets=["api_token"],
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="container", arguments={"message": "hi"}),
        {"api_token": "secret-value"},
    )

    assert result.status == "ok"
    payload = json.loads(next(cast(bytes, value).decode() for value in inputs if value))
    assert payload == {
        "arguments": {"message": "hi"},
        "secrets": {"api_token": "secret-value"},
    }


@pytest.mark.anyio
async def test_docker_executor_reserves_secret_schemas_for_credentialed_tools(monkeypatch):
    from harness.execution import container as container_module

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = command, kwargs
        raise AssertionError("docker should not be called")

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(_schema(allow_secrets=True, read_only_root=True)),
    )
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="container"),
        {},
    )

    assert result.status == "denied"
    assert "reserved for credentialed tools" in (result.error or "")


@pytest.mark.anyio
async def test_docker_executor_isolates_secret_enabled_containers_by_tool(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: bytes = b"") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(_schema(allow_secrets=True, read_only_root=True)),
    )

    await executor.execute(
        ToolDefinition(
            name="secret-a",
            description="Secret A",
            required_secrets=["api_token"],
            execution_mode=ExecutionMode.CONTAINER,
            container_command=["python", "-c", "print('{}')"],
        ),
        ToolCall(session_id="session-1", name="secret-a"),
        {"api_token": "a"},
    )
    await executor.execute(
        ToolDefinition(
            name="secret-b",
            description="Secret B",
            required_secrets=["api_token"],
            execution_mode=ExecutionMode.CONTAINER,
            container_command=["python", "-c", "print('{}')"],
        ),
        ToolCall(session_id="session-1", name="secret-b"),
        {"api_token": "b"},
    )

    run_commands = [command for command in commands if command[1] == "run"]
    remove_commands = [command for command in commands if command[1] == "rm"]
    assert len(run_commands) == 2
    assert len(remove_commands) == 2
    assert all(command[2] == "-fv" for command in remove_commands)
    assert (
        run_commands[0][run_commands[0].index("--name") + 1]
        != run_commands[1][run_commands[1].index("--name") + 1]
    )


@pytest.mark.anyio
async def test_docker_executor_reuses_persistent_secret_container_per_tool(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: bytes = b"") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(_schema(allow_secrets=True, read_only_root=True, persistent_secrets=True)),
    )
    definition = ToolDefinition(
        name="secret-tool",
        description="Secret tool",
        required_secrets=["api_token"],
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    first = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="secret-tool"),
        {"api_token": "a"},
    )
    second = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="secret-tool"),
        {"api_token": "b"},
    )

    assert first.status == "ok"
    assert second.status == "ok"
    assert [command[1] for command in commands if command[1] not in {"image", "ps"}] == [
        "run",
        "exec",
        "exec",
    ]
    exec_commands = [command for command in commands if command[1] == "exec"]
    assert exec_commands[0][3] == exec_commands[1][3]


@pytest.mark.anyio
async def test_docker_executor_isolates_persistent_secret_containers_by_tool(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: bytes = b"") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(_schema(allow_secrets=True, read_only_root=True, persistent_secrets=True)),
    )

    for tool_name in ("secret-a", "secret-b"):
        await executor.execute(
            ToolDefinition(
                name=tool_name,
                description=tool_name,
                required_secrets=["api_token"],
                execution_mode=ExecutionMode.CONTAINER,
                container_command=["python", "-c", "print('{}')"],
            ),
            ToolCall(session_id="session-1", name=tool_name),
            {"api_token": tool_name},
        )

    run_commands = [command for command in commands if command[1] == "run"]
    exec_commands = [command for command in commands if command[1] == "exec"]
    assert len(run_commands) == 2
    assert exec_commands[0][3] != exec_commands[1][3]


@pytest.mark.anyio
async def test_docker_executor_rejects_secret_image_with_declared_volumes(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: bytes = b"", returncode: int = 0) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return Completed(b'{"/data": {}}')
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(_schema(allow_secrets=True, read_only_root=True)),
    )
    definition = ToolDefinition(
        name="secret-tool",
        description="Secret tool",
        required_secrets=["api_token"],
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="secret-tool"),
        {"api_token": "a"},
    )

    assert result.status == "error"
    assert "declares Docker volumes" in (result.error or "")
    assert [command[1] for command in commands if command[1] != "ps"] == ["image"]
    assert "run" not in [command[1] for command in commands]


@pytest.mark.anyio
async def test_docker_executor_bounds_captured_output(monkeypatch):
    from harness.execution import container as container_module
    from harness.execution.process import ProcessOutputLimitExceeded

    commands: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = b""
        stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            raise ProcessOutputLimitExceeded(stdout=b"x" * 64, stderr=b"", limit_bytes=64)
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('x' * 1024)"],
        max_output_bytes=64,
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="container"), {}
    )

    assert result.status == "error"
    assert "output exceeded 64 bytes" in (result.error or "")
    assert result.output == {"stdout": "x" * 64}
    assert [command[1] for command in commands if command[1] != "ps"] == [
        "run",
        "exec",
        "rm",
    ]


@pytest.mark.anyio
async def test_docker_executor_startup_cleanup_does_not_change_success_to_timeout(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = b'{"ok": true}'
        stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "rm":
            await container_module.anyio.sleep(1)
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker", cleanup_timeout_seconds=0.01)
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
        timeout_seconds=0.01,
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="container"), {}
    )

    assert result.status == "ok"
    assert [command[1] for command in commands if command[1] != "ps"] == [
        "rm",
        "run",
        "exec",
    ]


@pytest.mark.anyio
async def test_docker_executor_retains_container_state_when_cleanup_fails(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []
    rm_count = 0

    class Completed:
        def __init__(self, stdout: bytes = b"", returncode: int = 0) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        nonlocal rm_count
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        if command[1] == "rm":
            rm_count += 1
            return Completed(returncode=1 if rm_count == 1 else 0)
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="container"), {}
    )

    assert result.status == "ok"
    assert len(executor._containers) == 1

    await executor.cleanup_all()

    assert len(executor._containers) == 1

    await executor.cleanup_all()

    assert len(executor._containers) == 0
    assert [command[1] for command in commands if command[1] != "ps"] == [
        "run",
        "exec",
        "rm",
        "rm",
    ]


@pytest.mark.anyio
async def test_docker_executor_rejects_success_when_secret_cleanup_fails(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: bytes = b"", returncode: int = 0) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return Completed(b"{}")
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        if command[1] == "rm":
            return Completed(returncode=1)
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(_schema(allow_secrets=True, read_only_root=True)),
    )
    definition = ToolDefinition(
        name="secret-tool",
        description="Secret tool",
        required_secrets=["api_token"],
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="secret-tool"),
        {"api_token": "secret"},
    )

    assert result.status == "error"
    assert "cleanup failed" in (result.error or "")
    assert result.metadata["cleanup_failed"] is True
    assert [command[1] for command in commands if command[1] not in {"image", "ps"}] == [
        "run",
        "exec",
        "rm",
    ]


@pytest.mark.anyio
async def test_docker_executor_does_not_reuse_secret_one_shot_after_cleanup_failure(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: bytes = b"", returncode: int = 0) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return Completed(b"{}")
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        if command[1] == "rm":
            return Completed(returncode=1)
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(_schema(allow_secrets=True, read_only_root=True)),
    )
    definition = ToolDefinition(
        name="secret-tool",
        description="Secret tool",
        required_secrets=["api_token"],
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )
    call = ToolCall(session_id="session-1", call_id="retryable", name="secret-tool")

    first = await executor.execute(definition, call, {"api_token": "first"})
    second = await executor.execute(definition, call, {"api_token": "second"})

    assert first.status == "error"
    assert second.status == "error"
    run_commands = [command for command in commands if command[1] == "run"]
    exec_commands = [command for command in commands if command[1] == "exec"]
    assert len(run_commands) == 2
    assert len(exec_commands) == 2
    assert (
        run_commands[0][run_commands[0].index("--name") + 1]
        != run_commands[1][run_commands[1].index("--name") + 1]
    )


@pytest.mark.anyio
async def test_docker_executor_schedules_delayed_session_cleanup(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    class Completed:
        def __init__(self, stdout: bytes = b"") -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            return Completed(b'{"ok": true}')
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="container"),
        {},
    )
    await executor.schedule_session_cleanup("session-1", delay_minutes=0)
    await container_module.anyio.sleep(0.01)

    assert result.status == "ok"
    assert [command[1] for command in commands if command[1] != "ps"][:3] == [
        "run",
        "exec",
        "rm",
    ]


@pytest.mark.anyio
async def test_docker_session_cleanup_retries_quarantined_secret_container(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []
    cleanup_attempts = 0

    class Completed:
        def __init__(
            self,
            stdout: bytes = b"",
            stderr: bytes = b"",
            returncode: int = 0,
        ) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        nonlocal cleanup_attempts
        _ = kwargs
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return Completed(b"{}")
        if command[1] == "rm":
            cleanup_attempts += 1
            if cleanup_attempts == 1:
                return Completed(stderr=b"cleanup failed", returncode=1)
            return Completed()
        if command[1] == "ps":
            return Completed()
        return Completed()

    async def fake_limited_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        return Completed(b'{"ok": true}')

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_limited_process)
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(_schema(allow_secrets=True, read_only_root=True)),
    )
    definition = ToolDefinition(
        name="secret-tool",
        description="Secret tool",
        required_secrets=["api_token"],
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="secret-tool"),
        {"api_token": "secret"},
    )
    await executor.schedule_session_cleanup("session-1", delay_minutes=0)
    await container_module.anyio.sleep(0.01)

    assert result.status == "error"
    assert result.metadata["cleanup_failed"]
    assert cleanup_attempts == 2
    assert executor._containers == {}


@pytest.mark.anyio
async def test_docker_executor_removes_named_container_on_timeout(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            raise TimeoutError

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
        timeout_seconds=0.01,
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="container"), {}
    )

    assert result.status == "timeout"
    run_command = next(command for command in commands if command[1] == "run")
    container_name = run_command[run_command.index("--name") + 1]
    assert ["docker", "rm", "-f", container_name] in commands


@pytest.mark.anyio
async def test_docker_timeout_waits_for_other_session_calls_before_cleanup(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []
    active_execs = 0
    both_execs_started = anyio.Event()
    release_second_timeout = anyio.Event()

    class Completed:
        returncode = 0
        stdout = b""
        stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        nonlocal active_execs
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            active_execs += 1
            if active_execs == 2:
                both_execs_started.set()
            await both_execs_started.wait()
            if command[-1] == "tool-b":
                await release_second_timeout.wait()
            raise TimeoutError
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")
    definition_a = ToolDefinition(
        name="tool-a",
        description="Container A",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["tool-a"],
        timeout_seconds=1,
    )
    definition_b = ToolDefinition(
        name="tool-b",
        description="Container B",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["tool-b"],
        timeout_seconds=1,
    )

    results: list[ToolResult] = []

    async def run(definition: ToolDefinition) -> None:
        results.append(
            await executor.execute(
                definition,
                ToolCall(session_id="session-1", name=definition.name),
                {},
            )
        )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run, definition_a)
        task_group.start_soon(run, definition_b)
        await both_execs_started.wait()
        await anyio.sleep(0.05)
        assert not any(command[1] == "rm" for command in commands)
        release_second_timeout.set()

    assert sorted(result.status for result in results) == ["timeout", "timeout"]
    assert any(command[1] == "rm" for command in commands)


@pytest.mark.anyio
async def test_docker_session_cleanup_blocks_new_session_calls(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []
    first_exec_started = anyio.Event()
    release_first_exec = anyio.Event()

    class Completed:
        def __init__(self, stdout: bytes = b'{"ok": true}') -> None:
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "exec" and command[-1] == "tool-a":
            first_exec_started.set()
            await release_first_exec.wait()
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker")
    definition_a = ToolDefinition(
        name="tool-a",
        description="Container A",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["tool-a"],
    )
    definition_b = ToolDefinition(
        name="tool-b",
        description="Container B",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["tool-b"],
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            executor.execute,
            definition_a,
            ToolCall(session_id="session-1", name="tool-a"),
            {},
        )
        await first_exec_started.wait()
        task_group.start_soon(executor.cleanup_session, "session-1")
        while not executor._session_lifecycle_locks["session-1"].locked():
            await anyio.sleep(0)
        task_group.start_soon(
            executor.execute,
            definition_b,
            ToolCall(session_id="session-1", name="tool-b"),
            {},
        )
        await anyio.sleep(0.05)
        assert not any(command[1] == "exec" and command[-1] == "tool-b" for command in commands)
        release_first_exec.set()

    rm_index = next(index for index, command in enumerate(commands) if command[1] == "rm")
    tool_b_exec_index = next(
        index
        for index, command in enumerate(commands)
        if command[1] == "exec" and command[-1] == "tool-b"
    )
    assert rm_index < tool_b_exec_index


@pytest.mark.anyio
async def test_docker_session_cleanup_prunes_lock_state_after_one_shot(monkeypatch):
    from harness.execution import container as container_module

    class Completed:
        def __init__(self, stdout: bytes = b'{"ok": true}', returncode: int = 0) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = b""

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        if command[1:3] == ["image", "inspect"]:
            return Completed(b"{}")
        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(_schema(allow_secrets=True, read_only_root=True)),
    )
    definition = ToolDefinition(
        name="secret-tool",
        description="Secret tool",
        required_secrets=["api_token"],
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="secret-tool"),
        {"api_token": "secret"},
    )
    await executor.schedule_session_cleanup("session-1", delay_minutes=0)

    assert result.status == "ok"
    assert "session-1" not in executor._session_lifecycle_locks
    assert "session-1" not in executor._session_activity_events
    assert not any(key[0] == "session-1" for key in executor._container_locks)
    assert not any(key[0] == "session-1" for key in executor._exec_locks)


@pytest.mark.anyio
async def test_docker_executor_applies_timeout_to_container_startup(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "run":
            await container_module.anyio.sleep(1)

        class Completed:
            returncode = 0
            stdout = b'{"ok": true}'
            stderr = b""

        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker", cleanup_timeout_seconds=0.01)
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
        timeout_seconds=0.01,
    )

    result = await executor.execute(
        definition,
        ToolCall(session_id="session-1", name="container"),
        {},
    )

    assert result.status == "timeout"
    run_command = next(command for command in commands if command[1] == "run")
    container_name = run_command[run_command.index("--name") + 1]
    assert ["docker", "rm", "-f", container_name] in commands


@pytest.mark.anyio
async def test_docker_executor_bounds_timeout_cleanup(monkeypatch):
    from harness.execution import container as container_module

    commands: list[list[str]] = []

    async def fake_run_process(command, **kwargs):  # noqa: ANN001
        _ = kwargs
        commands.append(command)
        if command[1] == "exec":
            raise TimeoutError
        if command[1] == "rm":
            await container_module.anyio.sleep(1)
            return None

        class Completed:
            returncode = 0
            stdout = b""
            stderr = b""

        return Completed()

    monkeypatch.setattr(container_module.anyio, "run_process", fake_run_process)
    monkeypatch.setattr(container_module, "run_limited_process", fake_run_process)
    executor = DockerContainerExecutor(docker_bin="docker", cleanup_timeout_seconds=0.01)
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
        timeout_seconds=0.01,
    )

    started = time.monotonic()
    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="container"), {}
    )

    assert result.status == "timeout"
    assert time.monotonic() - started < 0.2
    assert any(command[0:3] == ["docker", "rm", "-f"] for command in commands)


@pytest.mark.anyio
async def test_docker_executor_rejects_mount_without_allowed_root(tmp_path):
    executor = DockerContainerExecutor(
        docker_bin="docker",
        schemas=_schemas(_schema(mount=tmp_path)),
    )
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="container"), {}
    )

    assert result.status == "error"
    assert "allowed mount root" in (result.error or "")


@pytest.mark.anyio
async def test_docker_executor_rejects_mount_outside_allowed_root(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    executor = DockerContainerExecutor(
        docker_bin="docker",
        allowed_mount_root=allowed,
        schemas=_schemas(_schema(mount=outside)),
    )
    definition = ToolDefinition(
        name="container",
        description="Container",
        execution_mode=ExecutionMode.CONTAINER,
        container_command=["python", "-c", "print('{}')"],
    )

    result = await executor.execute(
        definition, ToolCall(session_id="session-1", name="container"), {}
    )

    assert result.status == "error"
    assert "outside the allowed mount root" in (result.error or "")


@pytest.mark.container
@pytest.mark.anyio
async def test_docker_container_executor(storage):
    settings = HarnessSettings()
    docker_bin = settings.docker_bin
    await _require_docker_available(docker_bin)

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="container-echo",
            description="Echo in a container",
            required_capabilities=["tool:container"],
            execution_mode=ExecutionMode.CONTAINER,
            container_command=[
                "python",
                "-c",
                (
                    "import json, sys; "
                    "payload=json.load(sys.stdin); "
                    "print(json.dumps({'echo': payload['arguments']['message']}))"
                ),
            ],
        )
    )
    executor = DockerContainerExecutor(docker_bin=docker_bin)
    gateway = ToolExecutionGateway(
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset({"tool:container"}))),
        storage=storage,
        executors={ExecutionMode.CONTAINER: executor},
    )

    result = await gateway.execute(
        ToolCall(session_id="container-e2e", name="container-echo", arguments={"message": "hi"})
    )

    assert result.status == "ok"
    assert result.output == {"echo": "hi"}
    await executor.cleanup_session("container-e2e")


@pytest.mark.container
@pytest.mark.anyio
async def test_docker_persistent_secret_container_executor():
    settings = HarnessSettings()
    docker_bin = settings.docker_bin
    await _require_docker_available(docker_bin)

    executor = DockerContainerExecutor(
        docker_bin=docker_bin,
        schemas=_schemas(_schema(allow_secrets=True, read_only_root=True, persistent_secrets=True)),
    )
    definition = ToolDefinition(
        name="secret-state",
        description="Persist state while receiving per-call secrets",
        required_secrets=["api_token"],
        execution_mode=ExecutionMode.CONTAINER,
        container_command=[
            "python",
            "-c",
            (
                "import json, pathlib, sys; "
                "payload=json.load(sys.stdin); "
                "path=pathlib.Path('/work/count.txt'); "
                "count=int(path.read_text()) if path.exists() else 0; "
                "count += 1; "
                "path.write_text(str(count)); "
                "print(json.dumps({'count': count, 'token': payload['secrets']['api_token']}))"
            ),
        ],
    )

    first = await executor.execute(
        definition,
        ToolCall(session_id="secret-container-e2e", name="secret-state"),
        {"api_token": "first-token"},
    )
    second = await executor.execute(
        definition,
        ToolCall(session_id="secret-container-e2e", name="secret-state"),
        {"api_token": "second-token"},
    )
    container_name = second.metadata["container"]

    assert first.status == "ok"
    assert second.status == "ok"
    assert first.output == {"count": 1, "token": "first-token"}
    assert second.output == {"count": 2, "token": "second-token"}
    assert first.metadata["container"] == container_name

    await executor.cleanup_session("secret-container-e2e")
    inspect = await anyio.run_process(
        [docker_bin, "inspect", container_name],
        check=False,
        env=executor._docker_env(),
    )
    assert inspect.returncode != 0
