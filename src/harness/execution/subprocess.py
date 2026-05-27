from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

from harness.execution.base import ToolExecutor
from harness.execution.process import ProcessOutputLimitExceeded, run_limited_process
from harness.process_env import DEFAULT_ENV_ALLOWLIST, filtered_env
from harness.schemas import ToolCall, ToolDefinition, ToolResult, utc_now


class SubprocessExecutor(ToolExecutor):
    def __init__(
        self,
        *,
        base_env: Mapping[str, str] | None = None,
        inherit_env: Iterable[str] = DEFAULT_ENV_ALLOWLIST,
    ) -> None:
        self.env = filtered_env(source=base_env, inherit=inherit_env)

    async def execute(
        self,
        definition: ToolDefinition,
        call: ToolCall,
        secrets: dict[str, str],
    ) -> ToolResult:
        if not definition.subprocess_command:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                error="Subprocess tool has no command",
            )

        started = utc_now()
        payload = json.dumps({"arguments": call.arguments, "secrets": secrets}).encode()
        try:
            completed = await run_limited_process(
                definition.subprocess_command,
                input=payload,
                env=self.env,
                timeout_seconds=definition.timeout_seconds,
                max_output_bytes=definition.max_output_bytes,
            )
        except TimeoutError:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="timeout",
                error=f"Tool timed out after {definition.timeout_seconds} seconds",
                started_at=started,
                ended_at=utc_now(),
            )
        except ProcessOutputLimitExceeded as exc:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                output={"stdout": exc.stdout.decode(errors="replace")},
                error=str(exc),
                started_at=started,
                ended_at=utc_now(),
            )
        except Exception as exc:  # noqa: BLE001 - normalize process launch failures.
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                error=f"Subprocess executor failed: {exc}",
                started_at=started,
                ended_at=utc_now(),
            )

        stderr_text = completed.stderr.decode(errors="replace")
        stdout_text = completed.stdout.decode(errors="replace")
        if completed.returncode != 0:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                output={"stdout": stdout_text},
                error=stderr_text or f"Process exited with {completed.returncode}",
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
            metadata={"stderr": stderr_text} if stderr_text else {},
            started_at=started,
            ended_at=utc_now(),
        )
