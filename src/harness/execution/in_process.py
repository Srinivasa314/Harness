from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import anyio

from harness.execution.base import ToolExecutor
from harness.schemas import ToolCall, ToolDefinition, ToolResult, utc_now


class InProcessExecutor(ToolExecutor):
    def __init__(self, function_lookup: Callable[[str], Callable[..., Any]]) -> None:
        self.function_lookup = function_lookup

    async def execute(
        self,
        definition: ToolDefinition,
        call: ToolCall,
        secrets: dict[str, str],
    ) -> ToolResult:
        started = utc_now()
        function = self.function_lookup(definition.name)
        if not inspect.iscoroutinefunction(function):
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                error=(
                    "In-process tools must be async callables; "
                    "use subprocess or container mode for sync work"
                ),
                started_at=started,
                ended_at=utc_now(),
            )
        try:
            with anyio.fail_after(definition.timeout_seconds):
                output = await function(call.arguments, secrets)
            status = "ok"
            error = None
        except TimeoutError:
            output = None
            status = "timeout"
            error = f"Tool timed out after {definition.timeout_seconds} seconds"
        except Exception as exc:  # noqa: BLE001 - normalize all tool exceptions.
            output = None
            status = "error"
            error = str(exc)
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            status=status,
            output=output,
            error=error,
            started_at=started,
            ended_at=utc_now(),
        )
