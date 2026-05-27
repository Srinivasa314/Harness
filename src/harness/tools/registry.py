from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from harness.schemas import ExecutionMode, ToolDefinition

ToolFunction = Callable[[dict[str, Any], dict[str, str]], Awaitable[Any] | Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._functions: dict[str, ToolFunction] = {}

    def register(
        self,
        definition: ToolDefinition,
        function: ToolFunction | None = None,
    ) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool already registered: {definition.name}")
        Draft202012Validator.check_schema(definition.input_schema)
        if definition.output_schema is not None:
            Draft202012Validator.check_schema(definition.output_schema)
        if definition.execution_mode == ExecutionMode.IN_PROCESS and function is None:
            raise ValueError("In-process tools require a Python function")
        if (
            definition.execution_mode == ExecutionMode.IN_PROCESS
            and function is not None
            and not inspect.iscoroutinefunction(function)
        ):
            raise ValueError("In-process tools require async Python functions")
        self._definitions[definition.name] = definition
        if function is not None:
            self._functions[definition.name] = function

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def function_for(self, name: str) -> ToolFunction:
        try:
            return self._functions[name]
        except KeyError as exc:
            raise KeyError(f"No in-process function registered for tool: {name}") from exc

    def list_definitions(self) -> list[ToolDefinition]:
        return list(self._definitions.values())


def load_tool_registry(
    path: str | Path,
    *,
    builtins: dict[str, ToolFunction] | None = None,
) -> ToolRegistry:
    payload = json.loads(Path(path).read_text())
    raw_tools = payload["tools"] if isinstance(payload, dict) else payload
    registry = ToolRegistry()
    builtin_functions = builtins or {}
    for raw_tool in raw_tools:
        definition = ToolDefinition.model_validate(raw_tool)
        if not definition.required_capabilities:
            raise ValueError(
                f"Externally loaded tool {definition.name!r} requires at least one capability"
            )
        function = None
        if definition.execution_mode == ExecutionMode.IN_PROCESS:
            function_name = raw_tool.get("function")
            if not isinstance(function_name, str) or function_name not in builtin_functions:
                raise ValueError(
                    f"In-process tool {definition.name!r} requires an explicit built-in function"
                )
            function = builtin_functions[function_name]
        registry.register(definition, function)
    return registry
