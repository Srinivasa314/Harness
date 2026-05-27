from __future__ import annotations

from abc import ABC, abstractmethod

from harness.schemas import ToolCall, ToolDefinition, ToolResult


class ToolExecutor(ABC):
    @abstractmethod
    async def execute(
        self,
        definition: ToolDefinition,
        call: ToolCall,
        secrets: dict[str, str],
    ) -> ToolResult:
        raise NotImplementedError
