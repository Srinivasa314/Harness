from __future__ import annotations

from datetime import datetime
from typing import Any

from harness.memory import MemoryManager
from harness.schemas import ExecutionMode, MemoryScope, ToolDefinition
from harness.tools.registry import ToolRegistry


def register_memory_tools(
    registry: ToolRegistry,
    memory: MemoryManager,
    *,
    source_session_id: str | None = None,
) -> None:
    async def memory_store(arguments: dict[str, Any], secrets: dict[str, str]) -> dict[str, str]:
        _ = secrets
        text = str(arguments["text"]).strip()
        if not text:
            raise ValueError("text must not be empty")
        scope = _memory_scope(arguments.get("scope"))
        metadata = arguments.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        confidence = float(arguments.get("confidence", 1.0))
        expires_at = _expires_at(arguments.get("expires_at"))
        record = await memory.remember(
            text,
            scope=scope,
            confidence=confidence,
            metadata=metadata,
            source_session_id=source_session_id,
            expires_at=expires_at,
        )
        return {"memory_id": record.id}

    registry.register(memory_store_definition(), memory_store)


def memory_store_definition() -> ToolDefinition:
    return ToolDefinition(
        name="memory.store",
        description=(
            "Store durable memory for future agent runs. Use this only for reusable "
            "user preferences, project facts, conventions, or standing instructions; "
            "do not store transient current-task facts unless the user asks you to."
        ),
        execution_mode=ExecutionMode.IN_PROCESS,
        required_capabilities=["memory:write"],
        input_schema={
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Durable memory text to retrieve in future runs.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["session", "agent", "global"],
                    "default": "agent",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 1.0,
                },
                "metadata": {
                    "type": "object",
                    "description": "Application-specific provenance for the memory.",
                    "additionalProperties": True,
                },
                "expires_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Optional ISO-8601 expiry timestamp.",
                },
            },
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "required": ["memory_id"],
            "properties": {"memory_id": {"type": "string"}},
            "additionalProperties": False,
        },
    )


def _memory_scope(value: object) -> MemoryScope:
    if value is None:
        return MemoryScope.AGENT
    if isinstance(value, str):
        try:
            return MemoryScope(value)
        except ValueError as exc:
            raise ValueError(f"Unknown memory scope: {value}") from exc
    raise ValueError("scope must be a string")


def _expires_at(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expires_at must be a string")
    return datetime.fromisoformat(value)
