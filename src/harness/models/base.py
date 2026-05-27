from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelMessage(BaseModel):
    role: str
    content: str


class ModelResponse(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    tool_calls: list[ModelToolCall] = Field(default_factory=list)


class ModelToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    kind: Literal["final", "tool_calls"]
    content: str = ""
    tool_calls: list[ModelToolCall] = Field(default_factory=list)


class ModelProvider(ABC):
    @abstractmethod
    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        raise NotImplementedError
