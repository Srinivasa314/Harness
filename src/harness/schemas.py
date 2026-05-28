from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class ExecutionMode(StrEnum):
    IN_PROCESS = "in_process"
    SUBPROCESS = "subprocess"
    CONTAINER = "container"


class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    name: str
    status: Literal["ok", "error", "denied", "timeout"]
    output: Any = None
    artifacts: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    id: str
    session_id: str
    tool_name: str
    status: str
    started_at: datetime
    ended_at: datetime
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    artifacts: list[str] = Field(default_factory=list)
    error: str | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class ToolDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    output_schema: dict[str, Any] | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    required_secrets: list[str] = Field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SUBPROCESS
    timeout_seconds: float = 30
    max_output_bytes: int = Field(default=1_000_000, ge=1)
    subprocess_command: list[str] | None = None
    container_schema: str = "default"
    container_command: list[str] | None = None

    @property
    def requires_secrets(self) -> bool:
        return bool(self.required_secrets)


class ContainerSchema(BaseModel):
    name: str
    image: str
    network: bool = False
    mount: Path | None = None
    mount_read_only: bool = True
    workdir: str = "/work"
    read_only_root: bool = True
    tmpfs_tmp: bool = True
    tmpfs_workdir: bool = True
    share_across_tools: bool = False
    memory: str = "512m"
    cpus: str = "1.0"
    pids_limit: int = 128
    cap_drop_all: bool = True
    no_new_privileges: bool = True
    allow_secrets: bool = False
    persistent_secrets: bool = False
    keepalive_command: list[str] = Field(default_factory=lambda: ["sleep", "infinity"])

    @model_validator(mode="after")
    def validate_secret_schema(self) -> ContainerSchema:
        if self.image.startswith("-"):
            raise ValueError("Container image must not start with '-'")
        if self.allow_secrets:
            if not self.read_only_root:
                raise ValueError("Secret-enabled container schemas require read_only_root=true")
            if self.mount is not None:
                raise ValueError("Secret-enabled container schemas cannot use host mounts")
            if self.persistent_secrets and not (self.tmpfs_tmp and self.tmpfs_workdir):
                raise ValueError(
                    "Persistent secret-enabled container schemas require tmpfs_tmp=true "
                    "and tmpfs_workdir=true"
                )
        elif self.persistent_secrets:
            raise ValueError("persistent_secrets requires allow_secrets=true")
        return self


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class MemoryScope(StrEnum):
    SESSION = "session"
    AGENT = "agent"
    GLOBAL = "global"


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    namespace: str
    text: str
    embedding: list[float]
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    scope: MemoryScope = MemoryScope.AGENT
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_session_id: str | None = None
    source_turn_id: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ArtifactRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    tool_call_id: str | None = None
    path: str
    media_type: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
