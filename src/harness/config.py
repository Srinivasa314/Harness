from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HarnessSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HARNESS_",
        env_file=(".env", ".env.development"),
        extra="ignore",
    )

    storage_backend: str = "sqlite"
    sqlite_path: Path = Path("data/harness.sqlite3")
    postgres_dsn: str | None = None

    openai_api_key: str | None = None
    openai_model: str = "gpt-5.2"
    openai_base_url: str = "https://api.openai.com/v1"

    model_provider: str = "none"
    codex_command: list[str] = Field(default_factory=lambda: ["codex", "exec"])
    codex_model: str | None = None

    embedding_provider: str = "minilm"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 32
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int | None = None
    memory_enabled: bool = True
    memory_namespace: str = "default"
    memory_retrieval_limit: int = 5
    memory_min_score: float = -1.0
    memory_max_context_chars: int = 4_000
    memory_auto_capture: bool = False
    session_lease_ttl_seconds: float = 300
    session_lease_heartbeat_seconds: float | None = None
    context_compaction_enabled: bool = True
    context_max_chars: int = 120_000
    context_compaction_trigger_ratio: float = 0.8
    context_compaction_preserve_recent_messages: int = 8
    context_compaction_summarizer_input_max_chars: int = 24_000
    context_compaction_summary_max_chars: int = 4_000

    docker_bin: str = "docker"
    docker_mount_root: Path | None = None
    container_schemas_path: Path | None = None
    container_cleanup_delay_minutes: float = 5
    tool_capabilities: list[str] = Field(default_factory=list)

    secret_backend: str = "env"
    secret_env_prefix: str = "HARNESS_SECRET_"

    def __init__(self, **data: Any) -> None:
        if os.environ.get("HARNESS_DISABLE_ENV_FILES") == "1":
            data.setdefault("_env_file", None)
        super().__init__(**data)


def load_settings() -> HarnessSettings:
    return HarnessSettings()
