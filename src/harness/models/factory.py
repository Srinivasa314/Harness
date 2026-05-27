from __future__ import annotations

from pathlib import Path

from harness.config import HarnessSettings
from harness.models.base import ModelProvider
from harness.models.codex import DEFAULT_CODEX_COMMAND, CodexCliProvider
from harness.models.openai import OpenAIResponsesProvider
from harness.models.registry import ModelProviderRegistry


def default_model_provider_registry() -> ModelProviderRegistry:
    registry = ModelProviderRegistry()
    registry.register("none", lambda _settings: None)
    registry.register("openai", _create_openai_provider)
    registry.register("codex", _create_codex_provider)
    return registry


def create_model_provider(
    settings: HarnessSettings,
    registry: ModelProviderRegistry | None = None,
) -> ModelProvider | None:
    return (registry or default_model_provider_registry()).create(settings.model_provider, settings)


def _create_openai_provider(settings: HarnessSettings) -> ModelProvider:
    if not settings.openai_api_key:
        raise ValueError("HARNESS_OPENAI_API_KEY is required for model_provider=openai")
    return OpenAIResponsesProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
    )


def _create_codex_provider(settings: HarnessSettings) -> ModelProvider:
    use_safe_defaults = _is_codex_exec_command(settings.codex_command)
    codex_command = (
        None if settings.codex_command == DEFAULT_CODEX_COMMAND else settings.codex_command
    )
    return CodexCliProvider(
        command=codex_command,
        model=settings.codex_model,
        use_safe_defaults=use_safe_defaults,
    )


def _is_codex_exec_command(command: list[str]) -> bool:
    return len(command) == 2 and Path(command[0]).name == "codex" and command[1] == "exec"
