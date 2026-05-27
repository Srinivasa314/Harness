from __future__ import annotations

from collections.abc import Callable

from harness.config import HarnessSettings
from harness.models.base import ModelProvider
from harness.plugins import ProviderRegistry

ModelProviderFactory = Callable[[HarnessSettings], ModelProvider | None]


class ModelProviderRegistry(ProviderRegistry[ModelProviderFactory]):
    """Registry for settings-selected inference provider factories."""

    def create(self, name: str, settings: HarnessSettings) -> ModelProvider | None:
        try:
            factory = self.get(name)
        except KeyError as exc:
            raise ValueError(f"Unknown model provider: {name}") from exc
        return factory(settings)
