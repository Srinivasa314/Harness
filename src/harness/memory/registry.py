from __future__ import annotations

from collections.abc import Callable

from harness.config import HarnessSettings
from harness.memory.embeddings import EmbeddingProvider
from harness.plugins import ProviderRegistry

EmbeddingProviderFactory = Callable[[HarnessSettings], EmbeddingProvider]


class EmbeddingProviderRegistry(ProviderRegistry[EmbeddingProviderFactory]):
    """Registry for settings-selected embedding provider factories."""

    def create(self, name: str, settings: HarnessSettings) -> EmbeddingProvider:
        try:
            factory = self.get(name)
        except KeyError as exc:
            raise ValueError(f"Unknown embedding provider: {name}") from exc
        return factory(settings)
