from __future__ import annotations

from harness.config import HarnessSettings
from harness.memory.embeddings import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    MiniLMEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from harness.memory.registry import EmbeddingProviderRegistry


def default_embedding_provider_registry() -> EmbeddingProviderRegistry:
    registry = EmbeddingProviderRegistry()
    registry.register(
        "hash",
        lambda settings: HashEmbeddingProvider(dimensions=settings.embedding_dimensions),
    )
    registry.register(
        "minilm",
        lambda settings: MiniLMEmbeddingProvider(model_name=settings.embedding_model),
    )
    registry.register("openai", _create_openai_embedding_provider)
    return registry


def create_embedding_provider(
    settings: HarnessSettings,
    registry: EmbeddingProviderRegistry | None = None,
) -> EmbeddingProvider:
    return (registry or default_embedding_provider_registry()).create(
        settings.embedding_provider,
        settings,
    )


def _create_openai_embedding_provider(settings: HarnessSettings) -> EmbeddingProvider:
    if not settings.openai_api_key:
        raise ValueError("HARNESS_OPENAI_API_KEY is required for embedding_provider=openai")
    return OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        base_url=settings.openai_base_url,
        dimensions=settings.openai_embedding_dimensions,
    )
