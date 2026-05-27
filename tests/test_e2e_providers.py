from __future__ import annotations

import os
import shutil

import pytest

from harness.agent import AgentSessionManager
from harness.config import HarnessSettings
from harness.memory import MiniLMEmbeddingProvider, OpenAIEmbeddingProvider
from harness.models import (
    DEFAULT_CODEX_COMMAND,
    CodexCliProvider,
    ModelMessage,
    OpenAIResponsesProvider,
    create_model_provider,
)
from harness.storage import PostgresStorage

pytestmark = pytest.mark.anyio


@pytest.mark.e2e
@pytest.mark.postgres
async def test_postgres_storage_e2e():
    dsn = os.environ.get("HARNESS_POSTGRES_DSN")
    if not dsn:
        pytest.skip("HARNESS_POSTGRES_DSN is not set")
    storage = PostgresStorage(dsn)
    try:
        await storage.migrate()
        session = await AgentSessionManager(storage).create({"e2e": True})
        fetched = await storage.get_session(session.id)
        assert fetched is not None
        assert fetched.id == session.id
    finally:
        await storage.close()


@pytest.mark.e2e
@pytest.mark.provider
async def test_openai_provider_e2e():
    if os.environ.get("HARNESS_RUN_OPENAI_E2E") != "1":
        pytest.skip("HARNESS_RUN_OPENAI_E2E=1 is not set")
    settings = HarnessSettings()
    if not settings.openai_api_key:
        pytest.skip("HARNESS_OPENAI_API_KEY is not set")
    provider = OpenAIResponsesProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        timeout_seconds=120,
    )

    response = await provider.complete(
        [ModelMessage(role="user", content="Return exactly: harness-ok")]
    )

    assert "harness-ok" in response.content
    assert response.metadata["request_id"]


@pytest.mark.e2e
@pytest.mark.provider
async def test_openai_embedding_provider_e2e():
    if os.environ.get("HARNESS_RUN_OPENAI_E2E") != "1":
        pytest.skip("HARNESS_RUN_OPENAI_E2E=1 is not set")
    settings = HarnessSettings()
    if not settings.openai_api_key:
        pytest.skip("HARNESS_OPENAI_API_KEY is not set")
    provider = OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        base_url=settings.openai_base_url,
        dimensions=settings.openai_embedding_dimensions,
        timeout_seconds=120,
    )

    embeddings = await provider.embed(["container sandbox", "memory retrieval"])

    assert len(embeddings) == 2
    if settings.openai_embedding_dimensions is None:
        assert len(embeddings[0]) > 100
    else:
        assert len(embeddings[0]) == settings.openai_embedding_dimensions
    assert len(embeddings[0]) == len(embeddings[1])


@pytest.mark.e2e
@pytest.mark.provider
async def test_codex_cli_provider_e2e():
    if os.environ.get("HARNESS_RUN_CODEX_E2E") != "1":
        pytest.skip("HARNESS_RUN_CODEX_E2E=1 is not set")
    if shutil.which("codex") is None:
        pytest.skip("codex CLI is not installed")
    provider = create_model_provider(
        HarnessSettings(
            model_provider="codex",
            codex_command=DEFAULT_CODEX_COMMAND,
        )
    )
    assert isinstance(provider, CodexCliProvider)
    provider.timeout_seconds = 600

    response = await provider.complete(
        [ModelMessage(role="user", content="Return exactly the text harness-ok and nothing else.")]
    )

    assert "harness-ok" in response.content
    command = response.metadata["command"]
    assert "--sandbox" in command
    assert "read-only" in command
    assert "--ephemeral" in command
    assert "--ignore-rules" in command
    assert "--skip-git-repo-check" in command


@pytest.mark.e2e
async def test_minilm_embedding_provider_e2e():
    if os.environ.get("HARNESS_RUN_MINILM_E2E") != "1":
        pytest.skip("HARNESS_RUN_MINILM_E2E=1 is not set")
    provider = MiniLMEmbeddingProvider()

    embeddings = await provider.embed(["container sandbox", "memory retrieval"])

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 100
    assert len(embeddings[0]) == len(embeddings[1])
