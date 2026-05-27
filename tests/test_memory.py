from __future__ import annotations

import pytest

from harness.memory import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    MemoryCandidate,
    MemoryExchange,
    MemoryExtractor,
    MemoryManager,
    MemoryPolicy,
    MemoryStore,
)
from harness.schemas import MemoryScope
from harness.storage import SQLiteStorage

pytestmark = pytest.mark.anyio


class CapturingEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return await super().embed(texts)


class KeywordEmbeddingProvider(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            lowered = text.lower()
            embeddings.append(
                [
                    1.0 if "alpha" in lowered else 0.0,
                    1.0 if "beta" in lowered else 0.0,
                ]
            )
        return embeddings


class StaticMemoryExtractor(MemoryExtractor):
    async def extract(self, exchange: MemoryExchange) -> list[MemoryCandidate]:
        return [
            MemoryCandidate(
                text=f"remembered {exchange.user_message}",
                scope=MemoryScope.SESSION,
                importance=0.7,
            )
        ]


async def _storage(tmp_path):
    storage = SQLiteStorage(tmp_path / "harness.sqlite3")
    await storage.migrate()
    return storage


async def test_memory_retrieval(tmp_path):
    storage = await _storage(tmp_path)
    memory = MemoryStore(storage, HashEmbeddingProvider())
    await memory.add("project", "docker container sandbox")
    await memory.add("project", "agent memory retrieval")

    results = await memory.retrieve("project", "container sandbox", limit=1)

    assert results[0].memory.text == "docker container sandbox"
    assert results[0].score > 0


async def test_memory_store_redacts_text_and_metadata(tmp_path):
    storage = await _storage(tmp_path)
    memory = MemoryStore(storage, HashEmbeddingProvider())

    await memory.add(
        "project",
        "api_key is sk-memory-secret",
        metadata={"api_key": "sk-memory-metadata"},
    )

    [saved] = await storage.list_memories("project")
    assert "sk-memory-secret" not in saved.text
    assert saved.metadata["api_key"] == "[REDACTED]"


async def test_memory_store_redacts_text_before_embedding(tmp_path):
    storage = await _storage(tmp_path)
    embeddings = CapturingEmbeddingProvider()
    memory = MemoryStore(storage, embeddings)

    await memory.add("project", "api_key is sk-memory-embed-secret")
    await memory.retrieve("project", "token=plain-query-token")

    assert "sk-memory-embed-secret" not in str(embeddings.texts)
    assert "plain-query-token" not in str(embeddings.texts)
    assert any("[REDACTED]" in text for text in embeddings.texts)


async def test_memory_store_accepts_class_based_embedding_provider(tmp_path):
    storage = await _storage(tmp_path)
    memory = MemoryStore(storage, KeywordEmbeddingProvider())

    await memory.add("project", "beta note")
    await memory.add("project", "alpha note")

    results = await memory.retrieve("project", "alpha query", limit=2)

    assert [result.memory.text for result in results] == ["alpha note", "beta note"]


async def test_memory_store_filters_records_from_other_embedding_config(tmp_path):
    storage = await _storage(tmp_path)
    hash_memory = MemoryStore(storage, HashEmbeddingProvider(dimensions=8))
    query_memory = MemoryStore(storage, HashEmbeddingProvider(dimensions=16))

    await hash_memory.add("project", "alpha note")

    results = await query_memory.retrieve("project", "alpha query")

    assert results == []


async def test_memory_manager_builds_context_with_structured_filters(tmp_path):
    storage = await _storage(tmp_path)
    manager = MemoryManager(
        MemoryStore(storage, KeywordEmbeddingProvider()),
        MemoryPolicy(namespace="project", scopes=[MemoryScope.AGENT], retrieval_limit=3),
    )
    await manager.remember(
        "alpha agent preference",
        scope=MemoryScope.AGENT,
        importance=0.9,
    )
    await manager.remember(
        "alpha session scratch",
        scope=MemoryScope.SESSION,
        source_session_id="other-session",
    )

    context = await manager.context_for("alpha query", session_id="session-1")

    assert "Relevant memory:" in context.content
    assert "alpha agent preference" in context.content
    assert "alpha session scratch" not in context.content
    assert context.memories[0].memory.scope == MemoryScope.AGENT
    [stored] = await storage.list_memories("project", scopes=[MemoryScope.AGENT])
    assert stored.last_used_at is not None


async def test_memory_manager_retrieves_session_scope_only_for_current_session(tmp_path):
    storage = await _storage(tmp_path)
    manager = MemoryManager(
        MemoryStore(storage, KeywordEmbeddingProvider()),
        MemoryPolicy(namespace="project", scopes=[MemoryScope.SESSION], retrieval_limit=3),
    )
    await manager.remember(
        "alpha current session note",
        scope=MemoryScope.SESSION,
        source_session_id="session-1",
    )
    await manager.remember(
        "alpha other session note",
        scope=MemoryScope.SESSION,
        source_session_id="session-2",
    )

    context = await manager.context_for("alpha query", session_id="session-1")

    assert "alpha current session note" in context.content
    assert "alpha other session note" not in context.content
    stored = await storage.list_memories("project", scopes=[MemoryScope.SESSION])
    by_text = {memory.text: memory for memory in stored}
    assert by_text["alpha current session note"].last_used_at is not None
    assert by_text["alpha other session note"].last_used_at is None


async def test_memory_manager_auto_capture_uses_extractor_when_enabled(tmp_path):
    storage = await _storage(tmp_path)
    manager = MemoryManager(
        MemoryStore(storage, HashEmbeddingProvider()),
        MemoryPolicy(namespace="project", auto_capture=True),
        extractor=StaticMemoryExtractor(),
    )

    records = await manager.capture(
        MemoryExchange(
            session_id="session-1",
            user_message="docker preference",
            assistant_message="noted",
            metadata={"source_turn_id": "turn-1"},
        )
    )

    assert len(records) == 1
    assert records[0].text == "remembered docker preference"
    assert records[0].source_session_id == "session-1"
    assert records[0].source_turn_id == "turn-1"
    assert records[0].scope == MemoryScope.SESSION
