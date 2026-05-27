from __future__ import annotations

import hashlib
import importlib
from abc import ABC, abstractmethod
from typing import Any

import httpx
from anyio.to_thread import run_sync


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embedding provider for required tests."""

    provider_name = "hash"

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            index = digest[0] % self.dimensions
            values[index] += 1.0
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]


class MiniLMEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embedding provider using all-MiniLM-L6-v2."""

    provider_name = "minilm"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        normalize: bool = True,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await run_sync(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        encoded = model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        raw_vectors = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        return [[float(value) for value in vector] for vector in raw_vectors]

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            module = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MiniLMEmbeddingProvider requires sentence-transformers to be installed."
            ) from exc
        model_cls = module.SentenceTransformer
        if self.device is None:
            self._model = model_cls(self.model_name)
        else:
            self._model = model_cls(self.model_name, device=self.device)
        return self._model


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings API provider."""

    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dimensions: int | None = None,
        timeout_seconds: float = 120,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        self.client = client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        if self.client is not None:
            response = await self.client.post(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json=payload,
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=self._headers(),
                    json=payload,
                )
        response.raise_for_status()
        data = response.json()
        vectors_by_index: dict[int, list[float]] = {}
        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            embedding = item.get("embedding")
            if isinstance(index, int) and isinstance(embedding, list):
                vectors_by_index[index] = [float(value) for value in embedding]
        expected_indexes = set(range(len(texts)))
        if set(vectors_by_index) != expected_indexes:
            raise RuntimeError("OpenAI embeddings response did not include one vector per input")
        vectors = [vectors_by_index[index] for index in range(len(texts))]
        return vectors

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
