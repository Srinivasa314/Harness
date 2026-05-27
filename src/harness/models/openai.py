from __future__ import annotations

from typing import Any

import httpx

from harness.models.base import ModelMessage, ModelProvider, ModelResponse


class OpenAIResponsesProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 120,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = client

    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        payload = {
            "model": self.model,
            "input": [_to_response_input(message) for message in messages],
        }
        if self.client is not None:
            response = await self.client.post(
                f"{self.base_url}/responses",
                headers=self._headers(),
                json=payload,
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/responses",
                    headers=self._headers(),
                    json=payload,
                )
        response.raise_for_status()
        data = response.json()
        return ModelResponse(
            content=_extract_text(data),
            metadata={
                "provider": "openai",
                "model": self.model,
                "response_id": data.get("id"),
                "usage": data.get("usage"),
                "request_id": response.headers.get("x-request-id"),
            },
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def _extract_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


def _to_response_input(message: ModelMessage) -> dict[str, Any]:
    role = message.role
    content = message.content
    content_type = "input_text"
    if role == "tool":
        role = "user"
        content = f"Tool result:\n{content}"
    elif role == "assistant":
        content_type = "output_text"
    elif role not in {"user", "assistant", "system", "developer"}:
        role = "user"
        content = f"{message.role}:\n{content}"
    return {
        "role": role,
        "content": [{"type": content_type, "text": content}],
    }
