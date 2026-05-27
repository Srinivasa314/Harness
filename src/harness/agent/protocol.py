from __future__ import annotations

import json
from typing import Any

from harness.models import AgentAction, ModelResponse, ModelToolCall


def parse_model_action(response: ModelResponse) -> AgentAction:
    if response.tool_calls:
        return AgentAction(kind="tool_calls", tool_calls=response.tool_calls)

    content = response.content.strip()
    if not content:
        return AgentAction(kind="final", content="")

    try:
        payload = _loads_first_json_object(content)
    except json.JSONDecodeError:
        return AgentAction(kind="final", content=response.content)

    if not isinstance(payload, dict):
        return AgentAction(kind="final", content=response.content)

    if "final" in payload:
        final = payload["final"]
        if isinstance(final, str):
            nested_tool_calls = _tool_calls_from_content(final)
            if nested_tool_calls:
                return AgentAction(kind="tool_calls", tool_calls=nested_tool_calls)
        return AgentAction(kind="final", content=str(final))

    calls = _tool_calls_from_payload(payload)
    if calls:
        return AgentAction(kind="tool_calls", tool_calls=calls)

    return AgentAction(kind="final", content=response.content)


def _tool_calls_from_content(content: str) -> list[ModelToolCall]:
    try:
        payload = _loads_first_json_object(content.strip())
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    return _tool_calls_from_payload(payload)


def _tool_calls_from_payload(payload: dict) -> list[ModelToolCall]:
    raw_calls = payload.get("tool_calls")
    if isinstance(raw_calls, list):
        calls = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            name = raw_call.get("name")
            arguments = raw_call.get("arguments", {})
            if isinstance(name, str) and isinstance(arguments, dict):
                calls.append(ModelToolCall(name=name, arguments=arguments))
        if calls:
            return calls
    return []


def _loads_first_json_object(content: str) -> Any:
    decoder = json.JSONDecoder()
    payload, _end = decoder.raw_decode(content)
    return payload
