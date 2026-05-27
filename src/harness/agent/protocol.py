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
        return AgentAction(kind="final", content=str(payload["final"]))

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
            return AgentAction(kind="tool_calls", tool_calls=calls)

    return AgentAction(kind="final", content=response.content)


def _loads_first_json_object(content: str) -> Any:
    decoder = json.JSONDecoder()
    payload, _end = decoder.raw_decode(content)
    return payload
