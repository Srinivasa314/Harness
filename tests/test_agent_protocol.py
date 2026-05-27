from __future__ import annotations

from harness.agent.protocol import parse_model_action
from harness.models import ModelResponse, ModelToolCall


def test_parse_model_action_treats_invalid_json_as_final():
    action = parse_model_action(ModelResponse(content="not json"))

    assert action.kind == "final"
    assert action.content == "not json"


def test_parse_model_action_treats_empty_content_as_empty_final():
    action = parse_model_action(ModelResponse(content="   "))

    assert action.kind == "final"
    assert action.content == ""


def test_parse_model_action_treats_non_dict_json_as_final():
    action = parse_model_action(ModelResponse(content='["not", "an", "object"]'))

    assert action.kind == "final"


def test_parse_model_action_ignores_invalid_tool_calls():
    action = parse_model_action(
        ModelResponse(content='{"tool_calls": [{"name": 123}, {"name": "x", "arguments": []}]}')
    )

    assert action.kind == "final"


def test_parse_model_action_accepts_tool_call_prefix_with_extra_text():
    action = parse_model_action(
        ModelResponse(
            content=(
                '{"tool_calls": [{"name": "text.uppercase", "arguments": {"text": "hello"}}]}'
                '{"final": "HELLO"}'
            )
        )
    )

    assert action.kind == "tool_calls"
    assert action.tool_calls[0].name == "text.uppercase"
    assert action.tool_calls[0].arguments == {"text": "hello"}


def test_parse_model_action_prefers_native_tool_calls():
    action = parse_model_action(
        ModelResponse(
            content='{"final": "ignored"}',
            tool_calls=[ModelToolCall(name="native", arguments={"ok": True})],
        )
    )

    assert action.kind == "tool_calls"
    assert action.tool_calls[0].name == "native"
