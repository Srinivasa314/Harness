from __future__ import annotations

import json
import sys

import pytest

from harness.schemas import ExecutionMode
from harness.tools import load_tool_registry


def test_load_subprocess_tool_registry(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "text.uppercase",
                        "description": "Uppercase text",
                        "required_capabilities": ["text:uppercase"],
                        "execution_mode": "subprocess",
                        "subprocess_command": [sys.executable, "tool.py"],
                    }
                ]
            }
        )
    )

    registry = load_tool_registry(path)

    assert registry.get("text.uppercase").execution_mode == ExecutionMode.SUBPROCESS


def test_loaded_tool_registry_requires_explicit_capability(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "text.uppercase",
                        "description": "Uppercase text",
                        "execution_mode": "subprocess",
                        "subprocess_command": [sys.executable, "tool.py"],
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="requires at least one capability"):
        load_tool_registry(path)


def test_tool_registry_rejects_malformed_json_schema(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "bad",
                        "description": "Bad schema",
                        "required_capabilities": ["tool:bad"],
                        "execution_mode": "subprocess",
                        "subprocess_command": [sys.executable, "tool.py"],
                        "input_schema": {"type": "definitely-not-a-json-schema-type"},
                    }
                ]
            }
        )
    )

    with pytest.raises(Exception, match="not valid under any"):
        load_tool_registry(path)


def test_in_process_tool_registry_requires_explicit_builtin(tmp_path):
    path = tmp_path / "tools.json"
    path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "unsafe",
                        "description": "Unsafe",
                        "required_capabilities": ["tool:unsafe"],
                        "execution_mode": "in_process",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="built-in function"):
        load_tool_registry(path)
