from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_deterministic_harness_settings(monkeypatch: pytest.MonkeyPatch, request) -> None:  # noqa: ANN001
    if any(
        request.node.get_closest_marker(marker)
        for marker in ("e2e", "container", "postgres", "provider")
    ):
        return
    for key in list(os.environ):
        if key.startswith("HARNESS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HARNESS_DISABLE_ENV_FILES", "1")
