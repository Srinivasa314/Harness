from __future__ import annotations

import tempfile
from pathlib import Path

import anyio

from harness.models.base import ModelMessage, ModelProvider, ModelResponse
from harness.process_env import DEFAULT_ENV_ALLOWLIST, filtered_env

CODEX_ENV_ALLOWLIST = (*DEFAULT_ENV_ALLOWLIST, "CODEX_HOME")
DEFAULT_CODEX_COMMAND = ["codex", "exec"]


class CodexCliProvider(ModelProvider):
    """Provider for local Codex CLI subscription/API-key auth.

    The provider shells out to `codex exec` and relies on the local Codex CLI
    authentication state. It does not manage Codex credentials.
    """

    def __init__(
        self,
        command: list[str] | None = None,
        model: str | None = None,
        timeout_seconds: float = 1800,
        use_safe_defaults: bool | None = None,
    ) -> None:
        self.command = command or DEFAULT_CODEX_COMMAND
        self.use_safe_defaults = (
            _is_codex_exec_command(self.command) if use_safe_defaults is None else use_safe_defaults
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.env = filtered_env(inherit=CODEX_ENV_ALLOWLIST)

    async def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        prompt = "\n\n".join(f"{message.role}: {message.content}" for message in messages)
        command = [*self.command]
        if self.model:
            command.extend(["--model", self.model])
        if self.use_safe_defaults:
            command.extend(
                [
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--ignore-rules",
                    "--ephemeral",
                ]
            )
        with tempfile.TemporaryDirectory(prefix="harness-codex-") as cwd:
            with anyio.fail_after(self.timeout_seconds):
                completed = await anyio.run_process(
                    [*command, "-"],
                    input=prompt.encode(),
                    check=False,
                    env=self.env,
                    cwd=cwd if self.use_safe_defaults else None,
                )
        stdout = completed.stdout.decode(errors="replace")
        stderr = completed.stderr.decode(errors="replace")
        if completed.returncode != 0:
            raise RuntimeError(stderr or f"Codex exited with {completed.returncode}")
        return ModelResponse(
            content=stdout,
            metadata={
                "provider": "codex_cli",
                "command": command,
                "model": self.model,
                "safe_defaults": self.use_safe_defaults,
            },
        )


def _is_codex_exec_command(command: list[str]) -> bool:
    return len(command) == 2 and Path(command[0]).name == "codex" and command[1] == "exec"
