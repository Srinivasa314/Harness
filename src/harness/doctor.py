from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from importlib.util import find_spec

from harness.config import HarnessSettings, load_settings
from harness.memory import default_embedding_provider_registry
from harness.models import default_model_provider_registry
from harness.storage import default_storage_registry
from harness.tools import default_secret_resolver_registry


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run_doctor_checks(settings: HarnessSettings | None = None) -> list[DoctorCheck]:
    settings = settings or load_settings()
    checks = [
        _command_check("uv", "uv"),
        DoctorCheck(name="python", ok=True, detail=sys.executable),
        _command_check("docker", settings.docker_bin, required=False),
    ]
    checks.extend(_registry_checks(settings))
    if settings.memory_enabled and settings.embedding_provider == "minilm":
        minilm_available = find_spec("sentence_transformers") is not None
        checks.append(
            DoctorCheck(
                name="embedding_dependencies",
                ok=minilm_available,
                detail=(
                    "sentence-transformers is installed"
                    if minilm_available
                    else "Install with: uv sync --extra embeddings"
                ),
            )
        )
    if settings.memory_enabled and settings.embedding_provider == "openai":
        checks.append(
            _settings_check(
                "openai_api_key",
                bool(settings.openai_api_key),
                "HARNESS_OPENAI_API_KEY",
            )
        )
    if settings.storage_backend == "postgres":
        checks.append(
            _settings_check("postgres_dsn", bool(settings.postgres_dsn), "HARNESS_POSTGRES_DSN")
        )
    if settings.model_provider == "openai" and settings.embedding_provider != "openai":
        checks.append(
            _settings_check(
                "openai_api_key",
                bool(settings.openai_api_key),
                "HARNESS_OPENAI_API_KEY",
            )
        )
    elif settings.embedding_provider != "openai":
        checks.append(
            _settings_check(
                "openai_api_key",
                bool(settings.openai_api_key),
                "HARNESS_OPENAI_API_KEY",
                required=False,
            )
        )
    codex_command = settings.codex_command[0] if settings.codex_command else ""
    if settings.model_provider == "codex":
        if codex_command:
            checks.append(_command_check("codex", codex_command))
        else:
            checks.append(
                DoctorCheck(
                    name="codex",
                    ok=False,
                    detail="HARNESS_CODEX_COMMAND must not be empty",
                )
            )
    else:
        if codex_command:
            checks.append(_command_check("codex", codex_command, required=False))
        else:
            checks.append(
                DoctorCheck(
                    name="codex",
                    ok=False,
                    detail="HARNESS_CODEX_COMMAND is empty",
                    required=False,
                )
            )
    return checks


def doctor_summary(checks: list[DoctorCheck]) -> dict:
    return {
        "ok": all(check.ok for check in checks if check.required),
        "checks": [
            {
                "name": check.name,
                "ok": check.ok,
                "detail": check.detail,
                "required": check.required,
            }
            for check in checks
        ],
    }


def _command_check(name: str, command: str, *, required: bool = True) -> DoctorCheck:
    path = shutil.which(command)
    return DoctorCheck(
        name=name,
        ok=path is not None,
        detail=path or f"{command} not found on PATH",
        required=required,
    )


def _settings_check(
    name: str,
    ok: bool,
    setting_name: str,
    *,
    required: bool = True,
) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        ok=ok,
        detail=f"{setting_name} is set" if ok else f"{setting_name} is not set",
        required=required,
    )


def _registry_checks(settings: HarnessSettings) -> list[DoctorCheck]:
    checks = [
        _registry_check(
            "storage_backend",
            settings.storage_backend,
            default_storage_registry().names(),
        ),
        _registry_check(
            "model_provider",
            settings.model_provider,
            default_model_provider_registry().names(),
        ),
        _registry_check(
            "secret_backend",
            settings.secret_backend,
            default_secret_resolver_registry().names(),
        ),
    ]
    if settings.memory_enabled:
        checks.append(
            _registry_check(
                "embedding_provider",
                settings.embedding_provider,
                default_embedding_provider_registry().names(),
            )
        )
    return checks


def _registry_check(name: str, value: str, allowed: list[str]) -> DoctorCheck:
    ok = value in allowed
    return DoctorCheck(
        name=name,
        ok=ok,
        detail=f"{value} is configured" if ok else f"Unknown {name}: {value}",
    )
