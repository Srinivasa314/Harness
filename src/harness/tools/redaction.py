from __future__ import annotations

import re
from typing import Any

SECRET_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[a-z0-9._\-]{12,}"),
    re.compile(r"(?i)(token=)[^&\s]+"),
    re.compile(r"(?i)(api[_-]?key[\"'\s:=]+)[^\"'\s,}]+"),
    re.compile(
        r"(?i)((?:clientsecret|access[_-]?token|refresh[_-]?token|private[_-]?key)"
        r"[\"'\s:=]+)[^\"'\s,}]+"
    ),
    re.compile(r"\bsk-[a-zA-Z0-9_-]{8,}\b"),
]
SENSITIVE_KEYWORDS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


def redact(value: Any, secrets: dict[str, str] | None = None) -> Any:
    secrets = secrets or {}
    if isinstance(value, str):
        return _redact_string(value, secrets)
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _is_sensitive_key(key) else redact(item, secrets)
            for key, item in value.items()
        }
    return value


def redact_with_detected_secrets(value: Any) -> Any:
    return redact(value, sensitive_values(value))


def sensitive_values(value: Any) -> dict[str, str]:
    values: dict[str, str] = {}

    def collect(item: Any, path: str = "secret") -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                nested_path = f"{path}.{key}"
                if _is_sensitive_key(key):
                    if isinstance(nested, str):
                        values[nested_path] = nested
                    elif isinstance(nested, list | tuple):
                        for index, nested_item in enumerate(nested):
                            if isinstance(nested_item, str):
                                values[f"{nested_path}.{index}"] = nested_item
                    elif nested is not None:
                        values[nested_path] = str(nested)
                collect(nested, nested_path)
        elif isinstance(item, list | tuple):
            for index, nested in enumerate(item):
                collect(nested, f"{path}.{index}")

    collect(value)
    return values


def _redact_string(value: str, secrets: dict[str, str]) -> str:
    redacted = value
    for secret in secrets.values():
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    for pattern in SECRET_PATTERNS:
        replacement = r"\1[REDACTED]" if pattern.groups else "[REDACTED]"
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    parts = [part for part in normalized.split("_") if part]
    compact = normalized.replace("_", "")
    if normalized in SENSITIVE_KEYWORDS:
        return True
    if "apikey" in compact:
        return True
    if compact in {"privatekey", "clientsecret", "accesstoken", "refreshtoken"}:
        return True
    if compact.endswith(("secret", "token", "password", "credential", "privatekey")):
        return True
    if any(keyword in parts for keyword in {"authorization", "credential", "password", "token"}):
        return True
    return (
        normalized in {"secret", "secret_key"}
        or normalized.endswith("_secret")
        or normalized.endswith("_secret_key")
    )
