from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable

from harness.config import HarnessSettings
from harness.plugins import ProviderRegistry


class SecretResolver(ABC):
    @abstractmethod
    async def resolve(self, names: list[str]) -> dict[str, str]:
        raise NotImplementedError


class EnvSecretResolver(SecretResolver):
    def __init__(self, prefix: str = "HARNESS_SECRET_") -> None:
        if not prefix:
            raise ValueError("EnvSecretResolver prefix must not be empty")
        self.prefix = prefix

    async def resolve(self, names: list[str]) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for name in names:
            env_name = f"{self.prefix}{_env_secret_name(name)}"
            value = os.environ.get(env_name)
            if value is None:
                raise KeyError(f"Missing secret: {name}")
            resolved[name] = value
        return resolved


SecretResolverFactory = Callable[[HarnessSettings], SecretResolver | None]


class SecretResolverRegistry(ProviderRegistry[SecretResolverFactory]):
    """Registry for credential resolver backends."""

    def create(self, name: str, settings: HarnessSettings) -> SecretResolver | None:
        try:
            factory = self.get(name)
        except KeyError as exc:
            raise ValueError(f"Unknown secret backend: {name}") from exc
        return factory(settings)


def default_secret_resolver_registry() -> SecretResolverRegistry:
    registry = SecretResolverRegistry()
    registry.register("none", lambda _settings: None)
    registry.register("env", lambda settings: EnvSecretResolver(prefix=settings.secret_env_prefix))
    return registry


def create_secret_resolver(
    settings: HarnessSettings,
    registry: SecretResolverRegistry | None = None,
) -> SecretResolver | None:
    resolver_registry = registry or default_secret_resolver_registry()
    return resolver_registry.create(settings.secret_backend, settings)


def _env_secret_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
