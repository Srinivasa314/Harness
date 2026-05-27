from __future__ import annotations

from collections.abc import Callable

from harness.config import HarnessSettings
from harness.storage.base import StorageBackend
from harness.storage.postgres import PostgresStorage
from harness.storage.sqlite import SQLiteStorage

StorageFactory = Callable[[HarnessSettings], StorageBackend]


class StorageBackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, StorageFactory] = {}

    def register(self, name: str, factory: StorageFactory, *, replace: bool = False) -> None:
        if not replace and name in self._factories:
            raise ValueError(f"Storage backend already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str, settings: HarnessSettings) -> StorageBackend:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ValueError(f"Unknown storage backend: {name}") from exc
        return factory(settings)

    def names(self) -> list[str]:
        return sorted(self._factories)


def default_storage_registry() -> StorageBackendRegistry:
    registry = StorageBackendRegistry()
    registry.register("sqlite", lambda settings: SQLiteStorage(settings.sqlite_path))
    registry.register("postgres", _create_postgres_storage)
    return registry


def create_storage(
    settings: HarnessSettings,
    registry: StorageBackendRegistry | None = None,
) -> StorageBackend:
    return (registry or default_storage_registry()).create(settings.storage_backend, settings)


def _create_postgres_storage(settings: HarnessSettings) -> StorageBackend:
    if not settings.postgres_dsn:
        raise ValueError("HARNESS_POSTGRES_DSN is required for postgres storage")
    return PostgresStorage(settings.postgres_dsn)
