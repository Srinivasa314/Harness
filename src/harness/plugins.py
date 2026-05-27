from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class ProviderRegistry(Generic[T]):
    """Small named registry for code-level plugin points."""

    def __init__(self, providers: dict[str, T] | None = None) -> None:
        self._providers: dict[str, T] = dict(providers or {})

    def register(self, name: str, provider: T, *, replace: bool = False) -> None:
        if not replace and name in self._providers:
            raise ValueError(f"Provider already registered: {name}")
        self._providers[name] = provider

    def get(self, name: str) -> T:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._providers)

    def items(self) -> Iterable[tuple[str, T]]:
        return self._providers.items()
