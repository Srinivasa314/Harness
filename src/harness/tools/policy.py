from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CapabilityGrant:
    capabilities: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def all(cls) -> CapabilityGrant:
        return cls(frozenset({"*"}))

    def allows(self, required: list[str]) -> bool:
        return "*" in self.capabilities or set(required).issubset(self.capabilities)


class CapabilityPolicy:
    def __init__(self, grant: CapabilityGrant | None = None) -> None:
        self.grant = grant or CapabilityGrant()

    def check(self, required: list[str]) -> bool:
        return self.grant.allows(required)
