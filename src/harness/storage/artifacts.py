from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from harness.schemas import ArtifactRecord
from harness.storage.base import StorageBackend
from harness.tools.redaction import redact_with_detected_secrets


class FileArtifactStore:
    def __init__(self, root: str | Path, storage: StorageBackend) -> None:
        self.root = Path(root).resolve()
        self.storage = storage

    async def save_bytes(
        self,
        data: bytes,
        *,
        filename: str,
        session_id: str | None = None,
        tool_call_id: str | None = None,
        media_type: str | None = None,
        metadata: dict | None = None,
    ) -> ArtifactRecord:
        directory = self.root / _safe_component(session_id or "global") / _safe_component(
            tool_call_id or "manual"
        )
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = _write_unique_bytes(directory, _safe_component(filename), data, root=self.root)
        artifact = ArtifactRecord(
            session_id=session_id,
            tool_call_id=tool_call_id,
            path=str(path),
            media_type=media_type,
            size_bytes=len(data),
            metadata=redact_with_detected_secrets(metadata or {}),
        )
        await self.storage.save_artifact(artifact)
        return artifact


def _safe_component(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.name != value or value in {"", ".", ".."}:
        raise ValueError(f"Unsafe artifact path component: {value!r}")
    return value


def _unique_path(directory: Path, filename: str) -> Path:
    path = (directory / filename).resolve()
    if not path.exists():
        return path
    source = Path(filename)
    return (directory / f"{source.stem}-{uuid4().hex[:12]}{source.suffix}").resolve()


def _write_unique_bytes(directory: Path, filename: str, data: bytes, *, root: Path) -> Path:
    for _ in range(100):
        path = _unique_path(directory, filename)
        if not path.is_relative_to(root):
            raise ValueError("Artifact path escapes artifact root")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as file:
                file.write(data)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate unique artifact filename for {filename!r}")
