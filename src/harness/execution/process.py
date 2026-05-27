from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LimitedCompletedProcess:
    returncode: int
    stdout: bytes
    stderr: bytes


class ProcessOutputLimitExceeded(RuntimeError):
    def __init__(self, *, stdout: bytes, stderr: bytes, limit_bytes: int) -> None:
        super().__init__(f"Process output exceeded {limit_bytes} bytes")
        self.stdout = stdout
        self.stderr = stderr
        self.limit_bytes = limit_bytes


async def run_limited_process(
    command: Sequence[str],
    *,
    input: bytes | None = None,  # noqa: A002 - mirrors subprocess APIs.
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout_seconds: float,
    max_output_bytes: int,
) -> LimitedCompletedProcess:
    if max_output_bytes < 1:
        raise ValueError("max_output_bytes must be at least 1")

    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if input is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env) if env is not None else None,
        cwd=str(cwd) if cwd is not None else None,
        start_new_session=True,
    )
    stdout = bytearray()
    stderr = bytearray()
    total_output_bytes = 0
    truncated = False

    async def feed_stdin() -> None:
        if input is None or process.stdin is None:
            return
        try:
            process.stdin.write(input)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            process.stdin.close()

    async def read_stream(stream: asyncio.StreamReader | None, output: bytearray) -> None:
        nonlocal total_output_bytes, truncated
        if stream is None:
            return
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            remaining = max_output_bytes - total_output_bytes
            if remaining > 0:
                output.extend(chunk[:remaining])
                total_output_bytes += min(len(chunk), remaining)
            if len(chunk) > remaining:
                truncated = True
                if process.returncode is None:
                    _kill_process_group(process)
                return

    tasks = [
        asyncio.create_task(feed_stdin()),
        asyncio.create_task(read_stream(process.stdout, stdout)),
        asyncio.create_task(read_stream(process.stderr, stderr)),
        asyncio.create_task(process.wait()),
    ]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_seconds)
    except TimeoutError:
        await _kill_and_wait(process)
        raise
    except BaseException:
        await _kill_and_wait(process)
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    if truncated:
        raise ProcessOutputLimitExceeded(
            stdout=bytes(stdout),
            stderr=bytes(stderr),
            limit_bytes=max_output_bytes,
        )
    return LimitedCompletedProcess(
        returncode=process.returncode or 0,
        stdout=bytes(stdout),
        stderr=bytes(stderr),
    )


async def _kill_and_wait(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        _kill_process_group(process)
    await asyncio.shield(process.wait())


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if process.pid is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()
