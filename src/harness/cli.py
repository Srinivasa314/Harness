from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from harness.agent import AgentSessionManager
from harness.config import load_settings
from harness.dashboard.app import build_app, is_loopback_host
from harness.doctor import doctor_summary, run_doctor_checks
from harness.models import ModelProvider
from harness.runtime import build_runtime_async
from harness.schemas import MemoryScope
from harness.storage import create_storage
from harness.tools.redaction import redact


async def migrate() -> None:
    storage = create_storage(load_settings())
    try:
        await storage.migrate()
    finally:
        await storage.close()


async def export_session(session_id: str, output: Path | None = None) -> int:
    settings = load_settings()
    storage = create_storage(settings)
    try:
        await storage.migrate()
        session = await storage.get_session(session_id)
        memories = await storage.list_memories(settings.memory_namespace)
        payload = {
            "session": redact(session.model_dump(mode="json")) if session else None,
            "turns": [
                redact(turn.model_dump(mode="json"))
                for turn in await storage.list_turns(session_id, limit=None)
            ],
            "events": [
                redact(event.model_dump(mode="json"))
                for event in await storage.list_events(session_id=session_id, limit=None)
            ],
            "tool_calls": [
                redact(call.model_dump(mode="json"))
                for call in await storage.list_tool_calls(session_id=session_id, limit=None)
            ],
            "artifacts": [
                redact(artifact.model_dump(mode="json"))
                for artifact in await storage.list_artifacts(session_id=session_id, limit=None)
            ],
            "memories": [
                redact(memory.model_dump(mode="json"))
                for memory in memories
                if memory.scope != MemoryScope.SESSION or memory.source_session_id == session_id
            ],
        }
    finally:
        await storage.close()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        print(text)
    else:
        output.write_text(text)
    return 0


async def run_agent(
    prompt: str,
    output: Path | None = None,
    tools_path: Path | None = None,
    capabilities: list[str] | None = None,
    model: ModelProvider | None = None,
) -> int:
    settings = load_settings()
    if capabilities is not None:
        settings.tool_capabilities = capabilities
    runtime = await build_runtime_async(settings=settings, tools_path=tools_path, model=model)
    try:
        await runtime.storage.migrate()
        session = await AgentSessionManager(runtime.storage).create({"source": "cli"})
        loop = runtime.agent_loop()
        result = await loop.run(session.id, prompt)
    finally:
        await runtime.close()
    payload = result.model_dump(mode="json")
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        print(text)
    else:
        output.write_text(text)
    return 0


def doctor(output: Path | None = None) -> int:
    payload = doctor_summary(run_doctor_checks())
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        print(text)
    else:
        output.write_text(text)
    return 0 if payload["ok"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("migrate", help="Create or update harness storage tables")
    dashboard = subcommands.add_parser("dashboard", help="Run the NiceGUI dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8080)
    dashboard.add_argument(
        "--allow-remote-dashboard",
        action="store_true",
        help="Allow unauthenticated dashboard binding to a non-loopback host",
    )
    export = subcommands.add_parser("export-session", help="Export session observability data")
    export.add_argument("session_id")
    export.add_argument("--output", type=Path)
    agent_run = subcommands.add_parser("run-agent", help="Run the single-agent loop")
    agent_run.add_argument("prompt")
    agent_run.add_argument("--output", type=Path)
    agent_run.add_argument("--tools", type=Path, help="JSON tool registry file")
    agent_run.add_argument(
        "--capability",
        action="append",
        dest="capabilities",
        help="Grant a tool capability to this run; repeat for multiple grants",
    )
    doctor_parser = subcommands.add_parser("doctor", help="Report local harness readiness")
    doctor_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "migrate":
        asyncio.run(migrate())
        return

    if args.command == "dashboard":
        from nicegui import ui

        if not is_loopback_host(args.host) and not args.allow_remote_dashboard:
            raise SystemExit(
                "Refusing to bind unauthenticated dashboard to a non-loopback host; "
                "pass --allow-remote-dashboard to acknowledge the exposure."
            )
        build_app()
        ui.run(host=args.host, port=args.port, title="Agentic Harness", reload=False)
        return

    if args.command == "export-session":
        raise SystemExit(asyncio.run(export_session(args.session_id, args.output)))

    if args.command == "run-agent":
        raise SystemExit(
            asyncio.run(
                run_agent(
                    args.prompt,
                    args.output,
                    args.tools,
                    args.capabilities,
                )
            )
        )

    if args.command == "doctor":
        raise SystemExit(doctor(args.output))

    raise SystemExit(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()
