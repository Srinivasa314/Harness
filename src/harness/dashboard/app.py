from __future__ import annotations

import argparse
import asyncio
from ipaddress import ip_address
from pathlib import Path

from nicegui import app, ui

from harness.config import load_settings
from harness.dashboard.viewmodels import (
    artifact_rows,
    event_rows,
    tool_call_rows,
    turn_rows,
)
from harness.storage import SQLiteStorage, create_storage


def build_app(db_path: str | Path | None = None) -> None:
    storage = SQLiteStorage(db_path) if db_path is not None else create_storage(load_settings())
    migrate_lock = asyncio.Lock()
    migrated = False
    app.on_shutdown(storage.close)

    async def ensure_migrated() -> None:
        nonlocal migrated
        if migrated:
            return
        async with migrate_lock:
            if not migrated:
                await storage.migrate()
                migrated = True

    @ui.page("/")
    async def index() -> None:
        await ensure_migrated()
        events = await storage.list_events(limit=50)
        recent_session_id = next((event.session_id for event in events if event.session_id), None)
        active_session_id = recent_session_id or ""
        turns = await storage.list_turns(active_session_id, limit=50) if active_session_id else []
        tool_calls = await storage.list_tool_calls(
            session_id=active_session_id or None,
            limit=50,
        )
        artifacts = await storage.list_artifacts(session_id=active_session_id or None, limit=50)

        ui.label("Agentic Harness").classes("text-2xl font-bold")
        with ui.row().classes("items-center"):
            ui.input("Session", value=active_session_id).props("dense readonly")
            tool_status = ui.select(
                ["", "ok", "error", "denied", "timeout"],
                value="",
                label="Tool status",
            ).props("dense")

        with ui.tabs().classes("w-full") as tabs:
            sessions_tab = ui.tab("Events")
            turns_tab = ui.tab("Turns")
            tools_tab = ui.tab("Tool Calls")
            artifacts_tab = ui.tab("Artifacts")

        with ui.tab_panels(tabs, value=sessions_tab).classes("w-full"):
            with ui.tab_panel(sessions_tab):
                ui.table(
                    columns=[
                        {"name": "ts", "label": "Time", "field": "ts"},
                        {"name": "event_type", "label": "Event", "field": "event_type"},
                        {"name": "session_id", "label": "Session", "field": "session_id"},
                        {"name": "session_detail", "label": "Detail", "field": "session_detail"},
                    ],
                    rows=event_rows(events),
                    row_key="ts",
                ).classes("w-full")
            with ui.tab_panel(turns_tab):
                ui.table(
                    columns=[
                        {"name": "created_at", "label": "Created", "field": "created_at"},
                        {"name": "session_id", "label": "Session", "field": "session_id"},
                        {"name": "role", "label": "Role", "field": "role"},
                        {"name": "content", "label": "Content", "field": "content"},
                    ],
                    rows=turn_rows(turns),
                    row_key="id",
                ).classes("w-full")
            with ui.tab_panel(tools_tab):
                tools_table = ui.table(
                    columns=[
                        {"name": "started_at", "label": "Started", "field": "started_at"},
                        {"name": "tool_name", "label": "Tool", "field": "tool_name"},
                        {"name": "status", "label": "Status", "field": "status"},
                        {"name": "error", "label": "Error", "field": "error"},
                    ],
                    rows=tool_call_rows(tool_calls, status=tool_status.value or None),
                    row_key="id",
                ).classes("w-full")
            with ui.tab_panel(artifacts_tab):
                ui.table(
                    columns=[
                        {"name": "created_at", "label": "Created", "field": "created_at"},
                        {"name": "path", "label": "Path", "field": "path"},
                        {"name": "tool_call_id", "label": "Tool Call", "field": "tool_call_id"},
                        {"name": "size_bytes", "label": "Bytes", "field": "size_bytes"},
                    ],
                    rows=artifact_rows(artifacts),
                    row_key="id",
                ).classes("w-full")
        def refresh_tool_filter(_event: object = None) -> None:
            tools_table.rows = tool_call_rows(tool_calls, status=tool_status.value or None)
            tools_table.update()

        tool_status.on_value_change(refresh_tool_filter)

    @ui.page("/sessions/{session_id}")
    async def session_detail(session_id: str) -> None:
        await ensure_migrated()
        session = await storage.get_session(session_id)
        turns = await storage.list_turns(session_id, limit=200)
        events = await storage.list_events(session_id=session_id, limit=200)
        tool_calls = await storage.list_tool_calls(session_id=session_id, limit=200)
        artifacts = await storage.list_artifacts(session_id=session_id, limit=200)

        ui.label("Session Detail").classes("text-2xl font-bold")
        if session is None:
            ui.label(f"Session not found: {session_id}")
            return
        ui.label(session.id).classes("font-mono")
        with ui.tabs().classes("w-full") as tabs:
            turns_tab = ui.tab("Turns")
            events_tab = ui.tab("Events")
            tools_tab = ui.tab("Tool Calls")
            artifacts_tab = ui.tab("Artifacts")
        with ui.tab_panels(tabs, value=turns_tab).classes("w-full"):
            with ui.tab_panel(turns_tab):
                ui.table(
                    columns=[
                        {"name": "created_at", "label": "Created", "field": "created_at"},
                        {"name": "role", "label": "Role", "field": "role"},
                        {"name": "content", "label": "Content", "field": "content"},
                    ],
                    rows=turn_rows(turns),
                    row_key="id",
                ).classes("w-full")
            with ui.tab_panel(events_tab):
                ui.table(
                    columns=[
                        {"name": "ts", "label": "Time", "field": "ts"},
                        {"name": "event_type", "label": "Event", "field": "event_type"},
                    ],
                    rows=event_rows(events),
                    row_key="ts",
                ).classes("w-full")
            with ui.tab_panel(tools_tab):
                ui.table(
                    columns=[
                        {"name": "started_at", "label": "Started", "field": "started_at"},
                        {"name": "tool_name", "label": "Tool", "field": "tool_name"},
                        {"name": "status", "label": "Status", "field": "status"},
                        {"name": "error", "label": "Error", "field": "error"},
                    ],
                    rows=tool_call_rows(tool_calls),
                    row_key="id",
                ).classes("w-full")
            with ui.tab_panel(artifacts_tab):
                ui.table(
                    columns=[
                        {"name": "created_at", "label": "Created", "field": "created_at"},
                        {"name": "path", "label": "Path", "field": "path"},
                        {"name": "size_bytes", "label": "Bytes", "field": "size_bytes"},
                    ],
                    rows=artifact_rows(artifacts),
                    row_key="id",
                ).classes("w-full")

def main() -> None:
    parser = argparse.ArgumentParser(prog="harness-dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--allow-remote-dashboard",
        action="store_true",
        help="Allow unauthenticated dashboard binding to a non-loopback host",
    )
    args = parser.parse_args()
    if not is_loopback_host(args.host) and not args.allow_remote_dashboard:
        raise SystemExit(
            "Refusing to bind unauthenticated dashboard to a non-loopback host; "
            "pass --allow-remote-dashboard to acknowledge the exposure."
        )
    build_app()
    ui.run(host=args.host, port=args.port, title="Agentic Harness", reload=False)


def is_loopback_host(host: str) -> bool:
    if host in {"localhost", "::1"}:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


if __name__ in {"__main__", "__mp_main__"}:
    main()
