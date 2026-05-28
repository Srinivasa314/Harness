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
    session_options,
    session_summary_rows,
    summary_cards,
    tool_call_rows,
    tool_status_rows,
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
        sessions = await storage.list_sessions(limit=200)
        active_session_id = ""
        events = await storage.list_events(limit=500)
        turns = await _turns_for_session(active_session_id)
        tool_calls = await storage.list_tool_calls(limit=500)
        artifacts = await storage.list_artifacts(limit=500)

        ui.label("Agentic Harness").classes("text-2xl font-bold")
        ui.label("Observability dashboard").classes("text-sm text-gray-600")
        with ui.row().classes("items-end gap-4"):
            session_select = ui.select(
                session_options(sessions),
                value=active_session_id,
                label="Session",
            ).props("dense outlined").classes("min-w-[24rem]")

        with ui.row().classes("w-full gap-3"):
            for card in summary_cards(
                sessions=sessions,
                events=events,
                turns=turns,
                tool_calls=tool_calls,
                artifacts=artifacts,
            ):
                with ui.card().classes("min-w-[8rem] p-3"):
                    ui.label(card["label"]).classes("text-xs uppercase text-gray-500")
                    ui.label(card["value"]).classes("text-2xl font-semibold")

        with ui.tabs().classes("w-full") as tabs:
            overview_tab = ui.tab("Overview")
            events_tab = ui.tab("Events")
            turns_tab = ui.tab("Turns")
            tools_tab = ui.tab("Tool Calls")
            artifacts_tab = ui.tab("Artifacts")

        with ui.tab_panels(tabs, value=overview_tab).classes("w-full"):
            with ui.tab_panel(overview_tab):
                session_table = ui.table(
                    columns=[
                        {"name": "created_at", "label": "Created", "field": "created_at"},
                        {"name": "id", "label": "Session", "field": "id"},
                        {"name": "events", "label": "Events", "field": "events"},
                        {"name": "turns", "label": "Turns", "field": "turns"},
                        {"name": "tool_calls", "label": "Tool Calls", "field": "tool_calls"},
                        {"name": "tool_issues", "label": "Tool Issues", "field": "tool_issues"},
                        {"name": "artifacts", "label": "Artifacts", "field": "artifacts"},
                        {"name": "detail", "label": "Detail", "field": "detail"},
                    ],
                    rows=session_summary_rows(
                        sessions,
                        events=events,
                        turns=turns,
                        tool_calls=tool_calls,
                        artifacts=artifacts,
                    ),
                    row_key="id",
                ).classes("w-full")
                status_table = ui.table(
                    columns=[
                        {"name": "status", "label": "Status", "field": "status"},
                        {"name": "count", "label": "Count", "field": "count"},
                    ],
                    rows=tool_status_rows(tool_calls),
                    row_key="status",
                ).classes("w-full")
            with ui.tab_panel(events_tab):
                events_table = ui.table(
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
                turns_table = ui.table(
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
                tool_status = ui.select(
                    ["", "ok", "error", "denied", "timeout"],
                    value="",
                    label="Tool status",
                ).props("dense outlined").classes("min-w-[12rem]")
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
                artifacts_table = ui.table(
                    columns=[
                        {"name": "created_at", "label": "Created", "field": "created_at"},
                        {"name": "path", "label": "Path", "field": "path"},
                        {"name": "tool_call_id", "label": "Tool Call", "field": "tool_call_id"},
                        {"name": "size_bytes", "label": "Bytes", "field": "size_bytes"},
                    ],
                    rows=artifact_rows(artifacts),
                    row_key="id",
                ).classes("w-full")

        async def refresh_dashboard(_event: object = None) -> None:
            selected_session = session_select.value or None
            next_events = await storage.list_events(session_id=selected_session, limit=500)
            next_turns = await _turns_for_session(selected_session or "")
            next_tool_calls = await storage.list_tool_calls(session_id=selected_session, limit=500)
            next_artifacts = await storage.list_artifacts(session_id=selected_session, limit=500)
            events_table.rows = event_rows(next_events)
            turns_table.rows = turn_rows(next_turns)
            tools_table.rows = tool_call_rows(next_tool_calls, status=tool_status.value or None)
            artifacts_table.rows = artifact_rows(next_artifacts)
            if selected_session:
                shown_sessions = [session for session in sessions if session.id == selected_session]
            else:
                shown_sessions = sessions
            session_table.rows = session_summary_rows(
                shown_sessions,
                events=next_events,
                turns=next_turns,
                tool_calls=next_tool_calls,
                artifacts=next_artifacts,
            )
            status_table.rows = tool_status_rows(next_tool_calls)
            for table in (
                events_table,
                turns_table,
                tools_table,
                artifacts_table,
                session_table,
                status_table,
            ):
                table.update()

        tool_status.on_value_change(refresh_dashboard)
        session_select.on_value_change(refresh_dashboard)

    async def _turns_for_session(session_id: str) -> list:
        if session_id:
            return await storage.list_turns(session_id, limit=500)
        sessions = await storage.list_sessions(limit=50)
        turns = []
        for session in sessions:
            turns.extend(await storage.list_turns(session.id, limit=50))
        return turns[:500]

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
