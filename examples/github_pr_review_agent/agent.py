from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, NamedTuple

import aiosqlite
import anyio
import httpx
import jwt

from harness.agent import ContextCompactionPolicy, RollingSummaryContextCompactor
from harness.config import HarnessSettings
from harness.execution import (
    ContainerSchemaRegistry,
    DockerContainerExecutor,
    InProcessExecutor,
)
from harness.memory import MemoryCandidate, MemoryExchange, MemoryExtractor
from harness.runtime import build_runtime_async
from harness.schemas import (
    ContainerSchema,
    ExecutionMode,
    MemoryScope,
    Session,
    ToolDefinition,
    utc_now,
)
from harness.storage import SQLiteStorage
from harness.tools import CapabilityGrant, CapabilityPolicy, EnvSecretResolver, ToolRegistry

ROOT = Path(__file__).resolve().parents[2]
DEMO_DB = ROOT / "data" / "github_pr_review_agent.sqlite3"
HF_CACHE = ROOT / "data" / "hf-cache"
ST_CACHE = ROOT / "data" / "sentence-transformers"
GITHUB_API = "https://api.github.com"
AGENT_COMMENT_MARKER = "<!-- harness-pr-review-agent -->"
GITHUB_PAGE_SIZE = 100
MAX_GITHUB_PAGES = 5
MAX_CHANGED_FILES_IN_CONTEXT = 80
MAX_CHECK_RUNS_IN_CONTEXT = 50
PREFERENCE_MARKERS = (
    "always",
    "avoid",
    "don't",
    "do not",
    "focus",
    "include",
    "never",
    "prefer",
    "prioritize",
    "skip",
)


class PullRequestReply(NamedTuple):
    comment_id: int
    author: str
    body: str

BASH_RUNNER = r"""
import json
import os
import subprocess
import sys
from pathlib import Path

payload = json.load(sys.stdin)
arguments = payload.get("arguments", {})
command = str(arguments["command"])
workdir = Path(os.environ.get("HARNESS_REPO_MOUNT", "/repo"))
if not workdir.exists():
    workdir = Path.cwd()
try:
    completed = subprocess.run(
        ["/bin/sh", "-lc", command],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=25,
    )
except subprocess.TimeoutExpired as exc:
    print(json.dumps({
        "command": command,
        "returncode": 124,
        "stdout": (exc.stdout or "")[-20000:],
        "stderr": ((exc.stderr or "") + "\nCommand timed out after 25 seconds.")[-12000:],
    }))
    raise SystemExit(0)
print(json.dumps({
    "command": command,
    "returncode": completed.returncode,
    "stdout": completed.stdout[-20000:],
    "stderr": completed.stderr[-12000:],
}))
"""


def _user_replies_from_message(message: str) -> list[PullRequestReply]:
    marker = "GitHub user replies:"
    if marker not in message:
        return []
    replies: list[PullRequestReply] = []
    current_id: int | None = None
    current_author = ""
    current_body: list[str] = []
    for raw_line in message.split(marker, 1)[1].splitlines():
        line = raw_line.strip()
        if line.startswith("Comment ") and " by " in line and line.endswith(":"):
            if current_id is not None:
                replies.append(
                    PullRequestReply(
                        comment_id=current_id,
                        author=current_author,
                        body="\n".join(current_body).strip(),
                    )
                )
            header = line.removesuffix(":")
            comment_part, author = header.split(" by ", 1)
            try:
                current_id = int(comment_part.removeprefix("Comment ").strip())
            except ValueError:
                current_id = None
            current_author = author.strip()
            current_body = []
            continue
        if current_id is not None:
            current_body.append(raw_line)
    if current_id is not None:
        replies.append(
            PullRequestReply(
                comment_id=current_id,
                author=current_author,
                body="\n".join(current_body).strip(),
            )
        )
    return [reply for reply in replies if reply.body]


def _preference_memories(reply: PullRequestReply) -> list[MemoryCandidate]:
    memories: list[MemoryCandidate] = []
    for raw_line in reply.body.splitlines():
        line = raw_line.strip(" -\t")
        if not line:
            continue
        lowered = line.lower()
        if not any(marker in lowered for marker in PREFERENCE_MARKERS):
            continue
        memories.append(
            MemoryCandidate(
                text=f"For GitHub PR reviews, user preference: {line}",
                scope=MemoryScope.AGENT,
                metadata={
                    "source": "github_pr_comment",
                    "github_comment_id": reply.comment_id,
                    "github_comment_author": reply.author,
                },
            )
        )
    if memories:
        return memories
    lowered_reply = reply.body.lower()
    if any(marker in lowered_reply for marker in PREFERENCE_MARKERS):
        return [
            MemoryCandidate(
                text=f"For GitHub PR reviews, user preference: {reply.body.strip()}",
                scope=MemoryScope.AGENT,
                metadata={
                    "source": "github_pr_comment",
                    "github_comment_id": reply.comment_id,
                    "github_comment_author": reply.author,
                },
            )
        ]
    return []


class ReviewPreferenceMemoryExtractor(MemoryExtractor):
    async def extract(self, exchange: MemoryExchange) -> list[MemoryCandidate]:
        replies = _replies_from_tool_outputs(exchange.tool_outputs)
        if not replies:
            replies = _user_replies_from_message(exchange.user_message)
        if replies:
            return [
                memory
                for reply in replies
                for memory in _preference_memories(reply)
            ]
        _ = exchange
        return []


def _replies_from_tool_outputs(tool_outputs: list[dict[str, Any]]) -> list[PullRequestReply]:
    replies: list[PullRequestReply] = []
    for output in tool_outputs:
        if output.get("name") != "github.pr_replies":
            continue
        payload = output.get("output")
        if not isinstance(payload, dict):
            continue
        raw_replies = payload.get("replies")
        if not isinstance(raw_replies, list):
            continue
        for raw_reply in raw_replies:
            if not isinstance(raw_reply, dict):
                continue
            comment_id = raw_reply.get("comment_id")
            author = raw_reply.get("author")
            body = raw_reply.get("body")
            if isinstance(comment_id, int) and isinstance(author, str) and isinstance(body, str):
                replies.append(PullRequestReply(comment_id, author, body))
    return replies


def _replies_from_tool_results(results: list[Any]) -> list[PullRequestReply]:
    tool_outputs = [
        result.model_dump(mode="json")
        for result in results
        if getattr(result, "name", None) == "github.pr_replies"
        and getattr(result, "status", None) == "ok"
    ]
    return _replies_from_tool_outputs(tool_outputs)


async def github_pr_context(arguments: dict[str, Any], secrets: dict[str, str]) -> dict[str, Any]:
    repo = str(arguments["repo"])
    pr_number = int(arguments["pr_number"])
    token = await github_app_installation_token(secrets)
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        pr_response = await client.get(
            f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}",
            headers=headers,
        )
        pr_response.raise_for_status()
        pr = pr_response.json()
        files = await _get_paginated_list(
            client,
            f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files",
            headers=headers,
            list_key=None,
        )
        checks = await _get_paginated_list(
            client,
            f"{GITHUB_API}/repos/{repo}/commits/{pr['head']['sha']}/check-runs",
            headers=headers,
            list_key="check_runs",
            tolerate_statuses={403, 404},
        )
    return {
        "repo": repo,
        "number": pr_number,
        "title": pr["title"],
        "state": pr["state"],
        "url": pr["html_url"],
        "author": pr["user"]["login"],
        "base": pr["base"]["ref"],
        "head": pr["head"]["ref"],
        "body_excerpt": (pr.get("body") or "")[:2_000],
        "changed_files_total": len(files),
        "changed_files_truncated": len(files) > MAX_CHANGED_FILES_IN_CONTEXT,
        "changed_files": [
            {
                "filename": item["filename"],
                "status": item["status"],
                "additions": item["additions"],
                "deletions": item["deletions"],
                "patch_excerpt": (item.get("patch") or "")[:1_500],
            }
            for item in files[:MAX_CHANGED_FILES_IN_CONTEXT]
        ],
        "check_runs_total": len(checks),
        "check_runs_truncated": len(checks) > MAX_CHECK_RUNS_IN_CONTEXT,
        "check_runs": [
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "conclusion": item.get("conclusion"),
            }
            for item in checks[:MAX_CHECK_RUNS_IN_CONTEXT]
        ],
    }


async def github_pr_comment(arguments: dict[str, Any], secrets: dict[str, str]) -> dict[str, str]:
    repo = str(arguments["repo"])
    pr_number = int(arguments["pr_number"])
    body = str(arguments["body"])
    token = await github_app_installation_token(secrets)
    url = await post_pr_comment(
        repo=repo,
        pr_number=pr_number,
        token=token,
        review=body,
    )
    return {"url": url}


async def github_app_installation_token(secrets: dict[str, str]) -> str:
    app_id = os.environ["HARNESS_GITHUB_APP_ID"]
    installation_id = os.environ["HARNESS_GITHUB_INSTALLATION_ID"]
    private_key = _normalize_private_key(secrets["github_app_private_key"])
    now = int(time.time())
    app_jwt = jwt.encode(
        {
            "iat": now - 60,
            "exp": now + 540,
            "iss": app_id,
        },
        private_key,
        algorithm="RS256",
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {app_jwt}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{GITHUB_API}/app/installations/{installation_id}/access_tokens",
            headers=headers,
        )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("GitHub App installation token response did not include a token.")
    return token


async def fetch_pr_user_replies(*, repo: str, pr_number: int, token: str) -> list[PullRequestReply]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    comments: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        comments = await _get_paginated_list(
            client,
            f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments",
            headers=headers,
            list_key=None,
        )
    last_agent_index = -1
    for index, comment in enumerate(comments):
        body = comment.get("body")
        if isinstance(body, str) and AGENT_COMMENT_MARKER in body:
            last_agent_index = index
    replies: list[PullRequestReply] = []
    for comment in comments[last_agent_index + 1:]:
        body = comment.get("body")
        user = comment.get("user")
        comment_id = comment.get("id")
        if not isinstance(body, str) or AGENT_COMMENT_MARKER in body:
            continue
        if not isinstance(user, dict) or not isinstance(user.get("login"), str):
            continue
        if not isinstance(comment_id, int):
            continue
        replies.append(
            PullRequestReply(
                comment_id=comment_id,
                author=user["login"],
                body=body.strip(),
            )
        )
    return replies


def _reply_payload(reply: PullRequestReply) -> dict[str, Any]:
    return {
        "comment_id": reply.comment_id,
        "author": reply.author,
        "body": reply.body,
    }


async def _get_paginated_list(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    list_key: str | None,
    tolerate_statuses: set[int] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    tolerated = tolerate_statuses or set()
    for page in range(1, MAX_GITHUB_PAGES + 1):
        response = await client.get(
            url,
            headers=headers,
            params={"per_page": GITHUB_PAGE_SIZE, "page": page},
        )
        if response.status_code in tolerated:
            return items
        response.raise_for_status()
        payload = response.json()
        raw_items = payload.get(list_key, []) if list_key else payload
        if not isinstance(raw_items, list):
            return items
        items.extend(item for item in raw_items if isinstance(item, dict))
        if len(raw_items) < GITHUB_PAGE_SIZE:
            break
    return items


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="github.pr_context",
            description="Fetch real GitHub pull request metadata, changed files, and check runs.",
            execution_mode=ExecutionMode.IN_PROCESS,
            required_capabilities=["github:pr"],
            required_secrets=["github_app_private_key"],
            input_schema={
                "type": "object",
                "required": ["repo", "pr_number"],
                "properties": {
                    "repo": {"type": "string"},
                    "pr_number": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "required": [
                    "repo",
                    "number",
                    "title",
                    "state",
                    "url",
                    "changed_files",
                    "check_runs",
                ],
                "properties": {
                    "repo": {"type": "string"},
                    "number": {"type": "integer"},
                    "title": {"type": "string"},
                    "state": {"type": "string"},
                    "url": {"type": "string"},
                    "changed_files_total": {"type": "integer"},
                    "changed_files_truncated": {"type": "boolean"},
                    "changed_files": {"type": "array"},
                    "check_runs_total": {"type": "integer"},
                    "check_runs_truncated": {"type": "boolean"},
                    "check_runs": {"type": "array"},
                },
                "additionalProperties": True,
            },
        ),
        github_pr_context,
    )
    registry.register(
        ToolDefinition(
            name="github.pr_comment",
            description="Post the final review as a GitHub pull request comment.",
            execution_mode=ExecutionMode.IN_PROCESS,
            required_capabilities=["github:comment"],
            required_secrets=["github_app_private_key"],
            input_schema={
                "type": "object",
                "required": ["repo", "pr_number", "body"],
                "properties": {
                    "repo": {"type": "string"},
                    "pr_number": {"type": "integer"},
                    "body": {"type": "string"},
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "required": ["url"],
                "properties": {"url": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        github_pr_comment,
    )
    registry.register(
        ToolDefinition(
            name="repo.bash",
            description=(
                "Run a read-only POSIX shell command in the local checkout mounted at /repo. "
                "Use this to inspect files, configuration, tests, and code quality signals. "
                "Available commands include sh, find, grep, sed, head, cat, and python; "
                "do not assume git, rg, package managers, or network tools are installed."
            ),
            execution_mode=ExecutionMode.CONTAINER,
            container_schema="python-analysis",
            container_command=["python", "-c", BASH_RUNNER],
            required_capabilities=["repo:sandbox"],
            timeout_seconds=30,
            max_output_bytes=200_000,
            input_schema={
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run with /repo as the working directory.",
                    },
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "required": ["command", "returncode", "stdout", "stderr"],
                "properties": {
                    "command": {"type": "string"},
                    "returncode": {"type": "integer"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                },
                "additionalProperties": False,
            },
        )
    )
    return registry


def register_run_tools(
    registry: ToolRegistry,
    *,
    storage: SQLiteStorage,
) -> None:
    async def github_pr_replies(
        arguments: dict[str, Any],
        secrets: dict[str, str],
    ) -> dict[str, Any]:
        repo = str(arguments["repo"])
        pr_number = int(arguments["pr_number"])
        token = await github_app_installation_token(secrets)
        replies = await _unprocessed_replies(
            storage,
            repo=repo,
            pr_number=pr_number,
            token=token,
        )
        return {"replies": [_reply_payload(reply) for reply in replies]}

    registry.register(
        ToolDefinition(
            name="github.pr_replies",
            description=(
                "Read unprocessed user replies on the GitHub PR after the latest Harness "
                "agent comment. Use this before deciding whether to process review "
                "preferences or run a fresh review."
            ),
            execution_mode=ExecutionMode.IN_PROCESS,
            required_capabilities=["github:replies"],
            required_secrets=["github_app_private_key"],
            input_schema={
                "type": "object",
                "required": ["repo", "pr_number"],
                "properties": {
                    "repo": {"type": "string"},
                    "pr_number": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "required": ["replies"],
                "properties": {
                    "replies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["comment_id", "author", "body"],
                            "properties": {
                                "comment_id": {"type": "integer"},
                                "author": {"type": "string"},
                                "body": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    }
                },
                "additionalProperties": False,
            },
        ),
        github_pr_replies,
    )


async def main() -> None:
    _load_local_env_file(ROOT / ".env.development")
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ST_CACHE))
    _reset_demo_database_if_requested()

    repo = os.environ.get("HARNESS_GITHUB_REPO", "")
    pr_number = int(os.environ.get("HARNESS_GITHUB_PR", "0") or "0")
    repo_path = _target_repo_path()
    tool_capabilities = ["github:pr", "github:replies", "github:comment", "repo:sandbox"]
    settings = HarnessSettings(
        storage_backend="sqlite",
        sqlite_path=DEMO_DB,
        model_provider=os.environ.get("HARNESS_MODEL_PROVIDER", "openai"),
        openai_model=os.environ.get("HARNESS_OPENAI_MODEL", "gpt-5.2"),
        embedding_provider=os.environ.get("HARNESS_EMBEDDING_PROVIDER", "minilm"),
        memory_enabled=True,
        memory_namespace="github-pr-review-demo",
        memory_auto_capture=True,
        memory_retrieval_limit=4,
        memory_max_context_chars=1_500,
        context_compaction_enabled=True,
        context_max_chars=18_000,
        context_compaction_trigger_ratio=0.35,
        context_compaction_preserve_recent_messages=4,
        context_compaction_summarizer_input_max_chars=4_000,
        context_compaction_summary_max_chars=800,
        container_cleanup_delay_minutes=5,
        tool_capabilities=tool_capabilities,
        secret_backend="env",
    )
    await _preflight(settings, repo=repo, pr_number=pr_number, repo_path=repo_path)

    storage = SQLiteStorage(DEMO_DB)
    await storage.migrate()
    registry = build_registry()
    schemas = ContainerSchemaRegistry(
        [
            ContainerSchema(
                name="python-analysis",
                image="python:3.12-alpine",
                mount=repo_path,
                workdir="/repo",
                network=False,
                read_only_root=True,
                tmpfs_tmp=True,
                tmpfs_workdir=False,
                share_across_tools=True,
            )
        ]
    )
    runtime = await build_runtime_async(
        settings=settings,
        storage=storage,
        registry=registry,
        policy=CapabilityPolicy(CapabilityGrant(frozenset(settings.tool_capabilities))),
        secret_resolver=EnvSecretResolver(prefix=settings.secret_env_prefix),
        executors={
            ExecutionMode.IN_PROCESS: InProcessExecutor(registry.function_for),
            ExecutionMode.CONTAINER: DockerContainerExecutor(
                docker_bin=settings.docker_bin,
                schemas=schemas,
                allowed_mount_root=repo_path,
            ),
        },
        own_storage=True,
        own_model=True,
        own_embeddings=True,
        own_executors=True,
    )
    try:
        if runtime.model is None:
            raise RuntimeError("This example requires a real model provider.")
        model = runtime.model
        assert runtime.memory is not None
        runtime.memory.extractor = ReviewPreferenceMemoryExtractor()

        session = Session(
            metadata={
                "example": "github_pr_review_agent",
                "repo": repo,
                "pr": pr_number,
                "repo_path": str(repo_path),
            }
        )
        await storage.create_session(session)
        register_run_tools(registry, storage=storage)

        loop = runtime.agent_loop(
            max_iterations=int(os.environ.get("HARNESS_PR_REVIEW_MAX_ITERATIONS", "32")),
            stop_after_tools={"github.pr_comment"},
            context_compactor=RollingSummaryContextCompactor(
                model,
                ContextCompactionPolicy(
                    enabled=True,
                    max_context_chars=18_000,
                    trigger_ratio=0.35,
                    preserve_recent_messages=4,
                    summarizer_input_max_chars=4_000,
                    summary_max_chars=800,
                ),
            )
        )
        result = await loop.run(
            session.id,
            _review_prompt(
                repo=repo,
                pr_number=pr_number,
                repo_path=repo_path,
            ),
        )
        replies = _replies_from_tool_results(result.tool_results)
        await _mark_replies_processed(
            storage,
            repo=repo,
            pr_number=pr_number,
            replies=replies,
        )

        turns = await storage.list_turns(session.id, limit=None)
        tool_calls = await storage.list_tool_calls(session.id, limit=None)
        events = await storage.list_events(session.id, limit=None)
        memories = await storage.list_memories("github-pr-review-demo")
        comment_url = _comment_url_from_results(result.tool_results)
        if not replies and comment_url is None:
            raise RuntimeError("Review finished without posting a PR comment.")

        print(
            json.dumps(
                {
                    "session_id": session.id,
                    "repo": repo,
                    "pr_number": pr_number,
                    "repo_path": str(repo_path),
                    "model_provider": settings.model_provider,
                    "embedding_provider": settings.embedding_provider,
                    "final": result.final,
                    "tool_statuses": {call.tool_name: call.status for call in tool_calls},
                    "memory_ids_injected": result.memory_ids,
                    "comment_url": comment_url,
                    "reply_comment_ids_processed": [reply.comment_id for reply in replies],
                    "counts": {
                        "turns": len(turns),
                        "tool_calls": len(tool_calls),
                        "events": len(events),
                        "memories": len(memories),
                    },
                    "dashboard": (
                        f"HARNESS_SQLITE_PATH={DEMO_DB} "
                        "uv run harness dashboard --host 127.0.0.1 --port 8080"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        await runtime.close()


async def post_pr_comment(*, repo: str, pr_number: int, token: str, review: str) -> str:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = _comment_body(review)
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments",
            headers=headers,
            json={"body": body},
        )
    response.raise_for_status()
    payload = response.json()
    url = payload.get("html_url")
    return str(url) if isinstance(url, str) else ""


async def _unprocessed_replies(
    storage: SQLiteStorage,
    *,
    repo: str,
    pr_number: int,
    token: str,
) -> list[PullRequestReply]:
    await _ensure_processed_comments_table(storage)
    replies = await fetch_pr_user_replies(repo=repo, pr_number=pr_number, token=token)
    async with aiosqlite.connect(storage.path) as db:
        rows = await db.execute_fetchall(
            """
            select comment_id from github_pr_review_processed_comments
            where repo = ? and pr_number = ?
            """,
            (repo, pr_number),
        )
    processed_ids = {int(row[0]) for row in rows}
    return [reply for reply in replies if reply.comment_id not in processed_ids]


async def _mark_replies_processed(
    storage: SQLiteStorage,
    *,
    repo: str,
    pr_number: int,
    replies: list[PullRequestReply],
) -> None:
    if not replies:
        return
    await _ensure_processed_comments_table(storage)
    async with aiosqlite.connect(storage.path) as db:
        await db.executemany(
            """
            insert or ignore into github_pr_review_processed_comments (
              repo, pr_number, comment_id, author, processed_at
            )
            values (?, ?, ?, ?, ?)
            """,
            [
                (repo, pr_number, reply.comment_id, reply.author, utc_now().isoformat())
                for reply in replies
            ],
        )
        await db.commit()


async def _ensure_processed_comments_table(storage: SQLiteStorage) -> None:
    async with aiosqlite.connect(storage.path) as db:
        await db.execute(
            """
            create table if not exists github_pr_review_processed_comments (
              repo text not null,
              pr_number integer not null,
              comment_id integer not null,
              author text not null,
              processed_at text not null,
              primary key (repo, pr_number, comment_id)
            )
            """
        )
        await db.commit()


def _comment_body(review: str) -> str:
    return (
        f"{AGENT_COMMENT_MARKER}\n"
        "### Harness PR Review Agent\n\n"
        f"{review.strip()}\n\n"
        "_Posted by the Harness PR review agent._"
    )


def _comment_url_from_results(results: list[Any]) -> str | None:
    for result in results:
        if getattr(result, "name", None) != "github.pr_comment" or result.status != "ok":
            continue
        output = result.output
        if isinstance(output, dict) and isinstance(output.get("url"), str):
            return output["url"]
    return None


def _review_prompt(
    *,
    repo: str,
    pr_number: int,
    repo_path: Path,
) -> str:
    return (
        "You are a GitHub PR review agent. Review the pull request like a pragmatic "
        "senior engineer: prioritize correctness, security, data integrity, runtime "
        "failures, missing tests, and release risk.\n\n"
        f"GitHub repo: {repo}\n"
        f"Pull request number: {pr_number}\n"
        f"Local repository root: {repo_path}\n\n"
        "Relevant review preferences may already appear in the context above. Treat "
        "them as standing instructions when they apply to this PR.\n\n"
        "Start by calling github.pr_replies. If it returns user replies, handle those "
        "replies before doing anything else. Treat preference-like replies as standing "
        "review guidance for future runs. Do not perform a fresh code review unless a "
        "reply explicitly asks for one; otherwise return a concise final answer "
        "summarizing what was processed.\n\n"
        "For a fresh review, call github.pr_context to fetch PR metadata, changed "
        "files, and checks. Then inspect the local checkout with repo.bash. Use as "
        "many repo.bash calls as needed to understand the project structure, changed "
        "code, tests, and likely failure modes. Stop exploring when you have enough "
        "evidence for a concrete review.\n\n"
        "repo.bash runs inside Docker with no network and read-only access to the "
        "local checkout at /repo. "
        "Prefer cheap read-only commands such as find, sed, grep, python one-liners, "
        "and test/config discovery. Do not inspect .env*, .git, .venv, data, dist, local "
        "caches, credential files, or secret-looking files. Do not use git, rg, package "
        "managers, network access, or commands that write to the repository. If a command "
        "is unavailable or fails, adapt with simpler POSIX tools.\n\n"
        "Produce a concise code review with concrete findings, test gaps, and a "
        "release-readiness recommendation. For fresh reviews, post the review with "
        "github.pr_comment exactly once, then return a final answer that includes "
        "the comment URL."
    )


def _normalize_private_key(value: str) -> str:
    return value.replace("\\n", "\n").strip()


async def _preflight(
    settings: HarnessSettings,
    *,
    repo: str,
    pr_number: int,
    repo_path: Path,
) -> None:
    if (
        settings.model_provider == "openai" or settings.embedding_provider == "openai"
    ) and not settings.openai_api_key:
        raise RuntimeError(
            "Set HARNESS_OPENAI_API_KEY or choose Codex with MiniLM embeddings."
        )
    if settings.embedding_provider not in {"minilm", "openai"}:
        raise RuntimeError(
            "This GitHub PR review example expects HARNESS_EMBEDDING_PROVIDER=minilm or openai."
        )
    if not repo:
        raise RuntimeError("Set HARNESS_GITHUB_REPO, for example owner/repository.")
    if pr_number <= 0:
        raise RuntimeError("Set HARNESS_GITHUB_PR to a real pull request number.")
    if not repo_path.is_dir():
        raise RuntimeError(f"HARNESS_REPO_PATH must point to a local checkout: {repo_path}")
    if not os.environ.get("HARNESS_GITHUB_APP_ID"):
        raise RuntimeError("Set HARNESS_GITHUB_APP_ID to the GitHub App id.")
    if not os.environ.get("HARNESS_GITHUB_INSTALLATION_ID"):
        raise RuntimeError("Set HARNESS_GITHUB_INSTALLATION_ID to the App installation id.")
    if not os.environ.get("HARNESS_SECRET_GITHUB_APP_PRIVATE_KEY"):
        raise RuntimeError("Set HARNESS_SECRET_GITHUB_APP_PRIVATE_KEY to the App private key.")
    docker = await anyio.run_process([settings.docker_bin, "info"], check=False)
    if docker.returncode != 0:
        raise RuntimeError(
            "Docker is required for the sandboxed tool. Start Docker and rerun this example."
        )


def _target_repo_path() -> Path:
    raw_path = os.environ.get("HARNESS_REPO_PATH")
    return Path(raw_path).expanduser().resolve() if raw_path else Path.cwd().resolve()


def _load_local_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("HARNESS_"):
            continue
        os.environ.setdefault(key, value.strip().strip("'\""))


def _reset_demo_database_if_requested() -> None:
    if os.environ.get("HARNESS_EXAMPLE_RESET_DB") != "1":
        return
    for suffix in ("", "-wal", "-shm"):
        Path(f"{DEMO_DB}{suffix}").unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
