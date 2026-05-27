from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import anyio
import httpx

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


class PullRequestMemoryExtractor(MemoryExtractor):
    async def extract(self, exchange: MemoryExchange) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for output in exchange.tool_outputs:
            if output.get("name") != "github.pr_context":
                continue
            pr_context = output.get("output")
            if not isinstance(pr_context, dict):
                continue
            repo = pr_context.get("repo")
            changed_files = pr_context.get("changed_files")
            changed_files_total = pr_context.get("changed_files_total")
            check_runs = pr_context.get("check_runs")
            if not isinstance(repo, str):
                continue
            changed_count = (
                changed_files_total
                if isinstance(changed_files_total, int)
                else len(changed_files)
                if isinstance(changed_files, list)
                else 0
            )
            failed_checks = [
                str(check.get("name"))
                for check in check_runs or []
                if isinstance(check, dict) and check.get("conclusion") not in {None, "success"}
            ]
            detail = f"Recent PR reviews for {repo} should consider {changed_count} changed files"
            if failed_checks:
                detail += f" and failed checks: {', '.join(failed_checks[:5])}"
            candidates.append(
                MemoryCandidate(
                    text=detail,
                    scope=MemoryScope.AGENT,
                    importance=0.7,
                    metadata={"source": "github_pr_context"},
                )
            )
        if candidates:
            return candidates
        if "release-readiness" in exchange.assistant_message.lower():
            return [
                MemoryCandidate(
                    text="Future PR reviews should include a release-readiness recommendation.",
                    scope=MemoryScope.AGENT,
                    importance=0.6,
                    metadata={"source": "assistant_review"},
                )
            ]
        return []


async def github_pr_context(arguments: dict[str, Any], secrets: dict[str, str]) -> dict[str, Any]:
    repo = str(arguments["repo"])
    pr_number = int(arguments["pr_number"])
    token = secrets["github_token"]
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
    url = await post_pr_comment(
        repo=repo,
        pr_number=pr_number,
        token=secrets["github_token"],
        review=body,
    )
    return {"url": url}


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


def build_registry(*, enable_comment_tool: bool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="github.pr_context",
            description="Fetch real GitHub pull request metadata, changed files, and check runs.",
            execution_mode=ExecutionMode.IN_PROCESS,
            required_capabilities=["github:pr"],
            required_secrets=["github_token"],
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
    if enable_comment_tool:
        registry.register(
            ToolDefinition(
                name="github.pr_comment",
                description="Post the final review as a GitHub pull request comment.",
                execution_mode=ExecutionMode.IN_PROCESS,
                required_capabilities=["github:comment"],
                required_secrets=["github_token"],
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


async def main() -> None:
    _load_local_env_file(ROOT / ".env.development")
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ST_CACHE))
    _reset_demo_database_if_requested()

    repo = os.environ.get("HARNESS_GITHUB_REPO", "")
    pr_number = int(os.environ.get("HARNESS_GITHUB_PR", "0") or "0")
    repo_path = _target_repo_path()
    comment_enabled = _comment_enabled()
    tool_capabilities = ["github:pr", "repo:sandbox"]
    if comment_enabled:
        tool_capabilities.append("github:comment")
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
    registry = build_registry(enable_comment_tool=comment_enabled)
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
        runtime.memory.extractor = PullRequestMemoryExtractor()

        session = Session(
            metadata={
                "example": "github_pr_review_agent",
                "repo": repo,
                "pr": pr_number,
                "repo_path": str(repo_path),
            }
        )
        await storage.create_session(session)

        loop = runtime.agent_loop(
            max_iterations=12,
            stop_after_tools={"github.pr_comment"} if comment_enabled else None,
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
                comment_enabled=comment_enabled,
            ),
        )

        turns = await storage.list_turns(session.id, limit=None)
        tool_calls = await storage.list_tool_calls(session.id, limit=None)
        events = await storage.list_events(session.id, limit=None)
        memories = await storage.list_memories("github-pr-review-demo")
        comment_url = _comment_url_from_results(result.tool_results)
        if comment_enabled and comment_url is None:
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


def _comment_body(review: str) -> str:
    return (
        f"{AGENT_COMMENT_MARKER}\n"
        "### Harness PR Review Agent\n\n"
        f"{review.strip()}\n\n"
        "_Posted by the Harness PR review agent._"
    )


def _comment_enabled() -> bool:
    return os.environ.get("HARNESS_GITHUB_COMMENT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


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
    comment_enabled: bool,
) -> str:
    comment_instruction = (
        "After drafting the review, call github.pr_comment with the exact review body. "
        "After the comment tool succeeds, return a final answer that includes the comment URL."
        if comment_enabled
        else "Do not post a PR comment in this run; return the review as the final answer."
    )
    return (
        "You are a GitHub PR review agent. Produce a concise code review with concrete "
        "findings, test gaps, and a release-readiness recommendation. Use the harness "
        "JSON protocol, with no markdown outside the JSON.\n\n"
        f"GitHub repo: {repo}\n"
        f"Pull request number: {pr_number}\n"
        f"Local repository root: {repo_path}\n\n"
        "Strict workflow: iteration 1 must call github.pr_context with the repo and PR "
        "number. Iteration 2 must call repo.bash with a batch of two to four focused "
        "commands of your choice. Iteration 3 must call github.pr_comment if commenting "
        "is enabled, otherwise return the final review. Do not call github.pr_context "
        "after iteration 1. The repo.bash tool runs inside Docker with no network and "
        "read-only access to the local checkout at /repo. "
        "A repo.bash tool call must be shaped exactly like "
        "{\"tool_calls\":[{\"name\":\"repo.bash\",\"arguments\":{\"command\":"
        "\"find . -maxdepth 2 \\( -path './.git' -o -path './.venv' -o "
        "-path './data' -o -name '.env*' -o -name '*cache*' \\) -prune -o "
        "-type f -print | sort | head -80\"}}]}. "
        "Run enough focused commands to inspect the PR surface, project configuration, "
        "tests, CI, risky files, and code quality signals before writing the review. "
        "Prefer cheap read-only commands such as find, sed, grep, python one-liners, "
        "and test/config discovery. Do not inspect .env*, .git, .venv, data, dist, local "
        "caches, credential files, or secret-looking files. Do not use git, rg, package "
        "managers, network access, or commands that write to the repository. If a command "
        "is unavailable or fails, adapt once with simpler POSIX tools inside the same "
        "exploration batch if possible. After the repo.bash batch, draft the review from "
        "the available evidence and proceed to the comment/final step. If commenting is "
        "enabled, you must call github.pr_comment exactly once before the final answer, "
        "shaped exactly like "
        "{\"tool_calls\":[{\"name\":\"github.pr_comment\",\"arguments\":"
        "{\"repo\":\"owner/name\",\"pr_number\":1,\"body\":\"review text\"}}]}.\n\n"
        f"{comment_instruction}"
    )


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
    if not os.environ.get("HARNESS_SECRET_GITHUB_TOKEN"):
        raise RuntimeError("Set HARNESS_SECRET_GITHUB_TOKEN to a real GitHub token.")
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
