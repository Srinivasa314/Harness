from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import aiosqlite
import httpx
import pytest


def test_github_pr_example_bash_runner_explores_generic_repo(tmp_path: Path) -> None:
    module = _load_example_module()
    repo = _generic_repo(tmp_path)
    payload = {
        "arguments": {
            "command": (
                "printf 'files='; find . -maxdepth 3 -type f | sort | sed 's#^./##'; "
                "printf '\\nconfig='; sed -n '1,5p' go.mod"
            )
        }
    }
    completed = subprocess.run(
        [sys.executable, "-c", module.BASH_RUNNER],
        input=json.dumps(payload),
        text=True,
        check=True,
        capture_output=True,
        cwd=repo,
    )

    result = json.loads(completed.stdout)

    assert result["returncode"] == 0
    assert "go.mod" in result["stdout"]
    assert "main_test.go" in result["stdout"]
    assert "module example.com/service" in result["stdout"]


def test_github_pr_example_bash_runner_reports_nonzero_exit(tmp_path: Path) -> None:
    module = _load_example_module()
    repo = _generic_repo(tmp_path)
    payload = {"arguments": {"command": "sed -n '1p' missing-file"}}

    completed = subprocess.run(
        [sys.executable, "-c", module.BASH_RUNNER],
        input=json.dumps(payload),
        text=True,
        check=True,
        capture_output=True,
        cwd=repo,
    )

    result = json.loads(completed.stdout)

    assert result["returncode"] != 0
    assert "missing-file" in result["stderr"]


def test_github_pr_example_comment_tool_is_optional() -> None:
    module = _load_example_module()

    without_comment = module.build_registry(enable_comment_tool=False)
    with_comment = module.build_registry(enable_comment_tool=True)
    without_comment_names = {
        definition.name for definition in without_comment.list_definitions()
    }
    with_comment_names = {definition.name for definition in with_comment.list_definitions()}

    assert "github.pr_comment" not in without_comment_names
    assert "github.pr_comment" in with_comment_names
    assert "repo.bash" in with_comment_names
    definitions = {definition.name: definition for definition in with_comment.list_definitions()}
    pr_context = definitions["github.pr_context"]
    pr_comment = definitions["github.pr_comment"]
    assert pr_context.required_secrets == ["github_app_private_key"]
    assert pr_comment.required_secrets == ["github_app_private_key"]


def test_github_pr_example_registers_run_scoped_tools(tmp_path: Path) -> None:
    module = _load_example_module()
    registry = module.build_registry(enable_comment_tool=False)
    storage = module.SQLiteStorage(tmp_path / "example.sqlite3")

    module.register_run_tools(registry, storage=storage)
    definitions = {definition.name: definition for definition in registry.list_definitions()}

    assert definitions["github.pr_replies"].required_capabilities == ["github:replies"]
    assert definitions["github.pr_replies"].required_secrets == ["github_app_private_key"]
    assert "memory.search" not in definitions


@pytest.mark.anyio
async def test_github_pr_example_extracts_user_review_preferences() -> None:
    module = _load_example_module()
    extractor = module.ReviewPreferenceMemoryExtractor()

    memories = await extractor.extract(
        module.MemoryExchange(
            session_id="session-1",
            user_message="review",
            assistant_message="noted",
            tool_outputs=[
                {
                    "name": "github.pr_replies",
                    "output": {
                        "replies": [
                            {
                                "comment_id": 101,
                                "author": "alice",
                                "body": (
                                    "- Prefer stricter comments on missing tests.\n"
                                    "- Always include migration risk."
                                ),
                            },
                            {
                                "comment_id": 102,
                                "author": "bob",
                                "body": "Thanks for the review.",
                            },
                        ]
                    },
                }
            ],
        )
    )

    assert [memory.metadata for memory in memories] == [
        {
            "source": "github_pr_comment",
            "github_comment_id": 101,
            "github_comment_author": "alice",
        },
        {
            "source": "github_pr_comment",
            "github_comment_id": 101,
            "github_comment_author": "alice",
        },
    ]
    assert memories[0].scope == module.MemoryScope.AGENT
    assert "Prefer stricter comments" in memories[0].text
    assert "Always include migration risk" in memories[1].text
    assert memories[0].metadata["github_comment_id"] == 101
    assert memories[0].metadata["github_comment_author"] == "alice"


@pytest.mark.anyio
async def test_github_pr_example_ignores_non_preference_replies() -> None:
    module = _load_example_module()
    extractor = module.ReviewPreferenceMemoryExtractor()

    memories = await extractor.extract(
        module.MemoryExchange(
            session_id="session-1",
            user_message="review",
            assistant_message="noted",
            tool_outputs=[
                {
                    "name": "github.pr_replies",
                    "output": {
                        "replies": [
                            {
                                "comment_id": 101,
                                "author": "alice",
                                "body": "Thanks for the review.",
                            }
                        ]
                    },
                }
            ],
        )
    )

    assert memories == []


@pytest.mark.anyio
async def test_github_pr_example_fetches_user_replies_after_last_agent_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("https://api.github.test/repos/owner/repo")
        return httpx.Response(
            200,
            json=[
                {
                    "id": 1,
                    "body": "Earlier user comment",
                    "user": {"login": "alice"},
                },
                {
                    "id": 2,
                    "body": f"{module.AGENT_COMMENT_MARKER}\nAgent review",
                    "user": {"login": "review-app"},
                },
                {
                    "id": 3,
                    "body": "Prefer stricter test comments.",
                    "user": {"login": "alice"},
                },
                {
                    "id": 4,
                    "body": f"{module.AGENT_COMMENT_MARKER}\nAgent follow-up",
                    "user": {"login": "review-app"},
                },
                {
                    "id": 5,
                    "body": "Always include rollout risk.",
                    "user": {"login": "bob"},
                },
            ],
        )

    monkeypatch.setattr(module, "GITHUB_API", "https://api.github.test")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: client)

    try:
        replies = await module.fetch_pr_user_replies(
            repo="owner/repo",
            pr_number=7,
            token="installation-token",
        )
    finally:
        await client.aclose()

    assert replies == [
        module.PullRequestReply(
            comment_id=5,
            author="bob",
            body="Always include rollout risk.",
        )
    ]


@pytest.mark.anyio
async def test_github_pr_example_filters_already_processed_replies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module()
    storage = module.SQLiteStorage(tmp_path / "example.sqlite3")
    await storage.migrate()
    await module._mark_replies_processed(
        storage,
        repo="owner/repo",
        pr_number=7,
        replies=[module.PullRequestReply(3, "alice", "Prefer tests.")],
    )

    async def fake_fetch_pr_user_replies(
        *,
        repo: str,
        pr_number: int,
        token: str,
    ) -> list[Any]:
        assert repo == "owner/repo"
        assert pr_number == 7
        assert token == "installation-token"
        return [
            module.PullRequestReply(3, "alice", "Prefer tests."),
            module.PullRequestReply(4, "bob", "Always include migration risk."),
        ]

    monkeypatch.setattr(module, "fetch_pr_user_replies", fake_fetch_pr_user_replies)

    replies = await module._unprocessed_replies(
        storage,
        repo="owner/repo",
        pr_number=7,
        token="installation-token",
    )

    assert replies == [module.PullRequestReply(4, "bob", "Always include migration risk.")]


@pytest.mark.anyio
async def test_github_pr_example_tracks_processed_replies_separately(tmp_path: Path) -> None:
    module = _load_example_module()
    storage = module.SQLiteStorage(tmp_path / "example.sqlite3")
    await storage.migrate()

    await module._mark_replies_processed(
        storage,
        repo="owner/repo",
        pr_number=7,
        replies=[
            module.PullRequestReply(3, "alice", "Prefer tests."),
            module.PullRequestReply(4, "bob", "Always include migration risk."),
        ],
    )

    async with aiosqlite.connect(storage.path) as db:
        rows = await db.execute_fetchall(
            """
            select repo, pr_number, comment_id, author
            from github_pr_review_processed_comments
            order by comment_id
            """
        )

    assert [tuple(row) for row in rows] == [
        ("owner/repo", 7, 3, "alice"),
        ("owner/repo", 7, 4, "bob"),
    ]


@pytest.mark.anyio
async def test_github_pr_example_paginates_github_lists() -> None:
    module = _load_example_module()

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        per_page = int(request.url.params.get("per_page", "100"))
        if str(request.url).startswith("https://api.github.test/files"):
            count = per_page if page == 1 else 2
            return httpx.Response(
                200,
                json=[
                    {
                        "filename": f"file-{page}-{index}.py",
                        "status": "modified",
                        "additions": 1,
                        "deletions": 0,
                    }
                    for index in range(count)
                ],
            )
        if str(request.url).startswith("https://api.github.test/checks"):
            count = per_page if page == 1 else 1
            return httpx.Response(
                200,
                json={
                    "check_runs": [
                        {"name": f"check-{page}-{index}", "conclusion": "success"}
                        for index in range(count)
                    ]
                },
            )
        raise AssertionError(str(request.url))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        files = await module._get_paginated_list(
            client,
            "https://api.github.test/files",
            headers={},
            list_key=None,
        )
        checks = await module._get_paginated_list(
            client,
            "https://api.github.test/checks",
            headers={},
            list_key="check_runs",
        )

    assert len(files) == module.GITHUB_PAGE_SIZE + 2
    assert files[-1]["filename"] == "file-2-1.py"
    assert len(checks) == module.GITHUB_PAGE_SIZE + 1
    assert checks[-1]["name"] == "check-2-0"


@pytest.mark.anyio
async def test_github_pr_example_tolerates_unavailable_check_runs() -> None:
    module = _load_example_module()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(403, json={}))
    ) as client:
        checks = await module._get_paginated_list(
            client,
            "https://api.github.test/checks",
            headers={},
            list_key="check_runs",
            tolerate_statuses={403},
        )

    assert checks == []


def test_github_pr_example_comment_body_is_agent_branded() -> None:
    module = _load_example_module()

    body = module._comment_body("Looks ready.")

    assert body.startswith(module.AGENT_COMMENT_MARKER)
    assert "Harness PR Review Agent" in body
    assert "Looks ready." in body
    assert "Posted by the Harness PR review agent" in body


@pytest.mark.anyio
async def test_github_pr_example_posts_agent_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module()
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={"html_url": "https://github.test/owner/repo/pull/7#issuecomment-1"},
        )

    monkeypatch.setattr(module, "GITHUB_API", "https://api.github.test")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: client)

    try:
        url = await module.post_pr_comment(
            repo="owner/repo",
            pr_number=7,
            token="secret-token",
            review="Release-ready.",
        )
    finally:
        await client.aclose()

    assert url == "https://github.test/owner/repo/pull/7#issuecomment-1"
    assert captured["url"] == "https://api.github.test/repos/owner/repo/issues/7/comments"
    assert captured["authorization"] == "Bearer secret-token"
    assert isinstance(captured["payload"], dict)
    assert "Harness PR Review Agent" in str(captured["payload"]["body"])
    assert "Release-ready." in str(captured["payload"]["body"])


@pytest.mark.anyio
async def test_github_pr_example_mints_github_app_installation_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module()
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        return httpx.Response(201, json={"token": "installation-token"})

    monkeypatch.setenv("HARNESS_GITHUB_APP_ID", "12345")
    monkeypatch.setenv("HARNESS_GITHUB_INSTALLATION_ID", "67890")
    monkeypatch.setattr(module, "GITHUB_API", "https://api.github.test")
    monkeypatch.setattr(module.jwt, "encode", lambda *args, **kwargs: "app-jwt")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: client)

    try:
        token = await module.github_app_installation_token(
            {"github_app_private_key": "line1\\nline2"}
        )
    finally:
        await client.aclose()

    assert token == "installation-token"
    assert captured["url"] == "https://api.github.test/app/installations/67890/access_tokens"
    assert captured["authorization"] == "Bearer app-jwt"


@pytest.mark.anyio
async def test_github_pr_example_comment_tool_uses_github_app_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module()
    captured: dict[str, Any] = {}

    async def fake_installation_token(secrets: dict[str, str]) -> str:
        captured["secrets"] = secrets
        return "installation-token"

    async def fake_post_pr_comment(
        *,
        repo: str,
        pr_number: int,
        token: str,
        review: str,
    ) -> str:
        captured["comment"] = {
            "repo": repo,
            "pr_number": pr_number,
            "token": token,
            "review": review,
        }
        return "https://github.test/comment"

    monkeypatch.setattr(module, "github_app_installation_token", fake_installation_token)
    monkeypatch.setattr(module, "post_pr_comment", fake_post_pr_comment)

    result = await module.github_pr_comment(
        {"repo": "owner/repo", "pr_number": 7, "body": "Ready."},
        {"github_app_private_key": "private-key"},
    )

    assert result == {"url": "https://github.test/comment"}
    assert captured["secrets"] == {"github_app_private_key": "private-key"}
    assert captured["comment"] == {
        "repo": "owner/repo",
        "pr_number": 7,
        "token": "installation-token",
        "review": "Ready.",
    }


def _load_example_module() -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / "github_pr_review_agent" / "agent.py"
    spec = importlib.util.spec_from_file_location("github_pr_review_agent_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generic_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".github" / "workflows" / "test.yml").write_text("name: test\nrun: go test ./...\n")
    (repo / "Dockerfile").write_text("FROM scratch\n")
    (repo / "go.mod").write_text("module example.com/service\n")
    (repo / "main.go").write_text("package main\n")
    (repo / "main_test.go").write_text("package main\n")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "ignored.js").write_text("")
    return repo
