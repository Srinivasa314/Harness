from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest


def test_github_pr_example_repo_snapshot_is_generic(tmp_path: Path) -> None:
    module = _load_example_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'other-project'\n")
    (repo / "package.json").write_text('{"scripts": {"test": "vitest"}}')
    (repo / "src").mkdir()
    (repo / "src" / "main.ts").write_text("export const value = 1;\n")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "ignored.js").write_text("")

    snapshot = module._repo_snapshot(repo)

    assert snapshot["root_name"] == "repo"
    assert "pyproject.toml" in snapshot["config_files"]
    assert "package.json" in snapshot["config_files"]
    assert "src/main.ts" in snapshot["files"]
    assert "node_modules/ignored.js" not in snapshot["files"]


def test_github_pr_example_project_checker_detects_generic_repo() -> None:
    module = _load_example_module()
    payload = {
        "arguments": {
            "repo_snapshot": {
                "root_name": "service",
                "files": [
                    ".github/workflows/test.yml",
                    "Dockerfile",
                    "go.mod",
                    "main.go",
                    "main_test.go",
                ],
                "config_files": {
                    "go.mod": "module example.com/service\n",
                    ".github/workflows/test.yml": "name: test\nrun: go test ./...\n",
                },
            }
        }
    }

    completed = subprocess.run(
        [sys.executable, "-c", module.PROJECT_CHECKER],
        input=json.dumps(payload),
        text=True,
        check=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)

    assert result["root_name"] == "service"
    assert "go" in result["languages"]
    assert "go modules" in result["package_managers"]
    assert "go test" in result["test_indicators"]
    assert "test files present" in result["test_indicators"]
    assert result["ci_present"] is True
    assert result["container_files"] == ["Dockerfile"]


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


def _load_example_module() -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / "github_pr_review_agent" / "agent.py"
    spec = importlib.util.spec_from_file_location("github_pr_review_agent_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
