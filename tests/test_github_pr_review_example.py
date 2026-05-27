from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


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


def _load_example_module() -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / "github_pr_review_agent" / "agent.py"
    spec = importlib.util.spec_from_file_location("github_pr_review_agent_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
