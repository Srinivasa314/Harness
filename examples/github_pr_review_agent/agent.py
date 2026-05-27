from __future__ import annotations

import asyncio
import json
import os
import subprocess
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
MAX_REPO_FILES = 500
MAX_CONFIG_CHARS = 4_000
CONFIG_FILENAMES = {
    ".github/dependabot.yml",
    ".github/dependabot.yaml",
    ".pre-commit-config.yaml",
    ".semgrep.yml",
    "Cargo.toml",
    "Dockerfile",
    "Gemfile",
    "Makefile",
    "build.gradle",
    "build.gradle.kts",
    "compose.yaml",
    "docker-compose.yml",
    "go.mod",
    "package.json",
    "pnpm-lock.yaml",
    "pom.xml",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements.txt",
    "tsconfig.json",
    "tox.ini",
    "uv.lock",
    "yarn.lock",
}
SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "target",
}

PROJECT_CHECKER = r"""
from collections import Counter
import json
import sys

payload = json.load(sys.stdin)
snapshot = payload["arguments"]["repo_snapshot"]
files = [str(path) for path in snapshot.get("files", [])]
config_files = snapshot.get("config_files", {})
lower_files = [path.lower() for path in files]
suffixes = Counter(path.rsplit(".", 1)[-1] for path in lower_files if "." in path)

language_markers = {
    "python": [".py", "pyproject.toml", "requirements.txt", "tox.ini"],
    "javascript": [".js", "package.json"],
    "typescript": [".ts", ".tsx", "tsconfig.json"],
    "go": [".go", "go.mod"],
    "rust": [".rs", "Cargo.toml"],
    "java": [".java", "pom.xml", "build.gradle", "build.gradle.kts"],
    "ruby": [".rb", "Gemfile"],
    "shell": [".sh"],
    "docker": ["Dockerfile", "docker-compose.yml", "compose.yaml"],
}

def has_marker(markers):
    return any(
        path.endswith(marker.lower()) or path == marker.lower()
        for marker in markers
        for path in lower_files
    ) or any(marker in config_files for marker in markers)

languages = sorted(name for name, markers in language_markers.items() if has_marker(markers))
package_managers = []
if "package.json" in config_files:
    package_managers.append("npm-compatible")
if "pnpm-lock.yaml" in lower_files:
    package_managers.append("pnpm")
if "yarn.lock" in lower_files:
    package_managers.append("yarn")
if "pyproject.toml" in config_files or "requirements.txt" in config_files:
    package_managers.append("python")
if "uv.lock" in config_files:
    package_managers.append("uv")
if "go.mod" in config_files:
    package_managers.append("go modules")
if "Cargo.toml" in config_files:
    package_managers.append("cargo")
if "pom.xml" in config_files:
    package_managers.append("maven")
if "build.gradle" in config_files or "build.gradle.kts" in config_files:
    package_managers.append("gradle")
if "Gemfile" in config_files:
    package_managers.append("bundler")

config_text = "\n".join(str(value).lower() for value in config_files.values())
test_indicators = sorted({
    indicator
    for indicator in [
        "pytest",
        "unittest",
        "jest",
        "vitest",
        "mocha",
        "go test",
        "cargo test",
        "junit",
        "rspec",
    ]
    if indicator in config_text
})
has_test_file = any(
    "/test" in path or path.startswith("test") or ".test." in path or "_test." in path
    for path in lower_files
)
if has_test_file:
    test_indicators.append("test files present")

ci_present = any(path.startswith(".github/workflows/") for path in lower_files) or any(
    path in {".gitlab-ci.yml", "circle.yml", ".circleci/config.yml"} for path in lower_files
)
container_suffixes = ("dockerfile", "docker-compose.yml", "compose.yaml")
container_files = sorted(path for path in files if path.lower().endswith(container_suffixes))
security_indicators = sorted({
    indicator
    for indicator in [
        "dependabot",
        "codeql",
        "semgrep",
        "trivy",
        "bandit",
        "npm audit",
        "cargo audit",
    ]
    if indicator in config_text or any(indicator in path for path in lower_files)
})

notes = []
if not test_indicators:
    notes.append("No obvious test configuration or test files were detected in the local snapshot.")
if not ci_present:
    notes.append("No common CI workflow file was detected in the local snapshot.")
if not security_indicators:
    notes.append("No common dependency or static-analysis security configuration was detected.")

print(json.dumps({
    "root_name": snapshot.get("root_name"),
    "file_count_sampled": len(files),
    "top_extensions": suffixes.most_common(8),
    "languages": languages,
    "package_managers": sorted(set(package_managers)),
    "test_indicators": sorted(set(test_indicators)),
    "ci_present": ci_present,
    "container_files": container_files[:20],
    "security_indicators": security_indicators,
    "notes": notes,
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
            check_runs = pr_context.get("check_runs")
            if not isinstance(repo, str):
                continue
            changed_count = len(changed_files) if isinstance(changed_files, list) else 0
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
        files_response = await client.get(
            f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files",
            headers=headers,
            params={"per_page": 100},
        )
        files_response.raise_for_status()
        checks_response = await client.get(
            f"{GITHUB_API}/repos/{repo}/commits/{pr_response.json()['head']['sha']}/check-runs",
            headers=headers,
            params={"per_page": 50},
        )
    pr = pr_response.json()
    files = files_response.json()
    checks = checks_response.json() if checks_response.status_code == 200 else {"check_runs": []}
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
        "changed_files": [
            {
                "filename": item["filename"],
                "status": item["status"],
                "additions": item["additions"],
                "deletions": item["deletions"],
                "patch_excerpt": (item.get("patch") or "")[:1_500],
            }
            for item in files[:40]
        ],
        "check_runs": [
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "conclusion": item.get("conclusion"),
            }
            for item in checks.get("check_runs", [])[:20]
        ],
    }


def build_registry() -> ToolRegistry:
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
                    "changed_files": {"type": "array"},
                    "check_runs": {"type": "array"},
                },
                "additionalProperties": True,
            },
        ),
        github_pr_context,
    )
    registry.register(
        ToolDefinition(
            name="repo.project_check",
            description="Analyze a local repository snapshot inside Docker.",
            execution_mode=ExecutionMode.CONTAINER,
            container_schema="python-analysis",
            container_command=["python", "-c", PROJECT_CHECKER],
            required_capabilities=["repo:sandbox"],
            timeout_seconds=30,
            input_schema={
                "type": "object",
                "required": ["repo_snapshot"],
                "properties": {
                    "repo_snapshot": {
                        "type": "object",
                        "required": ["root_name", "files", "config_files"],
                        "properties": {
                            "root_name": {"type": "string"},
                            "files": {"type": "array", "items": {"type": "string"}},
                            "config_files": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "additionalProperties": True,
                    },
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "required": [
                    "root_name",
                    "file_count_sampled",
                    "top_extensions",
                    "languages",
                    "package_managers",
                    "test_indicators",
                    "ci_present",
                    "container_files",
                    "security_indicators",
                    "notes",
                ],
                "properties": {
                    "root_name": {"type": "string"},
                    "file_count_sampled": {"type": "integer"},
                    "top_extensions": {"type": "array"},
                    "languages": {"type": "array", "items": {"type": "string"}},
                    "package_managers": {"type": "array", "items": {"type": "string"}},
                    "test_indicators": {"type": "array", "items": {"type": "string"}},
                    "ci_present": {"type": "boolean"},
                    "container_files": {"type": "array", "items": {"type": "string"}},
                    "security_indicators": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "array", "items": {"type": "string"}},
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
    settings = HarnessSettings(
        storage_backend="sqlite",
        sqlite_path=DEMO_DB,
        model_provider=os.environ.get("HARNESS_MODEL_PROVIDER", "openai"),
        openai_model=os.environ.get("HARNESS_OPENAI_MODEL", "gpt-4.1-mini"),
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
        tool_capabilities=["github:pr", "repo:sandbox"],
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
                network=False,
                read_only_root=True,
                tmpfs_tmp=True,
                tmpfs_workdir=True,
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
            _review_prompt(repo=repo, pr_number=pr_number, repo_path=repo_path),
        )

        turns = await storage.list_turns(session.id, limit=None)
        tool_calls = await storage.list_tool_calls(session.id, limit=None)
        events = await storage.list_events(session.id, limit=None)
        memories = await storage.list_memories("github-pr-review-demo")

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


def _review_prompt(*, repo: str, pr_number: int, repo_path: Path) -> str:
    repo_snapshot = _repo_snapshot(repo_path)
    repo_snapshot_json = json.dumps(repo_snapshot, indent=2, sort_keys=True)
    return (
        "You are a GitHub PR review agent. Produce a concise code review with concrete "
        "findings, test gaps, and a release-readiness recommendation. Use the harness "
        "JSON protocol, with no markdown outside the JSON.\n\n"
        f"GitHub repo: {repo}\n"
        f"Pull request number: {pr_number}\n"
        f"Local repository root: {repo_path}\n\n"
        "Recommended first tool batch: call github.pr_context with the repo and PR number "
        "and repo.project_check with the repo_snapshot below.\n\n"
        "repo_snapshot:\n"
        f"{repo_snapshot_json}"
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


def _repo_snapshot(repo_path: Path) -> dict[str, Any]:
    files = _repo_files(repo_path)
    return {
        "root_name": repo_path.name,
        "files": files[:MAX_REPO_FILES],
        "config_files": _read_config_files(repo_path, files),
        "truncated": len(files) > MAX_REPO_FILES,
    }


def _repo_files(repo_path: Path) -> list[str]:
    git_files = _git_files(repo_path)
    if git_files:
        return git_files
    files: list[str] = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repo_path)
        parts = set(relative.parts)
        if parts & SKIP_DIRS:
            continue
        files.append(relative.as_posix())
        if len(files) >= MAX_REPO_FILES:
            break
    return sorted(files)


def _git_files(repo_path: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "ls-files"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    return sorted(
        line.strip()
        for line in result.stdout.decode(errors="replace").splitlines()
        if line.strip()
    )


def _read_config_files(repo_path: Path, files: list[str]) -> dict[str, str]:
    configs: dict[str, str] = {}
    for relative in files:
        if relative not in CONFIG_FILENAMES and Path(relative).name not in CONFIG_FILENAMES:
            continue
        path = (repo_path / relative).resolve()
        if not path.is_relative_to(repo_path) or not path.is_file():
            continue
        try:
            configs[relative] = path.read_text(errors="replace")[:MAX_CONFIG_CHARS]
        except OSError:
            continue
    return configs


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
