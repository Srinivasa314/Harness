from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_ci_cli_smoke_commands_are_supported(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(root / "src"),
        "HARNESS_STORAGE_BACKEND": "sqlite",
        "HARNESS_SQLITE_PATH": str(tmp_path / "harness.sqlite3"),
        "HARNESS_MODEL_PROVIDER": "none",
        "HARNESS_DISABLE_ENV_FILES": "1",
    }
    commands = [
        [sys.executable, "-m", "harness.cli", "migrate"],
        [
            sys.executable,
            "-m",
            "harness.cli",
            "doctor",
            "--output",
            str(tmp_path / "doctor.json"),
        ],
    ]

    for command in commands:
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr or completed.stdout


def test_github_workflow_does_not_reference_removed_eval_commands():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "test.yml").read_text()

    assert "eval-regression" not in workflow
    assert "export-evals" not in workflow
    assert "regression_pass.json" not in workflow
