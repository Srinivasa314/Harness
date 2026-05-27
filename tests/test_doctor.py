from __future__ import annotations

from harness.config import HarnessSettings
from harness.doctor import DoctorCheck, doctor_summary, run_doctor_checks


def test_doctor_summary_reports_overall_status():
    summary = doctor_summary(
        [
            DoctorCheck(name="ok", ok=True, detail="ok"),
            DoctorCheck(name="missing", ok=False, detail="missing"),
        ]
    )

    assert summary["ok"] is False
    assert summary["checks"][1]["name"] == "missing"


def test_doctor_summary_ignores_failed_optional_checks():
    summary = doctor_summary(
        [
            DoctorCheck(name="required", ok=True, detail="ok"),
            DoctorCheck(name="optional", ok=False, detail="missing", required=False),
        ]
    )

    assert summary["ok"] is True
    assert summary["checks"][1]["required"] is False


def test_doctor_requires_postgres_dsn_only_for_postgres_backend():
    sqlite_checks = run_doctor_checks(
        HarnessSettings(
            storage_backend="sqlite",
            model_provider="none",
            postgres_dsn=None,
            openai_api_key=None,
        )
    )
    postgres_checks = run_doctor_checks(
        HarnessSettings(
            storage_backend="postgres",
            model_provider="none",
            postgres_dsn=None,
            openai_api_key=None,
        )
    )

    sqlite_summary = doctor_summary(sqlite_checks)
    postgres_summary = doctor_summary(postgres_checks)

    assert sqlite_summary["ok"] is True
    postgres_dsn = next(check for check in postgres_checks if check.name == "postgres_dsn")
    assert postgres_dsn.required is True
    assert postgres_dsn.ok is False
    assert postgres_summary["ok"] is False


def test_doctor_reports_unknown_configured_backends():
    checks = run_doctor_checks(
        HarnessSettings(
            storage_backend="typo",
            model_provider="missing",
            embedding_provider="unknown",
            secret_backend="vault",
            openai_api_key=None,
        )
    )
    summary = doctor_summary(checks)

    assert summary["ok"] is False
    failed = {check.name: check for check in checks if not check.ok}
    assert "Unknown storage_backend" in failed["storage_backend"].detail
    assert "Unknown model_provider" in failed["model_provider"].detail
    assert "Unknown embedding_provider" in failed["embedding_provider"].detail
    assert "Unknown secret_backend" in failed["secret_backend"].detail


def test_doctor_uses_running_python_executable(monkeypatch):
    monkeypatch.setenv("PATH", "")

    checks = run_doctor_checks(HarnessSettings(model_provider="none", openai_api_key=None))

    python = next(check for check in checks if check.name == "python")
    assert python.ok is True
    assert python.required is True


def test_doctor_reports_missing_minilm_extra(monkeypatch):
    import harness.doctor as doctor_module

    monkeypatch.setattr(doctor_module, "find_spec", lambda name: None)

    checks = run_doctor_checks(
        HarnessSettings(
            model_provider="none",
            openai_api_key=None,
            embedding_provider="minilm",
        )
    )

    embedding = next(check for check in checks if check.name == "embedding_dependencies")
    assert embedding.ok is False
    assert "extra embeddings" in embedding.detail


def test_doctor_requires_openai_key_for_openai_embeddings():
    checks = run_doctor_checks(
        HarnessSettings(
            model_provider="none",
            openai_api_key=None,
            embedding_provider="openai",
        )
    )
    summary = doctor_summary(checks)

    openai_key = next(check for check in checks if check.name == "openai_api_key")
    assert openai_key.required is True
    assert openai_key.ok is False
    assert summary["ok"] is False


def test_doctor_skips_embedding_checks_when_memory_disabled(monkeypatch):
    import harness.doctor as doctor_module

    monkeypatch.setattr(doctor_module, "find_spec", lambda name: None)

    checks = run_doctor_checks(
        HarnessSettings(
            model_provider="none",
            openai_api_key=None,
            memory_enabled=False,
            embedding_provider="unknown",
        )
    )
    summary = doctor_summary(checks)

    assert summary["ok"] is True
    assert all(check.name != "embedding_dependencies" for check in checks)
    assert all(check.name != "embedding_provider" for check in checks)


def test_doctor_reports_empty_codex_command():
    checks = run_doctor_checks(
        HarnessSettings(
            model_provider="codex",
            codex_command=[],
        )
    )

    codex = next(check for check in checks if check.name == "codex")
    assert codex.ok is False
    assert codex.required is True
