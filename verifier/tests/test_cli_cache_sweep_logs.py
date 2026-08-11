"""Structured stderr JSONL logging for ``aibom cache-sweep`` (issue #30).

Mirrors ``test_cli_logs.py`` capture patterns. Local ``_parse`` / ``_aged_proxy``
helpers keep ``test_cache_sweep.py`` focused on stdout/exit coverage.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from hf_fakes import read_stderr_jsonl

from aibom_verifier import cli_cache_sweep as sweep_cli
from aibom_verifier.slots.proxy_store import (
    ArtifactMeta,
    InMemoryBlobBackend,
    InMemoryMetadataBackend,
    ProxyArtifactStore,
)

LOGGER_NAME = "aibom_verifier.cli_cache_sweep"
_ENVELOPE_KEYS = frozenset(
    {"ts", "level", "logger", "event", "run_id", "policy_version", "tool_version"}
)


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sweep_cli.add_arguments(parser)
    return parser.parse_args(argv)


def _with_lat(row: ArtifactMeta, last_accessed_at: datetime) -> ArtifactMeta:
    return ArtifactMeta(
        key=row.key,
        blob_object=row.blob_object,
        size_bytes=row.size_bytes,
        created_at=row.created_at,
        last_accessed_at=last_accessed_at,
    )


def _aged_proxy(*, age: timedelta) -> ProxyArtifactStore:
    meta = InMemoryMetadataBackend()
    blobs = InMemoryBlobBackend()
    store = ProxyArtifactStore(meta, blobs)
    store.put("stale", b"old")
    row = meta.get("stale")
    assert row is not None
    meta.put(_with_lat(row, datetime.now(UTC) - age))
    return store


def _assert_envelope(payload: dict[str, object]) -> None:
    assert set(payload) >= _ENVELOPE_KEYS
    assert payload["logger"] == LOGGER_NAME
    UUID(str(payload["run_id"]))


def test_sweep_happy_path_emits_sweep_start_then_sweep_finished_with_shared_run_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
):
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)
    args = _parse(["--cache-dir", str(tmp_path)])

    assert sweep_cli.run(args) == 0

    stderr_payloads = read_stderr_jsonl(capsys)

    assert [payload["event"] for payload in stderr_payloads] == [
        "sweep_start",
        "sweep_finished",
    ]
    for payload in stderr_payloads:
        _assert_envelope(payload)
    run_ids = {str(payload["run_id"]) for payload in stderr_payloads}
    assert len(run_ids) == 1


def test_sweep_finished_fields_and_stdout_shape_with_proxy_store(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)
    proxy = _aged_proxy(age=timedelta(days=40))
    monkeypatch.setattr(sweep_cli, "build_artifact_store", lambda **kw: proxy)
    args = _parse(["--max-age-days", "30", "--store", "proxy"])

    assert sweep_cli.run(args) == 0

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    assert stdout_payload == {"deleted": 1, "max_age_days": 30}

    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]
    assert [payload["event"] for payload in stderr_payloads] == [
        "sweep_start",
        "sweep_finished",
    ]

    sweep_start = stderr_payloads[0]
    assert sweep_start["max_age_days"] == 30
    assert sweep_start["store"] == "proxy"

    sweep_finished = stderr_payloads[1]
    assert sweep_finished["max_age_days"] == 30
    assert sweep_finished["store"] == "proxy"
    assert sweep_finished["deleted"] == 1
    assert sweep_finished["exit_code"] == 0
    assert isinstance(sweep_finished["duration_ms"], int)


def test_negative_max_age_days_emits_single_sweep_failed_and_preserves_stdout_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)
    monkeypatch.setattr(
        sweep_cli,
        "build_artifact_store",
        lambda **kw: pytest.fail("build_artifact_store should not run for invalid max_age_days"),
    )
    args = _parse(["--max-age-days", "-1"])

    assert sweep_cli.run(args) == 1

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    assert stdout_payload == {
        "ok": False,
        "error": "invalid_max_age_days",
        "message": "max_age_days must be >= 0",
    }

    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]
    assert [payload["event"] for payload in stderr_payloads] == ["sweep_failed"]
    sweep_failed = stderr_payloads[0]
    _assert_envelope(sweep_failed)
    assert sweep_failed["error_code"] == "invalid_max_age_days"
    assert sweep_failed["exit_code"] == 1
    assert isinstance(sweep_failed["duration_ms"], int)


def test_invalid_log_level_emits_sweep_failed_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("AIBOM_LOG_LEVEL", "TRACE")
    monkeypatch.setattr(
        sweep_cli,
        "build_artifact_store",
        lambda **kw: pytest.fail("build_artifact_store should not run for invalid log level"),
    )
    args = _parse([])

    assert sweep_cli.run(args) == 1

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    assert stdout_payload["ok"] is False
    assert stdout_payload["error"] == "invalid_log_level"

    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]
    assert [payload["event"] for payload in stderr_payloads] == ["sweep_failed"]
    sweep_failed = stderr_payloads[0]
    _assert_envelope(sweep_failed)
    assert sweep_failed["error_code"] == "invalid_log_level"
    assert sweep_failed["exit_code"] == 1


def test_error_log_level_suppresses_info_events_for_successful_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
):
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)
    info_args = _parse(["--cache-dir", str(tmp_path)])
    assert sweep_cli.run(info_args) == 0
    info_payloads = read_stderr_jsonl(capsys)
    assert [payload["event"] for payload in info_payloads] == [
        "sweep_start",
        "sweep_finished",
    ]

    monkeypatch.setenv("AIBOM_LOG_LEVEL", "ERROR")
    error_args = _parse(["--cache-dir", str(tmp_path)])
    assert sweep_cli.run(error_args) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    stdout_payload = json.loads(captured.out)
    assert stdout_payload == {"deleted": 0, "max_age_days": 30}


def test_secrets_are_never_logged_in_sweep_events(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)
    monkeypatch.setenv("AIBOM_PG_DSN", "postgresql://sweeper:S3cr3tPass@db.internal:5432/aibom")
    monkeypatch.setenv("AIBOM_MINIO_ACCESS_KEY", "AKIAFAKEACCESSKEYID")
    monkeypatch.setenv("AIBOM_MINIO_SECRET_KEY", "fake-minio-secret-value")

    proxy = _aged_proxy(age=timedelta(days=40))
    monkeypatch.setattr(sweep_cli, "build_artifact_store", lambda **kw: proxy)
    args = _parse(["--max-age-days", "30", "--store", "proxy"])

    assert sweep_cli.run(args) == 0

    captured = capsys.readouterr()
    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]
    assert [payload["event"] for payload in stderr_payloads] == [
        "sweep_start",
        "sweep_finished",
    ]

    raw_stderr = captured.err
    for secret in (
        "S3cr3tPass",
        "AKIAFAKEACCESSKEYID",
        "fake-minio-secret-value",
        "postgresql://",
    ):
        assert secret not in raw_stderr

    allowed_fields = _ENVELOPE_KEYS | {
        "max_age_days",
        "store",
        "deleted",
        "duration_ms",
        "exit_code",
    }
    for payload in stderr_payloads:
        assert set(payload) <= allowed_fields
        assert "stale" not in json.dumps(payload)
