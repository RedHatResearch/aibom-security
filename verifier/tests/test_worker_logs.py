"""Structured stderr JSONL for Compose/SSH worker processes (FR-L2)."""

from __future__ import annotations

import json
from uuid import UUID

import pytest
from hf_fakes import read_stderr_jsonl

from aibom_verifier import run_test as run_test_mod
from aibom_verifier.backends import compose_queue as cq
from aibom_verifier.slots import worker as worker_slot
from aibom_verifier.types import TestOutcome

_ENVELOPE_KEYS = frozenset(
    {"ts", "level", "logger", "event", "run_id", "policy_version", "tool_version"}
)

_SHARED_RUN_ID = "12345678-1234-5678-1234-567812345678"


def _stub_registry():
    def stub_node(inputs: dict, store) -> TestOutcome:
        store.put("seen", b"weight-bytes")
        return TestOutcome(
            test_id="stub_node",
            status="pass",
            detail={"echo": inputs.get("value"), "has_api": "api" in inputs},
        )

    return {"stub_node": stub_node}


def _node_exception_registry():
    def exploding(inputs: dict, store) -> TestOutcome:
        del inputs, store
        return TestOutcome(
            test_id="stub_node",
            status="error",
            reason_codes=["node_exception"],
            detail={
                "message": "probe exploded",
                "exception_type": "ValueError",
                "secret": "hide-me",
                "token": "hf-secret-token",
            },
        )

    return {"stub_node": exploding}


def test_process_job_emits_worker_stderr_jsonl_with_shared_run_id(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(worker_slot, "HfApi", lambda: object())
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)

    job = json.loads(
        cq._encode_job(
            job_id="j1",
            node_id="stub_node",
            inputs={"value": 9, "api": object(), "hf_token": "hide-me"},
            store_config=cq._store_config(
                store="filesystem",
                store_dir=str(tmp_path),
                ignore_cache=False,
            ),
            run_id=_SHARED_RUN_ID,
        )
    )

    payload = cq.process_job(job, registry=_stub_registry())
    captured = capsys.readouterr()

    assert payload["ok"] is True
    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]

    assert [row["event"] for row in stderr_payloads] == [
        "test_started",
        "test_finished",
    ]
    assert {row["logger"] for row in stderr_payloads} == {"aibom_verifier.worker"}
    run_ids = {str(row["run_id"]) for row in stderr_payloads}
    assert run_ids == {_SHARED_RUN_ID}
    UUID(_SHARED_RUN_ID)

    started = stderr_payloads[0]
    assert started["test_id"] == "stub_node"
    assert started["mode"] == "execute"
    assert started["level"] == "INFO"

    finished = stderr_payloads[1]
    assert finished["test_id"] == "stub_node"
    assert finished["status"] == "pass"
    assert finished["reason_codes"] == []
    assert isinstance(finished["duration_ms"], int)
    assert "cache" not in finished
    assert "detail" not in finished

    raw_stderr = captured.err
    assert "hide-me" not in raw_stderr
    assert "hf-secret-token" not in raw_stderr
    assert "weight-bytes" not in raw_stderr


def test_process_job_node_exception_emits_exception_between_started_and_finished(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(worker_slot, "HfApi", lambda: object())
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)

    job = json.loads(
        cq._encode_job(
            job_id="j2",
            node_id="stub_node",
            inputs={"value": 1},
            store_config=cq._store_config(
                store="filesystem",
                store_dir=str(tmp_path),
                ignore_cache=False,
            ),
            run_id=_SHARED_RUN_ID,
        )
    )

    cq.process_job(job, registry=_node_exception_registry())
    stderr_payloads = read_stderr_jsonl(capsys)

    assert [row["event"] for row in stderr_payloads] == [
        "test_started",
        "exception",
        "test_finished",
    ]
    exception = stderr_payloads[1]
    assert exception["level"] == "ERROR"
    assert exception["test_id"] == "stub_node"
    assert exception["exception_type"] == "ValueError"
    assert exception["message"] == "probe exploded"
    assert "hide-me" not in json.dumps(exception)
    assert "hf-secret-token" not in json.dumps(exception)


def test_run_test_main_emits_run_test_logger_jsonl_with_env_run_id(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(worker_slot, "HfApi", lambda: object())
    monkeypatch.setenv("AIBOM_RUN_ID", _SHARED_RUN_ID)
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)

    code = run_test_mod.main(
        [
            "--node-id",
            "stub_node",
            "--inputs-json",
            json.dumps({"value": 42, "api": "strip-me", "config": {"token": "hide-me"}}),
            "--store-dir",
            str(tmp_path),
        ],
        registry=_stub_registry(),
    )
    assert code == 0

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    stderr_payloads = [json.loads(line) for line in captured.err.splitlines() if line]

    assert stdout_payload["test_id"] == "stub_node"
    assert stdout_payload["status"] == "pass"
    assert [row["event"] for row in stderr_payloads] == [
        "test_started",
        "test_finished",
    ]
    assert {row["logger"] for row in stderr_payloads} == {"aibom_verifier.run_test"}
    assert {str(row["run_id"]) for row in stderr_payloads} == {_SHARED_RUN_ID}
    assert "hide-me" not in captured.err
    assert "strip-me" not in captured.err


def test_process_job_without_run_id_mints_worker_run_id(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(worker_slot, "HfApi", lambda: object())
    monkeypatch.delenv("AIBOM_RUN_ID", raising=False)
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)

    job = {
        "job_id": "legacy",
        "node_id": "stub_node",
        "inputs": {"value": 1},
        "store_config": cq._store_config(
            store="filesystem",
            store_dir=str(tmp_path),
            ignore_cache=False,
        ),
    }
    cq.process_job(job, registry=_stub_registry())
    stderr_payloads = read_stderr_jsonl(capsys)

    run_ids = {str(row["run_id"]) for row in stderr_payloads}
    assert len(run_ids) == 1
    UUID(run_ids.pop())


def test_worker_log_fields_stay_within_allowed_set(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(worker_slot, "HfApi", lambda: object())
    monkeypatch.delenv("AIBOM_LOG_LEVEL", raising=False)

    job = json.loads(
        cq._encode_job(
            job_id="j3",
            node_id="stub_node",
            inputs={"value": 1},
            store_config=cq._store_config(
                store="filesystem",
                store_dir=str(tmp_path),
                ignore_cache=False,
            ),
            run_id=_SHARED_RUN_ID,
        )
    )
    cq.process_job(job, registry=_stub_registry())
    stderr_payloads = read_stderr_jsonl(capsys)

    allowed_fields = _ENVELOPE_KEYS | {
        "test_id",
        "mode",
        "status",
        "reason_codes",
        "duration_ms",
        "exception_type",
        "message",
    }
    for payload in stderr_payloads:
        assert set(payload) <= allowed_fields
