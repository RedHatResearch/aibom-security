"""JobRunner behavior with a stubbed verify subprocess."""

from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from aibom_server.jobs import JobRunner, mint_run_id

RESULT_JSON = '{"target": "org/model", "base": null, "verdict": "insufficient_evidence"}'
TELEMETRY_JSONL = (
    '{"event": "run_start", "run_id": "stub"}\n'
    '{"event": "run_finished", "run_id": "stub", "verdict": "insufficient_evidence"}\n'
)

Spawn = Callable[..., subprocess.CompletedProcess]


def _fake_spawn(calls: list[dict[str, Any]]) -> Spawn:
    def spawn(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout=RESULT_JSON, stderr=TELEMETRY_JSONL
        )

    return spawn


def _wait_for_status(run_dir: Path, status: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    record: dict[str, Any] = {}
    while time.monotonic() < deadline:
        job_file = run_dir / "job.json"
        if job_file.exists():
            record = json.loads(job_file.read_text())
            if record.get("status") == status:
                return record
        time.sleep(0.02)
    pytest.fail(f"job never reached status={status}; last record: {record}")


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition never became true")


def test_submit_runs_verify_subprocess_and_writes_run_dir(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    runner = JobRunner(tmp_path / "runs", spawn=_fake_spawn(calls))

    job = runner.submit("org/model")

    run_dir = tmp_path / "runs" / job.run_id
    record = _wait_for_status(run_dir, "done")

    assert len(calls) == 1
    assert calls[0]["argv"] == [
        sys.executable,
        "-m",
        "aibom",
        "verify",
        "org/model",
        "--accept",
    ]
    assert calls[0]["env"]["AIBOM_RUN_ID"] == job.run_id

    assert record["run_id"] == job.run_id
    assert record["model_id"] == "org/model"
    assert record["status"] == "done"
    assert record["exit_code"] == 0
    for field in ("submitted_at", "started_at", "finished_at"):
        assert record[field]

    assert (run_dir / "result.json").read_text() == RESULT_JSON
    assert (run_dir / "telemetry.jsonl").read_text() == TELEMETRY_JSONL


def test_nonzero_exit_records_error_status(tmp_path: Path) -> None:
    def spawn(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=argv, returncode=2, stdout="", stderr="boom\n")

    runner = JobRunner(tmp_path / "runs", spawn=spawn)

    job = runner.submit("org/model")

    record = _wait_for_status(tmp_path / "runs" / job.run_id, "error")
    assert record["exit_code"] == 2


def test_subprocess_timeout_records_error(tmp_path: Path) -> None:
    def spawn(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=0.1)

    runner = JobRunner(tmp_path / "runs", spawn=spawn)

    job = runner.submit("org/model")

    record = _wait_for_status(tmp_path / "runs" / job.run_id, "error")
    assert "timed out" in record["error"]


def test_full_queue_rejects_and_pending_tracks_busy_jobs(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_spawn(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        started.set()
        release.wait(timeout=10)
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="{}", stderr="")

    runner = JobRunner(tmp_path / "runs", queue_size=1, spawn=blocking_spawn)
    try:
        busy = runner.submit("org/busy")
        assert started.wait(timeout=5)
        assert runner.pending() == 1
        runner.submit("org/queued")
        with pytest.raises(queue.Full):
            runner.submit("org/rejected")
    finally:
        release.set()

    _wait_for_status(tmp_path / "runs" / busy.run_id, "done")
    _wait_until(lambda: runner.pending() == 0)


def test_mint_run_id_is_unique_and_filesystem_safe() -> None:
    ids = {mint_run_id() for _ in range(100)}
    assert len(ids) == 100
    for run_id in ids:
        assert re.fullmatch(r"\d{8}T\d+Z-[0-9a-f]{12}", run_id)
