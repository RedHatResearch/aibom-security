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
    assert calls[0]["timeout"] == runner.timeout_seconds
    assert calls[0]["errors"] == "replace"

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


def test_timeout_keeps_partial_telemetry(tmp_path: Path) -> None:
    def spawn(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=0.1, stderr=b'{"event": "run_start"}\n')

    runner = JobRunner(tmp_path / "runs", spawn=spawn)

    job = runner.submit("org/model")

    record = _wait_for_status(tmp_path / "runs" / job.run_id, "error")
    assert "timed out" in record["error"]
    assert (tmp_path / "runs" / job.run_id / "telemetry.jsonl").read_text() == (
        '{"event": "run_start"}\n'
    )


def test_worker_survives_unexpected_exception(tmp_path: Path) -> None:
    state = {"calls": 0}

    def spawn(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        state["calls"] += 1
        if state["calls"] == 1:
            raise ValueError("boom")
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout=RESULT_JSON, stderr=TELEMETRY_JSONL
        )

    runner = JobRunner(tmp_path / "runs", spawn=spawn)

    first = runner.submit("org/model")
    second = runner.submit("org/model")

    record = _wait_for_status(tmp_path / "runs" / first.run_id, "error")
    assert "worker crashed" in record["error"]
    assert _wait_for_status(tmp_path / "runs" / second.run_id, "done")["exit_code"] == 0


def test_post_spawn_write_failure_records_persist_error(tmp_path: Path) -> None:
    def sabotaging_spawn(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        # The running record is on disk before spawn is called; occupy the
        # telemetry path with a directory so the post-spawn write fails.
        run_dir = tmp_path / "runs" / kwargs["env"]["AIBOM_RUN_ID"]
        (run_dir / "telemetry.jsonl").mkdir()
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout=RESULT_JSON, stderr=TELEMETRY_JSONL
        )

    runner = JobRunner(tmp_path / "runs", spawn=sabotaging_spawn)

    job = runner.submit("org/model")

    record = _wait_for_status(tmp_path / "runs" / job.run_id, "error")
    assert "failed to persist run output" in record["error"]
    assert record["started_at"]


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
        busy_record = _wait_for_status(tmp_path / "runs" / busy.run_id, "running")
        assert busy_record["started_at"]
        runner.submit("org/queued")
        with pytest.raises(queue.Full):
            runner.submit("org/rejected")
    finally:
        release.set()

    _wait_for_status(tmp_path / "runs" / busy.run_id, "done")
    _wait_until(lambda: runner.pending() == 0)


def test_multiple_workers_run_jobs_concurrently(tmp_path: Path) -> None:
    release = threading.Event()
    lock = threading.Lock()
    entered: list[threading.Event] = []

    def spawn(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        arrived = threading.Event()
        with lock:
            entered.append(arrived)
        arrived.set()
        release.wait(timeout=10)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout=RESULT_JSON, stderr=TELEMETRY_JSONL
        )

    runner = JobRunner(tmp_path / "runs", concurrency=4, spawn=spawn)
    jobs = [runner.submit(f"org/model-{i}") for i in range(4)]

    _wait_until(lambda: len(entered) == 4)
    with lock:
        assert len(entered) >= 2  # peak parallelism: several workers busy at once

    release.set()
    for job in jobs:
        assert _wait_for_status(tmp_path / "runs" / job.run_id, "done")["exit_code"] == 0


def test_startup_sweep_marks_nonterminal_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    queued = {
        "run_id": "20250101T000000000000Z-000000000001",
        "model_id": "org/queued",
        "submitted_at": "2025-01-01T00:00:00+00:00",
        "status": "queued",
    }
    running = {
        "run_id": "20250101T000000000000Z-000000000002",
        "model_id": "org/running",
        "submitted_at": "2025-01-01T00:00:00+00:00",
        "status": "running",
        "started_at": "2025-01-01T00:00:01+00:00",
    }
    done = {
        "run_id": "20250101T000000000000Z-000000000003",
        "model_id": "org/done",
        "submitted_at": "2025-01-01T00:00:00+00:00",
        "status": "done",
        "started_at": "2025-01-01T00:00:01+00:00",
        "finished_at": "2025-01-01T00:00:02+00:00",
        "exit_code": 0,
    }
    for record in (queued, running, done):
        run_dir = runs / record["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "job.json").write_text(json.dumps(record, indent=2) + "\n")
    done_before = (runs / done["run_id"] / "job.json").read_bytes()

    JobRunner(runs, spawn=_fake_spawn([]))

    for original in (queued, running):
        record = json.loads((runs / original["run_id"] / "job.json").read_text())
        assert record["status"] == "error"
        assert record["error"] == "interrupted by restart"
        assert record["finished_at"]
    assert "started_at" in json.loads((runs / running["run_id"] / "job.json").read_text())
    assert (runs / done["run_id"] / "job.json").read_bytes() == done_before


def test_unwritable_runs_dir_fails_fast(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    runs.chmod(0o555)
    try:
        with pytest.raises(RuntimeError, match="not writable"):
            JobRunner(runs, spawn=_fake_spawn([]))
    finally:
        runs.chmod(0o755)


def test_mint_run_id_is_unique_and_filesystem_safe() -> None:
    ids = {mint_run_id() for _ in range(100)}
    assert len(ids) == 100
    for run_id in ids:
        assert re.fullmatch(r"\d{8}T\d+Z-[0-9a-f]{12}", run_id)
