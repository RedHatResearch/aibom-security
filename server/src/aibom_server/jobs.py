"""In-process job queue: one ``aibom verify`` subprocess per submitted model id.

Each job gets a run directory under ``runs_dir`` named by its ``run_id``:

- ``job.json`` — submission metadata and status (``queued`` → ``running`` →
  ``done`` / ``error``)
- ``telemetry.jsonl`` — the subprocess stderr (JSONL envelope, see
  ``docs/job-contract.md``)
- ``result.json`` — the subprocess stdout (``VerificationResult`` JSON)

The subprocess runs ``python -m aibom verify <model_id> --accept`` with
``AIBOM_RUN_ID`` set to the job's run id, so every telemetry line carries the
same correlation id. Accept mode keeps start failures (missing ``base_model``
claim, gated repo, unknown repo) as logged ``done`` runs with a
non-confirming verdict instead of errors. The environment is inherited, so
``HF_TOKEN``, ``AIBOM_CACHE_DIR``, ``AIBOM_LOG_LEVEL`` and friends pass
through to the verify run unchanged.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import secrets
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_CONCURRENCY = 1
DEFAULT_QUEUE_SIZE = 100
DEFAULT_TIMEOUT_SECONDS = 3600.0

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"


def mint_run_id() -> str:
    """Timestamp-prefixed unique id, filesystem-safe as a directory name."""
    now = datetime.now(UTC)
    return f"{now.strftime('%Y%m%dT%H%M%S%f')}Z-{secrets.token_hex(6)}"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class VerifyJob:
    run_id: str
    model_id: str
    submitted_at: str


class JobRunner:
    """Bounded FIFO queue with daemon worker threads.

    Workers are started in the constructor and die with the process; a job
    killed mid-run keeps ``status: running`` in its ``job.json``.
    """

    def __init__(
        self,
        runs_dir: Path | str,
        *,
        concurrency: int = DEFAULT_CONCURRENCY,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        spawn: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.timeout_seconds = timeout_seconds
        self._spawn = spawn
        self._queue: queue.Queue[VerifyJob] = queue.Queue(maxsize=max(1, queue_size))
        self._pending_lock = threading.Lock()
        self._pending = 0
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._workers = [
            threading.Thread(target=self._work, name=f"aibom-serve-worker-{i}", daemon=True)
            for i in range(max(1, concurrency))
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, model_id: str) -> VerifyJob:
        """Queue a verify run; raises ``queue.Full`` when the queue is full."""
        job = VerifyJob(run_id=mint_run_id(), model_id=model_id, submitted_at=_utc_now_iso())
        run_dir = self._run_dir(job)
        # Persist the queued record before the job is visible to a worker, so
        # job.json always has a single writer at a time.
        self._write_record(run_dir, self._load_record(run_dir, job))
        with self._pending_lock:
            self._pending += 1
        try:
            self._queue.put(job, block=False)
        except queue.Full:
            with self._pending_lock:
                self._pending -= 1
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        return job

    def pending(self) -> int:
        """Jobs queued or running right now."""
        with self._pending_lock:
            return self._pending

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._execute(job)
            finally:
                with self._pending_lock:
                    self._pending -= 1
                self._queue.task_done()

    def _execute(self, job: VerifyJob) -> None:
        run_dir = self._run_dir(job)
        record = self._load_record(run_dir, job)
        record["status"] = STATUS_RUNNING
        record["started_at"] = _utc_now_iso()
        self._write_record(run_dir, record)

        try:
            proc = self._spawn(
                [sys.executable, "-m", "aibom", "verify", job.model_id, "--accept"],
                capture_output=True,
                text=True,
                env=self._subprocess_env(job),
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            self._finish_error(run_dir, record, "verify subprocess timed out")
            return
        except OSError as exc:
            self._finish_error(run_dir, record, f"verify subprocess failed to start: {exc}")
            return

        (run_dir / "telemetry.jsonl").write_text(proc.stderr)
        (run_dir / "result.json").write_text(proc.stdout)
        record["exit_code"] = proc.returncode
        # --accept makes verify exit 0 for start failures too; only crashes are errors.
        record["status"] = STATUS_DONE if proc.returncode == 0 else STATUS_ERROR
        record["finished_at"] = _utc_now_iso()
        self._write_record(run_dir, record)

    def _finish_error(self, run_dir: Path, record: dict[str, object], message: str) -> None:
        record["status"] = STATUS_ERROR
        record["error"] = message
        record["finished_at"] = _utc_now_iso()
        self._write_record(run_dir, record)

    def _subprocess_env(self, job: VerifyJob) -> dict[str, str]:
        return {**os.environ, "AIBOM_RUN_ID": job.run_id}

    def _run_dir(self, job: VerifyJob) -> Path:
        return self.runs_dir / job.run_id

    def _load_record(self, run_dir: Path, job: VerifyJob) -> dict[str, object]:
        record: dict[str, object] = {
            "run_id": job.run_id,
            "model_id": job.model_id,
            "submitted_at": job.submitted_at,
            "status": STATUS_QUEUED,
        }
        job_file = run_dir / "job.json"
        if job_file.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                record.update(json.loads(job_file.read_text()))
        return record

    def _write_record(self, run_dir: Path, record: dict[str, object]) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = run_dir / "job.json.tmp"
        tmp_file.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(tmp_file, run_dir / "job.json")
