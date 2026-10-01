"""HTTP surface of the verify server (stubbed verify subprocess)."""

from __future__ import annotations

import json
import queue
import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from aibom_server.app import create_app
from aibom_server.jobs import JobRunner

RESULT_JSON = '{"target": "org/model", "base": null, "verdict": "insufficient_evidence"}'


def _client(tmp_path: Path, *, token: str | None = None) -> tuple[TestClient, JobRunner]:
    def spawn(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=RESULT_JSON, stderr="")

    runner = JobRunner(tmp_path / "runs", spawn=spawn)
    return TestClient(create_app(runner=runner, token=token)), runner


def _wait_for_run(tmp_path: Path, run_id: str, timeout: float = 5.0) -> dict[str, Any]:
    run_dir = tmp_path / "runs" / run_id
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job_file = run_dir / "job.json"
        if job_file.exists() and json.loads(job_file.read_text()).get("status") == "done":
            return json.loads(job_file.read_text())
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never finished")


def test_post_verify_returns_202_and_runs_job_to_disk(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)

    response = client.post("/verify", json={"model_id": "org/model"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"

    record = _wait_for_run(tmp_path, body["run_id"])
    assert record["model_id"] == "org/model"
    result = json.loads((tmp_path / "runs" / body["run_id"] / "result.json").read_text())
    assert result["verdict"] == "insufficient_evidence"


def test_post_verify_rejects_malformed_model_ids(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    for bad in ["", "   ", "org /model", "/model", "org//model", "a/b/c", "org/model?x"]:
        assert client.post("/verify", json={"model_id": bad}).status_code == 422, bad
    assert client.post("/verify", json={}).status_code == 422
    assert client.post("/verify", json={"model_id": 123}).status_code == 422


def test_healthz_reports_ok_and_pending(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)

    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["pending"], int)


def test_bearer_token_gates_post_verify_only(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, token="s3cret")

    assert client.post("/verify", json={"model_id": "org/model"}).status_code == 401
    wrong = client.post(
        "/verify", json={"model_id": "org/model"}, headers={"Authorization": "Bearer nope"}
    )
    assert wrong.status_code == 401
    ok = client.post(
        "/verify", json={"model_id": "org/model"}, headers={"Authorization": "Bearer s3cret"}
    )
    assert ok.status_code == 202
    assert client.get("/healthz").status_code == 200


def test_post_verify_maps_full_queue_to_503(tmp_path: Path) -> None:
    client, runner = _client(tmp_path)

    def full_submit(model_id: str) -> None:
        raise queue.Full()

    runner.submit = full_submit  # type: ignore[method-assign]
    response = client.post("/verify", json={"model_id": "org/model"})

    assert response.status_code == 503
