"""HTTP surface of the verify server (stubbed verify subprocess)."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from aibom_server.app import create_app
from aibom_server.jobs import JobRunner

RESULT_JSON = '{"target": "org/model", "base": null, "verdict": "insufficient_evidence"}'

Spawn = Callable[..., subprocess.CompletedProcess]


def _stub_spawn(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=argv, returncode=0, stdout=RESULT_JSON, stderr="")


def _client(
    tmp_path: Path, *, token: str | None = None, spawn: Spawn | None = None
) -> tuple[TestClient, JobRunner]:
    runner = JobRunner(tmp_path / "runs", spawn=spawn or _stub_spawn)
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


def _wait_for_pending(client: TestClient, pending: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/healthz").json()["pending"] == pending:
            return
        time.sleep(0.02)
    raise AssertionError(f"healthz never reported pending={pending}")


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
    bad = [
        "",
        "   ",
        "org /model",
        "/model",
        "org//model",
        "a/b/c",
        "org/model?x",
        # Flag-shaped and dot-prefixed ids the verify subprocess must never
        # receive: argparse would treat them as options, not model ids.
        "--help",
        "-h",
        "--",
        "--accept",
        "-x",
        "-org/model",
        ".hidden",
        "..",
        "org/-model",
        "org/.model",
    ]
    for model_id in bad:
        assert client.post("/verify", json={"model_id": model_id}).status_code == 422, model_id
    assert client.post("/verify", json={}).status_code == 422
    assert client.post("/verify", json={"model_id": 123}).status_code == 422


def test_post_verify_accepts_valid_model_ids(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)

    for model_id in ["gpt2", "org/model-name.v2", "Org/Model_1"]:
        assert client.post("/verify", json={"model_id": model_id}).status_code == 202, model_id


def test_healthz_reports_ok_and_pending(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)

    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["pending"], int)


def test_healthz_pending_reflects_runner(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_spawn(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        started.set()
        release.wait(timeout=10)
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=RESULT_JSON, stderr="")

    client, _ = _client(tmp_path, spawn=blocking_spawn)

    response = client.post("/verify", json={"model_id": "org/model"})

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert started.wait(timeout=5)
    try:
        _wait_for_pending(client, 1)
    finally:
        release.set()
    _wait_for_pending(client, 0)

    record = _wait_for_run(tmp_path, run_id)
    assert record["model_id"] == "org/model"


def test_bearer_token_gates_post_verify_only(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, token="s3cret")

    assert client.post("/verify", json={"model_id": "org/model"}).status_code == 401
    for header in ["s3cret", "Basic abc", "Bearer", "bearer s3cret"]:
        wrong = client.post(
            "/verify", json={"model_id": "org/model"}, headers={"Authorization": header}
        )
        assert wrong.status_code == 401, header
    ok = client.post(
        "/verify",
        json={"model_id": "org/model"},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert ok.status_code == 202
    assert client.get("/healthz").status_code == 200


def test_gate_rejects_before_body_read(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, token="s3cret")

    # Auth runs before the body is parsed: an unauthenticated malformed body
    # gets 401, not 422.
    assert client.post("/verify", content=b"{bad").status_code == 401
    # With valid auth the same body reaches the JSON parser and is 422.
    authed = client.post("/verify", content=b"{bad", headers={"Authorization": "Bearer s3cret"})
    assert authed.status_code == 422
    # The body cap is checked before auth and parsing, with or without a token.
    assert client.post("/verify", json={"model_id": "x" * 5000}).status_code == 413
    oversized = client.post(
        "/verify",
        json={"model_id": "x" * 5000},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert oversized.status_code == 413
    # A valid small request without auth is still 401.
    assert client.post("/verify", json={"model_id": "org/model"}).status_code == 401


def test_post_verify_maps_full_queue_to_503(tmp_path: Path) -> None:
    client, runner = _client(tmp_path)

    def full_submit(model_id: str) -> None:
        raise queue.Full()

    runner.submit = full_submit  # type: ignore[method-assign]
    response = client.post("/verify", json={"model_id": "org/model"})

    assert response.status_code == 503
