"""Compose-queue ExecutionBackend (fakeredis; no real Redis required)."""

from __future__ import annotations

import json
import threading

import fakeredis
import pytest

from aibom_verifier.backends import compose_queue as cq
from aibom_verifier.backends.compose_queue import (
    ComposeQueueBackend,
    process_next_job,
    worker_main,
)
from aibom_verifier.slots import worker as worker_slot
from aibom_verifier.slots.artifact_store import InMemoryArtifactStore
from aibom_verifier.types import TestOutcome


def _stub_registry():
    def stub_node(inputs: dict, store) -> TestOutcome:
        store.put("seen", b"1")
        return TestOutcome(
            test_id="stub_node",
            status="pass",
            detail={"echo": inputs.get("value"), "has_api": "api" in inputs},
        )

    return {"stub_node": stub_node}


def test_compose_queue_enqueue_worker_returns_outcome(tmp_path, monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    registry = _stub_registry()
    monkeypatch.setattr(worker_slot, "HfApi", lambda: object())

    thread = threading.Thread(
        target=lambda: worker_main(
            client=client,
            registry=registry,
            max_jobs=1,
            brpop_timeout=1,
        ),
        daemon=True,
    )
    thread.start()

    backend = ComposeQueueBackend(
        client=client,
        store_dir=str(tmp_path),
        store="filesystem",
        timeout=5,
    )
    result = backend.run(
        "stub_node",
        {"value": 7, "api": "must-not-serialize"},
        InMemoryArtifactStore(),
    )
    thread.join(timeout=5)
    assert not thread.is_alive()

    assert result.test_id == "stub_node"
    assert result.status == "pass"
    assert result.detail == {"echo": 7, "has_api": True}
    assert (tmp_path / "seen").read_bytes() == b"1"


def test_compose_queue_job_payload_omits_api_and_carries_store_config():
    raw = cq._encode_job(
        job_id="j1",
        node_id="stub_node",
        inputs={"value": 1, "api": object()},
        store_config=cq._store_config(
            store="proxy",
            store_dir="/cache",
            ignore_cache=True,
        ),
    )
    job = json.loads(raw)
    assert job["node_id"] == "stub_node"
    assert "api" not in job["inputs"]
    assert job["inputs"]["value"] == 1
    assert job["store_config"] == {
        "store": "proxy",
        "store_dir": "/cache",
        "ignore_cache": True,
    }
    assert job["result_key"] == "aibom:result:j1"


def test_compose_queue_timeout_raises():
    client = fakeredis.FakeRedis(decode_responses=True)
    backend = ComposeQueueBackend(client=client, timeout=1)
    with pytest.raises(RuntimeError, match="timed out"):
        backend.run("stub_node", {}, InMemoryArtifactStore())


def test_process_next_job_pushes_error_envelope(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)

    def _raise(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(cq, "process_job", _raise)
    client.lpush(
        "aibom:jobs",
        cq._encode_job(
            job_id="j1",
            node_id="x",
            inputs={},
            store_config={},
        ),
    )
    assert process_next_job(client, timeout=1) is True
    item = client.brpop("aibom:result:j1", timeout=1)
    assert item is not None
    payload = json.loads(item[1])
    assert payload["ok"] is False
    assert payload["error"] == "worker_failed"
    assert "boom" in payload["message"]


def test_process_next_job_poison_json_keeps_looping():
    client = fakeredis.FakeRedis(decode_responses=True)
    client.lpush("aibom:jobs", "not-json")
    assert process_next_job(client, timeout=1) is True
    assert client.llen("aibom:jobs") == 0


def test_compose_queue_backend_raises_on_non_json_result():
    client = fakeredis.FakeRedis(decode_responses=True)

    def bad_result_worker() -> None:
        item = client.brpop("aibom:jobs", timeout=2)
        assert item is not None
        job = json.loads(item[1])
        client.lpush(job["result_key"], "not-json")

    thread = threading.Thread(target=bad_result_worker, daemon=True)
    thread.start()
    backend = ComposeQueueBackend(client=client, timeout=5)
    with pytest.raises(RuntimeError, match="non-JSON"):
        backend.run("stub_node", {}, InMemoryArtifactStore())
    thread.join(timeout=5)


def test_process_next_job_ignores_spoofed_result_key(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)

    def _ok(job, *, registry=None):
        return {"ok": True, "outcome": {"test_id": "x", "status": "pass"}}

    monkeypatch.setattr(cq, "process_job", _ok)
    client.lpush(
        "aibom:jobs",
        json.dumps(
            {
                "job_id": "real-id",
                "node_id": "x",
                "inputs": {},
                "store_config": {},
                "result_key": "aibom:result:spoofed",
            }
        ),
    )
    assert process_next_job(client, timeout=1) is True
    assert client.brpop("aibom:result:spoofed", timeout=1) is None
    item = client.brpop("aibom:result:real-id", timeout=1)
    assert item is not None


def test_connect_redis_sets_socket_timeout_none():
    # redis-py 8 Connection defaults to 5s when socket_timeout is omitted from
    # from_url; assert the live connection object, not just kwargs.
    client = cq.connect_redis("redis://example:6379/1")
    conn = client.connection_pool.make_connection()
    assert conn.socket_timeout is None
    assert conn.socket_connect_timeout == cq._REDIS_CONNECT_TIMEOUT


def test_compose_queue_backend_raises_on_worker_error():
    client = fakeredis.FakeRedis(decode_responses=True)

    def fail_worker() -> None:
        item = client.brpop("aibom:jobs", timeout=2)
        assert item is not None
        job = json.loads(item[1])
        client.lpush(
            job["result_key"],
            json.dumps({"ok": False, "error": "worker_failed", "message": "nope"}),
        )

    thread = threading.Thread(target=fail_worker, daemon=True)
    thread.start()
    backend = ComposeQueueBackend(client=client, timeout=5)
    with pytest.raises(RuntimeError, match="nope"):
        backend.run("stub_node", {}, InMemoryArtifactStore())
    thread.join(timeout=5)
