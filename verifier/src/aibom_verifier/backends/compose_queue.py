"""Compose-queue ExecutionBackend: Redis list + JSON jobs (no RQ).

Client ``LPUSH``es a job and ``BRPOP``s the per-job result list.
Workers ``BRPOP`` the shared queue, rebuild the store from ``store_config`` /
env, run one node, and ``LPUSH`` the result.

Ignores the in-process ``store`` object (same remote contract as SSH).
Never serialize ``api`` into the job payload — the worker builds ``HfApi()``.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from typing import Any

from redis import Redis

from aibom_verifier.registry import DEFAULT_REGISTRY
from aibom_verifier.run_log import get_run_id, resolve_run_id
from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.slots.worker import NodeFn, without_api
from aibom_verifier.store_factory import build_artifact_store
from aibom_verifier.types import TestOutcome, outcome_from_remote_dict
from aibom_verifier.worker_log import (
    WORKER_LOGGER,
    init_worker_logging,
    run_node_logged,
)

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_QUEUE_KEY = "aibom:jobs"
DEFAULT_RESULT_TTL_SECONDS = 3600
_REDIS_CONNECT_TIMEOUT = 5.0


def default_redis_url() -> str:
    return os.environ.get("AIBOM_REDIS_URL", DEFAULT_REDIS_URL)


def default_queue_key() -> str:
    return os.environ.get("AIBOM_REDIS_QUEUE", DEFAULT_QUEUE_KEY)


def connect_redis(url: str | None = None) -> Redis:
    # Connection defaults to 5s when socket_timeout is omitted; BRPOP must outlive that.
    return Redis.from_url(
        url or default_redis_url(),
        decode_responses=True,
        socket_connect_timeout=_REDIS_CONNECT_TIMEOUT,
        socket_timeout=None,
    )


def result_list_key(job_id: str) -> str:
    return f"aibom:result:{job_id}"


def _store_config(
    *,
    store: str | None,
    store_dir: str | None,
    ignore_cache: bool,
) -> dict[str, Any]:
    return {
        "store": store,
        "store_dir": store_dir,
        "ignore_cache": ignore_cache,
    }


def _build_store(store_config: Mapping[str, Any]) -> ArtifactStore:
    return build_artifact_store(
        store=store_config.get("store"),
        cache_dir=store_config.get("store_dir"),
        ignore_cache=bool(store_config.get("ignore_cache", False)),
    )


def _encode_job(
    *,
    job_id: str,
    node_id: str,
    inputs: dict,
    store_config: Mapping[str, Any],
    run_id: str,
) -> str:
    return json.dumps(
        {
            "job_id": job_id,
            "node_id": node_id,
            "inputs": without_api(inputs),
            "store_config": dict(store_config),
            "result_key": result_list_key(job_id),
            "run_id": run_id,
        }
    )


def _push_result(client: Redis, result_key: str, payload: dict[str, Any]) -> None:
    pipe = client.pipeline()
    pipe.lpush(result_key, json.dumps(payload))
    pipe.expire(result_key, DEFAULT_RESULT_TTL_SECONDS)
    pipe.execute()


def _error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "error": "worker_failed",
        "message": str(exc),
        "exception_type": type(exc).__name__,
    }


def _store_cache_key(store_config: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        store_config.get("store"),
        store_config.get("store_dir"),
        bool(store_config.get("ignore_cache", False)),
    )


def _store_for_job(
    store_config: Mapping[str, Any],
    store_cache: dict[tuple[object, ...], ArtifactStore] | None,
) -> ArtifactStore:
    if store_cache is None:
        return _build_store(store_config)
    key = _store_cache_key(store_config)
    store = store_cache.get(key)
    if store is None:
        store = _build_store(store_config)
        store_cache[key] = store
    return store


def process_job(
    job: Mapping[str, Any],
    *,
    registry: dict[str, NodeFn] | None = None,
    store_cache: dict[tuple[object, ...], ArtifactStore] | None = None,
) -> dict[str, Any]:
    """Run one decoded job; return the result envelope (does not touch Redis)."""
    raw_run_id = job.get("run_id")
    job_run_id: str | None = None
    if raw_run_id is not None:
        job_run_id = str(raw_run_id).strip() or None
    run_id = resolve_run_id(job_run_id)
    observer = init_worker_logging(run_id)
    store = _store_for_job(job.get("store_config") or {}, store_cache)
    outcome = run_node_logged(
        job["node_id"],
        dict(job.get("inputs") or {}),
        store=store,
        registry=registry or DEFAULT_REGISTRY,
        logger=WORKER_LOGGER,
        observer=observer,
    )
    return {"ok": True, "outcome": outcome.to_dict()}


def process_next_job(
    client: Redis,
    *,
    queue_key: str | None = None,
    registry: dict[str, NodeFn] | None = None,
    timeout: int = 5,
    store_cache: dict[tuple[object, ...], ArtifactStore] | None = None,
) -> bool:
    """``BRPOP`` one job, run it, push result. Return False on queue timeout."""
    key = queue_key or default_queue_key()
    item = client.brpop(key, timeout=timeout)
    if item is None:
        return False
    _, raw = item
    result_key: str | None = None
    try:
        job = json.loads(raw)
        job_id = job["job_id"]
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        # Always derive from job_id — never trust a client-supplied result_key.
        result_key = result_list_key(job_id)
        payload = process_job(job, registry=registry, store_cache=store_cache)
    except Exception as exc:
        payload = _error_payload(exc)
        if result_key is None:
            # Poison message with no recoverable result key — drop and keep looping.
            return True
    _push_result(client, result_key, payload)
    return True


def worker_main(
    *,
    redis_url: str | None = None,
    client: Redis | None = None,
    queue_key: str | None = None,
    registry: dict[str, NodeFn] | None = None,
    max_jobs: int | None = None,
    brpop_timeout: int = 5,
) -> None:
    """Blocking worker loop.

    ``max_jobs`` limits processed jobs (including failures/poison). Empty-queue
    ``BRPOP`` timeouts do not count. ``None`` runs forever.

    Rebuilds the artifact store at most once per distinct ``store_config``.
    """
    redis_client = client if client is not None else connect_redis(redis_url)
    store_cache: dict[tuple[object, ...], ArtifactStore] = {}
    done = 0
    while max_jobs is None or done < max_jobs:
        if process_next_job(
            redis_client,
            queue_key=queue_key,
            registry=registry,
            timeout=brpop_timeout,
            store_cache=store_cache,
        ):
            done += 1


class ComposeQueueBackend:
    """Enqueue one node on a Redis list and await the worker result."""

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        client: Redis | None = None,
        queue_key: str | None = None,
        store_dir: str | None = None,
        store: str | None = None,
        ignore_cache: bool = False,
        timeout: int = 600,
    ) -> None:
        self._client = client if client is not None else connect_redis(redis_url)
        self._queue_key = queue_key or default_queue_key()
        self._store_dir = store_dir
        self._store = store
        self._ignore_cache = ignore_cache
        self._timeout = timeout

    def run(self, node_id: str, inputs: dict, store: ArtifactStore) -> TestOutcome:
        _ = store  # remotes rebuild from store_config / env
        job_id = uuid.uuid4().hex
        result_key = result_list_key(job_id)
        job = _encode_job(
            job_id=job_id,
            node_id=node_id,
            inputs=inputs,
            store_config=_store_config(
                store=self._store,
                store_dir=self._store_dir,
                ignore_cache=self._ignore_cache,
            ),
            run_id=get_run_id(),
        )
        self._client.lpush(self._queue_key, job)
        item = self._client.brpop(result_key, timeout=self._timeout)
        if item is None:
            raise RuntimeError(
                f"compose queue timed out waiting for job {job_id} ({self._timeout}s)"
            )
        _, raw = item
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"compose queue returned non-JSON result: {exc}") from exc
        if not payload.get("ok"):
            message = payload.get("message") or payload.get("error") or "worker failed"
            raise RuntimeError(f"compose queue job failed: {message}")
        return outcome_from_remote_dict(
            payload.get("outcome"),
            context="compose queue result",
        )


__all__ = [
    "ComposeQueueBackend",
    "connect_redis",
    "default_queue_key",
    "default_redis_url",
    "process_job",
    "process_next_job",
    "result_list_key",
    "worker_main",
]
