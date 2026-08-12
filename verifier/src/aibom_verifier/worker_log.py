"""Worker-side stderr JSONL for one node job (Compose queue + SSH run_test)."""

from __future__ import annotations

import os
from time import perf_counter_ns
from uuid import uuid4

from aibom_verifier.observer import StderrJsonlObserver, safe_on_event
from aibom_verifier.registry import DEFAULT_REGISTRY
from aibom_verifier.run_log import configure_logging, elapsed_ms, set_run_id
from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.slots.worker import NodeFn, run_one_node
from aibom_verifier.types import TestOutcome

WORKER_LOGGER = "aibom_verifier.worker"
RUN_TEST_LOGGER = "aibom_verifier.run_test"


def resolve_run_id(job_run_id: str | None = None) -> str:
    if job_run_id is not None:
        job_run_id = job_run_id.strip()
    if job_run_id:
        return job_run_id
    env = os.environ.get("AIBOM_RUN_ID", "").strip()
    if env:
        return env
    return str(uuid4())


def init_worker_logging(run_id: str) -> StderrJsonlObserver:
    set_run_id(run_id)
    log_level = os.environ.get("AIBOM_LOG_LEVEL", "INFO")
    try:
        configure_logging(log_level)
    except ValueError:
        configure_logging("INFO")
    return StderrJsonlObserver()


def run_node_logged(
    node_id: str,
    inputs: dict,
    *,
    store: ArtifactStore,
    registry: dict[str, NodeFn] | None = None,
    logger: str,
    observer: StderrJsonlObserver | None = None,
) -> TestOutcome:
    ob = observer or StderrJsonlObserver()
    safe_on_event(ob, "test_started", logger=logger, test_id=node_id, mode="execute")
    start_ns = perf_counter_ns()
    outcome = run_one_node(
        node_id,
        inputs,
        store=store,
        registry=registry or DEFAULT_REGISTRY,
    )
    finished_fields: dict[str, object] = {
        "test_id": outcome.test_id,
        "status": outcome.status,
        "reason_codes": list(outcome.reason_codes),
        "duration_ms": elapsed_ms(start_ns),
    }
    if "node_exception" in outcome.reason_codes:
        detail = outcome.detail or {}
        safe_on_event(
            ob,
            "exception",
            logger=logger,
            test_id=outcome.test_id,
            exception_type=detail.get("exception_type", "Exception"),
            message=detail.get("message", ""),
        )
    safe_on_event(ob, "test_finished", logger=logger, **finished_fields)
    return outcome
