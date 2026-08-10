"""Structured orchestrator log coverage for Task 3."""

from __future__ import annotations

import pytest

from aibom_verifier.observer import RecordingObserver
from aibom_verifier.orchestrator import run_test_run
from aibom_verifier.rules import Requirement, Rule
from aibom_verifier.slots.artifact_store import InMemoryArtifactStore
from aibom_verifier.types import CompareStartError, TestOutcome


def _bootstrap_resolve_refs(inputs: dict, store) -> TestOutcome:
    del inputs
    store.put("resolve-hit", b"cached")
    store.get("resolve-hit")
    store.get("resolve-miss")
    return TestOutcome(
        test_id="resolve_refs",
        status="pass",
        detail={
            "target_sha": "target-sha",
            "base_sha": "base-sha",
            "base_repo": "org/base",
            "base_source": "cli",
            "target_config": {"model_type": "llama", "secret": "hide-me"},
            "base_config": {"model_type": "llama", "token": "hide-me"},
        },
    )


def _resolve_compare_start_error(inputs: dict, store) -> TestOutcome:
    del inputs, store
    raise CompareStartError("gated_unauthenticated", "no access")


def _support_pass(inputs: dict, store) -> TestOutcome:
    del inputs, store
    return TestOutcome(
        test_id="support_classify",
        status="pass",
        detail={"support_class": "dense_supported"},
    )


def _node_exception(inputs: dict, store) -> TestOutcome:
    del inputs
    store.put("node-hit", b"cached")
    store.get("node-hit")
    store.get("node-miss")
    return TestOutcome(
        test_id="probe",
        status="error",
        reason_codes=["node_exception"],
        detail={
            "message": "probe exploded",
            "exception_type": "ValueError",
            "secret": "should-not-log",
        },
    )


def _missing_node(inputs: dict, store) -> TestOutcome:
    del inputs, store
    return TestOutcome(
        test_id="probe",
        status="error",
        reason_codes=["missing_node"],
        detail={"message": "worker missing"},
    )


def _remote_node_exception(inputs: dict, store) -> TestOutcome:
    del inputs, store
    return TestOutcome(
        test_id="probe",
        status="error",
        reason_codes=["node_exception"],
        detail={
            "message": "remote probe exploded",
            "exception_type": "RuntimeError",
            "secret": "should-not-log",
        },
    )


class _RemoteBackend:
    def __init__(self, outcome: TestOutcome | None = None, exc: Exception | None = None) -> None:
        self._outcome = outcome
        self._exc = exc

    def run(self, node_id: str, inputs: dict, store) -> TestOutcome:
        del node_id, inputs, store
        if self._exc is not None:
            raise self._exc
        assert self._outcome is not None
        return self._outcome


def test_resolve_ok_emits_pinned_shape_with_local_cache():
    observer = RecordingObserver()

    run_test_run(
        "org/target",
        base_repo="org/base",
        revision_target="rev-target",
        revision_base=None,
        store=InMemoryArtifactStore(),
        rules=[Rule(test_id="support_classify", requires=[])],
        registry={
            "resolve_refs": _bootstrap_resolve_refs,
            "support_classify": _support_pass,
        },
        observer=observer,
    )

    resolve_ok = observer.events[0]
    assert resolve_ok["event"] == "resolve_ok"
    assert resolve_ok["logger"] == "aibom_verifier.orchestrator"
    fields = resolve_ok["fields"]
    assert fields["target"] == {
        "repo_id": "org/target",
        "revision": "rev-target",
        "sha": "target-sha",
    }
    assert fields["base"] == {
        "repo_id": "org/base",
        "revision": "main",
        "sha": "base-sha",
        "source": "cli",
    }
    assert isinstance(fields["duration_ms"], int)
    assert fields["duration_ms"] >= 0
    assert fields["cache"] == {"hits": 1, "misses": 1}
    assert "target_config" not in fields
    assert "base_config" not in fields


def test_resolve_failed_emits_then_reraises():
    observer = RecordingObserver()

    with pytest.raises(CompareStartError) as exc_info:
        run_test_run(
            "org/target",
            base_repo="org/base",
            store=InMemoryArtifactStore(),
            rules=[Rule(test_id="support_classify", requires=[])],
            registry={
                "resolve_refs": _resolve_compare_start_error,
                "support_classify": _support_pass,
            },
            observer=observer,
        )

    assert exc_info.value.error_code == "gated_unauthenticated"
    assert len(observer.events) == 1
    event = observer.events[0]
    assert event["event"] == "resolve_failed"
    assert event["logger"] == "aibom_verifier.orchestrator"
    assert event["fields"]["error_code"] == "gated_unauthenticated"
    assert event["fields"]["message"] == "no access"
    assert isinstance(event["fields"]["duration_ms"], int)
    assert event["fields"]["duration_ms"] >= 0


def test_skipped_rule_emits_single_test_skipped():
    observer = RecordingObserver()

    run_test_run(
        "org/target",
        base_repo="org/base",
        store=InMemoryArtifactStore(),
        rules=[
            Rule(test_id="support_classify", requires=[]),
            Rule(
                test_id="probe",
                requires=[
                    Requirement(
                        upstream_test_id="support_classify",
                        kind="status",
                        expect_status="fail",
                    )
                ],
            ),
        ],
        registry={
            "resolve_refs": _bootstrap_resolve_refs,
            "support_classify": _support_pass,
            "probe": _missing_node,
        },
        observer=observer,
    )

    assert [event["event"] for event in observer.events] == [
        "resolve_ok",
        "test_started",
        "test_finished",
        "test_skipped",
    ]
    skipped = observer.events[-1]
    assert skipped["logger"] == "aibom_verifier.orchestrator"
    assert skipped["fields"] == {
        "test_id": "probe",
        "upstream": "support_classify",
        "reason": "pass",
    }


def test_local_node_exception_emits_exception_and_cache():
    observer = RecordingObserver()

    run_test_run(
        "org/target",
        base_repo="org/base",
        store=InMemoryArtifactStore(),
        rules=[
            Rule(test_id="support_classify", requires=[]),
            Rule(test_id="probe", requires=[]),
        ],
        registry={
            "resolve_refs": _bootstrap_resolve_refs,
            "support_classify": _support_pass,
            "probe": _node_exception,
        },
        observer=observer,
    )

    assert [event["event"] for event in observer.events] == [
        "resolve_ok",
        "test_started",
        "test_finished",
        "test_started",
        "exception",
        "test_finished",
    ]
    exception_event = observer.events[-2]
    assert exception_event["fields"] == {
        "test_id": "probe",
        "exception_type": "ValueError",
        "message": "probe exploded",
    }
    assert "secret" not in exception_event["fields"]
    finished = observer.events[-1]
    assert finished["fields"]["test_id"] == "probe"
    assert finished["fields"]["status"] == "error"
    assert finished["fields"]["reason_codes"] == ["node_exception"]
    assert finished["fields"]["cache"] == {"hits": 1, "misses": 1}
    assert "detail" not in finished["fields"]


def test_local_missing_node_has_no_exception_event():
    observer = RecordingObserver()

    run_test_run(
        "org/target",
        base_repo="org/base",
        store=InMemoryArtifactStore(),
        rules=[
            Rule(test_id="support_classify", requires=[]),
            Rule(test_id="probe", requires=[]),
        ],
        registry={
            "resolve_refs": _bootstrap_resolve_refs,
            "support_classify": _support_pass,
            "probe": _missing_node,
        },
        observer=observer,
    )

    assert [event["event"] for event in observer.events] == [
        "resolve_ok",
        "test_started",
        "test_finished",
        "test_started",
        "test_finished",
    ]
    finished = observer.events[-1]
    assert finished["fields"]["reason_codes"] == ["missing_node"]


def test_remote_node_exception_omits_exception_and_cache():
    observer = RecordingObserver()
    backend = _RemoteBackend(outcome=_remote_node_exception({}, None))

    run_test_run(
        "org/target",
        base_repo="org/base",
        store=InMemoryArtifactStore(),
        rules=[
            Rule(test_id="support_classify", requires=[]),
            Rule(test_id="probe", requires=[]),
        ],
        registry={
            "resolve_refs": _bootstrap_resolve_refs,
            "support_classify": _support_pass,
            "probe": _node_exception,
        },
        backend=backend,
        observer=observer,
    )

    assert [event["event"] for event in observer.events] == [
        "resolve_ok",
        "test_started",
        "test_finished",
        "test_started",
        "test_finished",
    ]
    resolve_ok = observer.events[0]
    assert "cache" not in resolve_ok["fields"]
    finished = observer.events[-1]
    assert finished["fields"]["status"] == "error"
    assert finished["fields"]["reason_codes"] == ["node_exception"]
    assert "cache" not in finished["fields"]


def test_remote_backend_raise_emits_only_test_finished():
    observer = RecordingObserver()

    run_test_run(
        "org/target",
        base_repo="org/base",
        store=InMemoryArtifactStore(),
        rules=[Rule(test_id="support_classify", requires=[])],
        registry={"resolve_refs": _bootstrap_resolve_refs},
        backend=_RemoteBackend(exc=RuntimeError("ssh failed")),
        observer=observer,
    )

    assert [event["event"] for event in observer.events] == [
        "resolve_ok",
        "test_started",
        "test_finished",
    ]
    finished = observer.events[-1]
    assert finished["fields"]["status"] == "error"
    assert finished["fields"]["reason_codes"] == ["backend_exception"]
    assert "cache" not in finished["fields"]


def test_observer_none_is_silent():
    result = run_test_run(
        "org/target",
        base_repo="org/base",
        store=InMemoryArtifactStore(),
        rules=[Rule(test_id="support_classify", requires=[])],
        registry={
            "resolve_refs": _bootstrap_resolve_refs,
            "support_classify": _support_pass,
        },
        observer=None,
    )

    assert result.tests[0].test_id == "resolve_refs"


class _RaisingObserver:
    def on_event(self, event: str, *, logger: str, **fields: object) -> None:
        raise RuntimeError(f"observer boom: {event}")


def test_observer_self_failure_is_swallowed():
    result = run_test_run(
        "org/target",
        base_repo="org/base",
        store=InMemoryArtifactStore(),
        rules=[Rule(test_id="support_classify", requires=[])],
        registry={
            "resolve_refs": _bootstrap_resolve_refs,
            "support_classify": _support_pass,
        },
        observer=_RaisingObserver(),
    )

    assert result.final_verdict is not None
    assert result.tests[0].test_id == "resolve_refs"
