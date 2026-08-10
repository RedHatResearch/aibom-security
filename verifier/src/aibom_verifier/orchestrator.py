"""Forward-walk test-run orchestrator driven by a rule list."""

from __future__ import annotations

import json
from time import perf_counter_ns

from huggingface_hub import HfApi

from aibom_verifier.backends.local import LocalBackend
from aibom_verifier.nodes.verdict_synthesize import (
    arch_hash_gate_skip_reason,
    shapes_gate_skip_reason,
    support_gate_skip_reason,
    synthesize_final_verdict,
)
from aibom_verifier.observer import RunObserver, safe_on_event
from aibom_verifier.rules import Requirement, Rule, requirement_satisfied, skip_reason_for
from aibom_verifier.run_log import elapsed_ms
from aibom_verifier.slots.artifact_store import ArtifactStore, CountingArtifactStore
from aibom_verifier.slots.execution_backend import ExecutionBackend
from aibom_verifier.slots.worker import NodeFn, without_api
from aibom_verifier.types import CompareStartError, ModelRef, RunResult, TestOutcome

_ORCHESTRATOR_LOGGER = "aibom_verifier.orchestrator"


def _result_key(target_repo: str, target_sha: str, base_repo: str, base_sha: str) -> str:
    return f"result:{target_repo}:{target_sha}:{base_repo}:{base_sha}"


def _skip_outcome(test_id: str, upstream: str, reason: str) -> TestOutcome:
    return TestOutcome(
        test_id=test_id,
        status="skip",
        skipped_because={"upstream": upstream, "reason": reason},
    )


def _inputs_for(
    *,
    target_repo: str,
    target_sha: str,
    base_repo: str,
    base_sha: str,
    target_config: dict,
    base_config: dict,
    api: HfApi | None,
    extra_inputs: dict | None,
) -> dict:
    # Pass the full standard bag; nodes ignore keys they do not use.
    inputs = {
        "target_repo": target_repo,
        "target_sha": target_sha,
        "base_repo": base_repo,
        "base_sha": base_sha,
        "target_config": target_config,
        "base_config": base_config,
        "api": api,
    }
    if extra_inputs:
        inputs.update(extra_inputs)
    return inputs


def _live_gate_skip_reason(upstream_test_id: str, upstream: TestOutcome) -> str | None:
    """Rich skip reason from live T1 helpers, or None if the gate allows proceed."""
    if upstream_test_id == "support_classify":
        return support_gate_skip_reason(upstream)
    if upstream_test_id == "arch_hash":
        return arch_hash_gate_skip_reason(upstream)
    if upstream_test_id == "block0_shapes":
        return shapes_gate_skip_reason(upstream)
    return None


def _cache_delta(
    store: CountingArtifactStore,
    *,
    start_hits: int,
    start_misses: int,
) -> dict[str, int]:
    return {
        "hits": len(store.hits) - start_hits,
        "misses": len(store.misses) - start_misses,
    }


def _mark(store: CountingArtifactStore) -> tuple[int, int, int]:
    return perf_counter_ns(), len(store.hits), len(store.misses)


def _observe(observer: RunObserver | None, event: str, **fields: object) -> None:
    safe_on_event(observer, event, logger=_ORCHESTRATOR_LOGGER, **fields)


def _hybrid_skip(
    test_id: str,
    req: Requirement,
    upstream: TestOutcome,
) -> TestOutcome:
    """Skip ``test_id`` using hybrid reasons; propagate cascade when upstream skipped."""
    if upstream.status == "skip" and upstream.skipped_because is not None:
        return _skip_outcome(
            test_id,
            upstream.skipped_because["upstream"],
            upstream.skipped_because["reason"],
        )

    reason = _live_gate_skip_reason(req.upstream_test_id, upstream)
    if reason is None:
        reason = skip_reason_for(req, upstream)

    return _skip_outcome(test_id, req.upstream_test_id, reason)


def _first_blocking_requirement(
    rule: Rule,
    outcomes: dict[str, TestOutcome],
) -> tuple[Requirement, TestOutcome | None] | None:
    """Return the first requirement that blocks ``rule``, using hybrid WHETHER.

    Rules decide eligibility via predicates; for live T1 upstreams the skip
    helpers also enforce compatibility (e.g. support ``pass`` + ``unsupported``).
    """
    for req in rule.requires:
        upstream = outcomes.get(req.upstream_test_id)
        if upstream is None:
            return req, None
        if not requirement_satisfied(req, outcomes):
            return req, upstream
        if _live_gate_skip_reason(req.upstream_test_id, upstream) is not None:
            return req, upstream
    return None


def run_test_run(
    target_repo: str,
    *,
    base_repo: str | None = None,
    revision_target: str | None = None,
    revision_base: str | None = None,
    store: ArtifactStore,
    rules: list[Rule],
    registry: dict[str, NodeFn],
    backend: ExecutionBackend | None = None,
    api: HfApi | None = None,
    extra_inputs: dict | None = None,
    observer: RunObserver | None = None,
) -> RunResult:
    """Forward-walk ``rules`` after bootstrap ``resolve_refs``.

    Raises ``CompareStartError`` if resolve_refs cannot pin/resolve the pair.
    ``resolve_refs`` always runs in-process (not via ``backend``).
    Default backend is :class:`~aibom_verifier.backends.local.LocalBackend`.
    Non-local backends receive inputs without ``api``; remotes construct their
    own ``HfApi`` and rebuild the store from env.
    """
    if api is None:
        api = HfApi()
    counting_store = CountingArtifactStore(store)
    exec_backend: ExecutionBackend = backend if backend is not None else LocalBackend(registry)
    # Local logging (cache deltas / exception): in-process path only (keeps_api).
    local_execution = bool(getattr(exec_backend, "keeps_api", False))

    resolve_fn = registry["resolve_refs"]
    resolve_start_ns, resolve_start_hits, resolve_start_misses = _mark(counting_store)
    try:
        resolve_outcome = resolve_fn(
            {
                "target_repo": target_repo,
                "target_revision": revision_target,
                "base_repo": base_repo,
                "base_revision": revision_base,
                "api": api,
            },
            counting_store,
        )
    except CompareStartError as exc:
        _observe(
            observer,
            "resolve_failed",
            error_code=exc.error_code,
            message=exc.message,
            duration_ms=elapsed_ms(resolve_start_ns),
        )
        raise
    tests: list[TestOutcome] = [resolve_outcome]
    outcomes: dict[str, TestOutcome] = {"resolve_refs": resolve_outcome}

    target_sha: str = resolve_outcome.detail["target_sha"]
    base_sha: str = resolve_outcome.detail["base_sha"]
    resolved_base_repo: str = resolve_outcome.detail["base_repo"]
    base_source = resolve_outcome.detail["base_source"]
    target_config: dict = resolve_outcome.detail["target_config"]
    base_config: dict = resolve_outcome.detail["base_config"]

    target_ref = ModelRef(repo_id=target_repo, revision=revision_target or "main", sha=target_sha)
    base_ref = ModelRef(repo_id=resolved_base_repo, revision=revision_base or "main", sha=base_sha)
    resolve_fields: dict[str, object] = {
        "target": target_ref.to_dict(),
        "base": {**base_ref.to_dict(), "source": base_source},
        "duration_ms": elapsed_ms(resolve_start_ns),
    }
    if local_execution:
        resolve_fields["cache"] = _cache_delta(
            counting_store,
            start_hits=resolve_start_hits,
            start_misses=resolve_start_misses,
        )
    _observe(observer, "resolve_ok", **resolve_fields)

    for rule in rules:
        blocked = _first_blocking_requirement(rule, outcomes)
        if blocked is not None:
            unsatisfied, upstream = blocked
            if upstream is None:
                outcome = _skip_outcome(
                    rule.test_id,
                    unsatisfied.upstream_test_id,
                    "missing_upstream",
                )
            else:
                outcome = _hybrid_skip(rule.test_id, unsatisfied, upstream)
            skipped_because = outcome.skipped_because
            assert skipped_because is not None
            _observe(
                observer,
                "test_skipped",
                test_id=outcome.test_id,
                upstream=skipped_because["upstream"],
                reason=skipped_because["reason"],
            )
        else:
            inputs = _inputs_for(
                target_repo=target_repo,
                target_sha=target_sha,
                base_repo=resolved_base_repo,
                base_sha=base_sha,
                target_config=target_config,
                base_config=base_config,
                api=api,
                extra_inputs=extra_inputs,
            )
            if not local_execution:
                inputs = without_api(inputs)
            _observe(observer, "test_started", test_id=rule.test_id, mode="execute")
            test_start_ns, test_start_hits, test_start_misses = _mark(counting_store)
            try:
                outcome = exec_backend.run(rule.test_id, inputs, counting_store)
            except Exception as exc:
                # Backends should prefer returning error outcomes; this catch
                # keeps verify from crashing if a remote backend raises.
                outcome = TestOutcome(
                    test_id=rule.test_id,
                    status="error",
                    reason_codes=["backend_exception"],
                    detail={
                        "message": str(exc),
                        "exception_type": type(exc).__name__,
                    },
                )
            finished_fields: dict[str, object] = {
                "test_id": outcome.test_id,
                "status": outcome.status,
                "reason_codes": list(outcome.reason_codes),
                "duration_ms": elapsed_ms(test_start_ns),
            }
            if local_execution:
                finished_fields["cache"] = _cache_delta(
                    counting_store,
                    start_hits=test_start_hits,
                    start_misses=test_start_misses,
                )
            if local_execution and "node_exception" in outcome.reason_codes:
                _observe(
                    observer,
                    "exception",
                    test_id=outcome.test_id,
                    exception_type=outcome.detail.get("exception_type", "Exception"),
                    message=outcome.detail.get("message", ""),
                )
            _observe(observer, "test_finished", **finished_fields)

        tests.append(outcome)
        outcomes[rule.test_id] = outcome

    support_outcome = outcomes.get("support_classify")
    if support_outcome is None:
        raise RuntimeError("rules must include support_classify for verdict synthesis")

    final_verdict = synthesize_final_verdict(
        support_outcome,
        outcomes.get("arch_hash"),
        outcomes.get("block0_shapes"),
        outcomes.get("block0_values"),
    )

    result = RunResult(
        target=target_ref,
        base=base_ref,
        base_source=base_source,
        support_class=support_outcome.detail.get("support_class", "unsupported"),
        tests=tests,
        final_verdict=final_verdict,
        cache={"hits": counting_store.hits, "misses": counting_store.misses},
    )

    result_key = _result_key(target_repo, target_sha, resolved_base_repo, base_sha)
    counting_store.put(result_key, json.dumps(result.to_dict()).encode("utf-8"))

    return result
