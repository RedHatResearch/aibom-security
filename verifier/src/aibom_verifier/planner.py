"""Orchestrate resolve → support → arch_hash → shapes → values → verdict."""

from __future__ import annotations

import json

from huggingface_hub import HfApi

from aibom_verifier.nodes.arch_hash_gate import arch_hash_gate_node
from aibom_verifier.nodes.block0_shapes import block0_shapes_node
from aibom_verifier.nodes.block0_values import block0_values_node
from aibom_verifier.nodes.resolve_refs import resolve_refs_node
from aibom_verifier.nodes.support_classify import support_classify_node
from aibom_verifier.nodes.verdict_synthesize import (
    arch_hash_gate_skip_reason,
    shapes_gate_skip_reason,
    support_gate_skip_reason,
    synthesize_final_verdict,
)
from aibom_verifier.slots.artifact_store import ArtifactStore, CountingArtifactStore
from aibom_verifier.slots.worker import LocalWorker, NodeFn
from aibom_verifier.types import ModelRef, RunResult, TestOutcome


def _result_key(target_repo: str, target_sha: str, base_repo: str, base_sha: str) -> str:
    return f"result:{target_repo}:{target_sha}:{base_repo}:{base_sha}"


def _skip_outcome(test_id: str, upstream: str, reason: str) -> TestOutcome:
    return TestOutcome(
        test_id=test_id,
        status="skip",
        skipped_because={"upstream": upstream, "reason": reason},
    )


def run_compare(
    target_repo: str,
    *,
    base_repo: str | None = None,
    revision_target: str | None = None,
    revision_base: str | None = None,
    store: ArtifactStore,
    api: HfApi | None = None,
    node_overrides: dict[str, NodeFn] | None = None,
) -> RunResult:
    """Run the T1 gate chain and synthesize a ``RunResult``.

    Raises ``CompareStartError`` if resolve_refs cannot pin/resolve the target
    or base — that failure happens before the DAG conceptually starts.

    Cache bypass belongs on the ``store`` (construct with ``ignore_cache=True``).
    """
    counting_store = CountingArtifactStore(store)

    registry: dict[str, NodeFn] = {
        "resolve_refs": resolve_refs_node,
        "support_classify": support_classify_node,
        "arch_hash": arch_hash_gate_node,
        "block0_shapes": block0_shapes_node,
        "block0_values": block0_values_node,
    }
    if node_overrides:
        registry.update(node_overrides)
    worker = LocalWorker(registry)

    resolve_outcome = registry["resolve_refs"](
        {
            "target_repo": target_repo,
            "target_revision": revision_target,
            "base_repo": base_repo,
            "base_revision": revision_base,
            "api": api,
        },
        counting_store,
    )
    tests: list[TestOutcome] = [resolve_outcome]

    target_sha: str = resolve_outcome.detail["target_sha"]
    base_sha: str = resolve_outcome.detail["base_sha"]
    resolved_base_repo: str = resolve_outcome.detail["base_repo"]
    base_source = resolve_outcome.detail["base_source"]
    target_config: dict = resolve_outcome.detail["target_config"]
    base_config: dict = resolve_outcome.detail["base_config"]

    target_ref = ModelRef(repo_id=target_repo, revision=revision_target or "main", sha=target_sha)
    base_ref = ModelRef(repo_id=resolved_base_repo, revision=revision_base or "main", sha=base_sha)

    support_outcome = worker.run(
        "support_classify",
        {"target_config": target_config, "base_config": base_config},
        counting_store,
    )
    tests.append(support_outcome)

    arch_outcome: TestOutcome | None = None
    shapes_outcome: TestOutcome | None = None
    values_outcome: TestOutcome | None = None

    support_skip = support_gate_skip_reason(support_outcome)
    if support_skip is not None:
        arch_outcome = _skip_outcome("arch_hash", "support_classify", support_skip)
        shapes_outcome = _skip_outcome("block0_shapes", "support_classify", support_skip)
        values_outcome = _skip_outcome("block0_values", "support_classify", support_skip)
        tests.extend([arch_outcome, shapes_outcome, values_outcome])
    else:
        arch_outcome = worker.run(
            "arch_hash",
            {"target_config": target_config, "base_config": base_config},
            counting_store,
        )
        tests.append(arch_outcome)

        arch_skip = arch_hash_gate_skip_reason(arch_outcome)
        if arch_skip is not None:
            shapes_outcome = _skip_outcome("block0_shapes", "arch_hash", arch_skip)
            values_outcome = _skip_outcome("block0_values", "arch_hash", arch_skip)
            tests.extend([shapes_outcome, values_outcome])
        else:
            block0_inputs = {
                "target_repo": target_repo,
                "target_sha": target_sha,
                "base_repo": resolved_base_repo,
                "base_sha": base_sha,
                "api": api,
            }
            shapes_outcome = worker.run("block0_shapes", block0_inputs, counting_store)
            tests.append(shapes_outcome)

            shapes_skip = shapes_gate_skip_reason(shapes_outcome)
            if shapes_skip is not None:
                values_outcome = _skip_outcome("block0_values", "block0_shapes", shapes_skip)
                tests.append(values_outcome)
            else:
                values_outcome = worker.run("block0_values", block0_inputs, counting_store)
                tests.append(values_outcome)

    final_verdict = synthesize_final_verdict(
        support_outcome, arch_outcome, shapes_outcome, values_outcome
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
