"""Thin public wrapper: ``run_compare`` → rule-list forward-walk orchestrator."""

from __future__ import annotations

from huggingface_hub import HfApi

from aibom_verifier.orchestrator import run_test_run
from aibom_verifier.registry import DEFAULT_REGISTRY
from aibom_verifier.rules import load_rules
from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.slots.execution_backend import ExecutionBackend
from aibom_verifier.slots.worker import NodeFn
from aibom_verifier.types import RunResult

__all__ = ["DEFAULT_REGISTRY", "run_compare"]


def run_compare(
    target_repo: str,
    *,
    base_repo: str | None = None,
    revision_target: str | None = None,
    revision_base: str | None = None,
    store: ArtifactStore,
    api: HfApi | None = None,
    node_overrides: dict[str, NodeFn] | None = None,
    backend: ExecutionBackend | None = None,
) -> RunResult:
    """Run the T1 gate chain and synthesize a ``RunResult``.

    Raises ``CompareStartError`` if resolve_refs cannot pin/resolve the target
    or base — that failure happens before the DAG conceptually starts.

    Cache bypass belongs on the ``store`` (construct with ``ignore_cache=True``).
    Default ``backend`` is in-process :class:`~aibom_verifier.backends.local.LocalBackend`.
    """
    registry = {**DEFAULT_REGISTRY, **(node_overrides or {})}
    return run_test_run(
        target_repo,
        base_repo=base_repo,
        revision_target=revision_target,
        revision_base=revision_base,
        store=store,
        rules=load_rules(),
        registry=registry,
        backend=backend,
        api=api,
    )
