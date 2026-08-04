from collections.abc import Callable
from typing import Protocol

from huggingface_hub import HfApi

from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.types import TestOutcome

NodeFn = Callable[[dict, ArtifactStore], TestOutcome]


def without_api(inputs: dict) -> dict:
    """Shallow copy of ``inputs`` without ``api`` (not JSON-serializable for remotes)."""
    return {key: value for key, value in inputs.items() if key != "api"}


def run_one_node(
    node_id: str,
    inputs: dict,
    *,
    store: ArtifactStore,
    registry: dict[str, NodeFn],
) -> TestOutcome:
    """Strip ``api``, inject ``HfApi()``, run via :class:`LocalWorker`."""
    cleaned = without_api(dict(inputs))
    cleaned["api"] = HfApi()
    return LocalWorker(registry).run(node_id, cleaned, store)


class Worker(Protocol):
    def run(self, node_id: str, inputs: dict, store: ArtifactStore) -> TestOutcome: ...


class LocalWorker:
    def __init__(self, registry: dict[str, NodeFn]) -> None:
        self._registry = registry

    def run(self, node_id: str, inputs: dict, store: ArtifactStore) -> TestOutcome:
        try:
            node_fn = self._registry[node_id]
        except KeyError:
            return TestOutcome(
                test_id=node_id,
                status="error",
                reason_codes=["missing_node"],
                detail={"message": f"Unknown node_id: {node_id}"},
            )
        try:
            return node_fn(inputs, store)
        except Exception as exc:
            return TestOutcome(
                test_id=node_id,
                status="error",
                reason_codes=["node_exception"],
                detail={
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                },
            )
