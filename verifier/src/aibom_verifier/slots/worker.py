from collections.abc import Callable
from typing import Protocol

from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.types import TestOutcome

NodeFn = Callable[[dict, ArtifactStore], TestOutcome]


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
