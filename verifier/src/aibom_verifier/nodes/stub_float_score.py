"""Stub detection test that emits a float score for gate unit tests only.

Not part of the live T1 boolean chain. Used to prove ``scores.* > threshold``
rule predicates without inventing float scores on block0 shapes/values.
"""

from __future__ import annotations

from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.types import TestOutcome


def stub_float_score_node(inputs: dict, store: ArtifactStore) -> TestOutcome:
    """Return a pass outcome with ``scores["x"]`` taken from ``inputs["score_x"]``.

    ``store`` is unused; kept to match the ``NodeFn`` signature.
    """
    del store
    score = float(inputs["score_x"])
    return TestOutcome(
        test_id="stub_float_score",
        status="pass",
        scores={"x": score},
    )
