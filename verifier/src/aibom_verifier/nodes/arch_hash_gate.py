"""FR-4 — compare target/base ``arch_hash`` digests before tensor work."""

from __future__ import annotations

from aibom_verifier.arch_hash import compute_arch_hash
from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.types import Compatibility, TestOutcome


def arch_hash_gate_node(inputs: dict, store: ArtifactStore) -> TestOutcome:
    """Hard-negative gate: matching hashes pass; mismatch fails the claim."""
    del store
    target_config: dict = inputs["target_config"]
    base_config: dict = inputs["base_config"]

    target_hash = compute_arch_hash(target_config)
    base_hash = compute_arch_hash(base_config)
    match = target_hash == base_hash

    compatibility: Compatibility = "compatible" if match else "incompatible"
    status = "pass" if match else "fail"
    reason_codes = [] if match else ["arch_hash_mismatch"]

    return TestOutcome(
        test_id="arch_hash",
        status=status,
        compatibility=compatibility,
        reason_codes=reason_codes,
        detail={
            "target_arch_hash": target_hash,
            "base_arch_hash": base_hash,
            "match": match,
        },
    )
