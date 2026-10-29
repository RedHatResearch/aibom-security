"""FR-4 — compare target/base ``arch_hash`` digests before tensor work."""

from __future__ import annotations

from aibom_verifier.arch_hash import canonicalize_arch_config, compute_arch_hash
from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.types import Compatibility, TestOutcome


def _differing_fields(target_config: dict, base_config: dict) -> dict[str, list[object]]:
    """Per-field ``[target, base]`` pairs for every canonical field that differs.

    A config difference alone does not establish a different model: routine
    adaptation (vocabulary extension, position-embedding resize, rope
    recalibration, a quantization recipe dropping ``sliding_window``, a tie
    flip) changes these fields while the lineage claim stays true.
    """
    target = canonicalize_arch_config(target_config)
    base = canonicalize_arch_config(base_config)
    return {field: [target[field], base[field]] for field in target if target[field] != base[field]}


def arch_hash_gate_node(inputs: dict, store: ArtifactStore) -> TestOutcome:
    """Hard-negative gate: matching hashes pass; mismatch fails the gate.

    The verdict reports the differing fields and does not adjudicate.
    """
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
            "differing_fields": {} if match else _differing_fields(target_config, base_config),
        },
    )
