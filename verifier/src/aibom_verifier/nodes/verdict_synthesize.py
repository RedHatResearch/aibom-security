"""Verdict assembly helpers.

Not a ``NodeFn`` / registry entry — the planner assembles the run envelope
after the gate chain finishes. Pure decision functions live here so the
gate-outcome → verdict mapping stays independently testable.
"""

from __future__ import annotations

from aibom_verifier.types import FinalVerdict, TestOutcome


def support_gate_skip_reason(support_outcome: TestOutcome) -> str | None:
    """Reason to skip downstream gates given support_classify, or None to proceed."""
    if support_outcome.status == "pass" and support_outcome.compatibility == "compatible":
        return None
    if support_outcome.compatibility == "unsupported":
        return "unsupported"
    return support_outcome.status


def arch_hash_gate_skip_reason(arch_outcome: TestOutcome) -> str | None:
    """Reason to skip T1a/T1b given arch_hash, or None to proceed."""
    if arch_outcome.status == "pass" and arch_outcome.compatibility == "compatible":
        return None
    if "arch_hash_mismatch" in arch_outcome.reason_codes:
        return "arch_hash_mismatch"
    if arch_outcome.status == "fail":
        return "incompatible"
    return arch_outcome.status


def shapes_gate_skip_reason(shapes_outcome: TestOutcome) -> str | None:
    """Reason to skip block0_values given block0_shapes, or None to proceed."""
    if shapes_outcome.status == "pass" and shapes_outcome.compatibility == "compatible":
        return None
    if shapes_outcome.compatibility == "insufficient_evidence":
        if "not_safetensors" in shapes_outcome.reason_codes:
            return "not_safetensors"
        return "insufficient_evidence"
    if shapes_outcome.status == "fail":
        return "incompatible"
    return shapes_outcome.status


def synthesize_final_verdict(
    support_outcome: TestOutcome,
    arch_outcome: TestOutcome | None,
    shapes_outcome: TestOutcome | None,
    values_outcome: TestOutcome | None,
) -> FinalVerdict:
    """Map the gate chain's collected outcomes to the aibom Verdict taxonomy.

    Mapping (T1):
    - support abstain → ``unsupported``
    - arch_hash mismatch → ``fraudulent_claim``
    - shapes fail → ``incompatible``
    - shapes pass + values differ or match → ``verified_derivative``
    - anything inconclusive → ``insufficient_evidence``
    """
    if support_outcome.status != "pass":
        return "insufficient_evidence"
    if support_outcome.compatibility == "unsupported":
        return "unsupported"

    if arch_outcome is None:
        return "insufficient_evidence"
    if "arch_hash_mismatch" in arch_outcome.reason_codes or (
        arch_outcome.status == "fail" and arch_outcome.compatibility == "incompatible"
    ):
        return "fraudulent_claim"
    if arch_outcome.status != "pass" or arch_outcome.compatibility != "compatible":
        return "insufficient_evidence"

    if shapes_outcome is None:
        return "insufficient_evidence"
    if shapes_outcome.compatibility == "insufficient_evidence":
        return "insufficient_evidence"
    if shapes_outcome.status == "fail":
        return "incompatible"
    if shapes_outcome.status != "pass" or shapes_outcome.compatibility != "compatible":
        return "insufficient_evidence"

    if values_outcome is None:
        return "insufficient_evidence"
    if values_outcome.compatibility == "insufficient_evidence":
        return "insufficient_evidence"
    if values_outcome.status == "fail":
        # Honest fine-tune pattern: shapes align, block-0 bytes differ.
        return "verified_derivative"
    if values_outcome.status == "pass" and values_outcome.compatibility == "compatible":
        # Exact block-0 match still supports the claimed base relationship.
        return "verified_derivative"
    return "insufficient_evidence"


def verdict_message(verdict: FinalVerdict, *, tests: list[TestOutcome]) -> str:
    """Human-readable summary for the public VerificationResult envelope."""
    by_id = {t.test_id: t for t in tests}

    if verdict == "unsupported":
        support = by_id.get("support_classify")
        target_type = (support.detail.get("target_model_type") if support else None) or "?"
        base_type = (support.detail.get("base_model_type") if support else None) or "?"
        return (
            f"Architecture pair not in the dense allowlist "
            f"(target={target_type!r}, base={base_type!r}); abstaining."
        )
    if verdict == "fraudulent_claim":
        return "Architecture fingerprints (arch_hash) differ; claimed base does not match."
    if verdict == "incompatible":
        shapes = by_id.get("block0_shapes")
        if shapes and shapes.reason_codes:
            reasons = ", ".join(shapes.reason_codes)
        else:
            reasons = "shape mismatch"
        return f"Block-0 tensor inventory/shapes are incompatible ({reasons})."
    if verdict == "verified_derivative":
        values = by_id.get("block0_values")
        if values and values.status == "fail":
            return (
                "Block-0 shapes match and values differ — consistent with a fine-tune "
                "of the claimed base."
            )
        return "Block-0 shapes and values match the claimed base."
    return "Insufficient evidence to confirm or refute the claimed base relationship."
