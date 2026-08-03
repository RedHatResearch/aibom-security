"""Unit tests for FR-7 verdict synthesis mapping table."""

from aibom_verifier.nodes.verdict_synthesize import synthesize_final_verdict
from aibom_verifier.types import TestOutcome


def _support(*, compatibility="compatible", status="pass", model_type="llama"):
    return TestOutcome(
        test_id="support_classify",
        status=status,
        compatibility=compatibility,
        detail={
            "support_class": "dense_supported" if compatibility == "compatible" else "unsupported",
            "target_model_type": model_type,
            "base_model_type": model_type if compatibility == "compatible" else "mixtral",
        },
    )


def _arch(*, match=True, status="pass"):
    return TestOutcome(
        test_id="arch_hash",
        status=status if match else "fail",
        compatibility="compatible" if match else "incompatible",
        reason_codes=[] if match else ["arch_hash_mismatch"],
        detail={"match": match},
    )


def _shapes(*, status="pass", compatibility="compatible", reason_codes=None):
    return TestOutcome(
        test_id="block0_shapes",
        status=status,
        compatibility=compatibility,
        reason_codes=reason_codes or [],
    )


def _values(*, status="pass", compatibility="compatible"):
    return TestOutcome(
        test_id="block0_values",
        status=status,
        compatibility=compatibility,
        reason_codes=["byte_mismatch"] if status == "fail" else [],
    )


def test_verdict_unsupported():
    assert (
        synthesize_final_verdict(_support(compatibility="unsupported"), None, None, None)
        == "unsupported"
    )


def test_verdict_fraudulent_claim_on_arch_hash_mismatch():
    assert (
        synthesize_final_verdict(_support(), _arch(match=False), None, None) == "fraudulent_claim"
    )


def test_verdict_incompatible_on_shapes_fail():
    assert (
        synthesize_final_verdict(
            _support(),
            _arch(),
            _shapes(status="fail", compatibility="incompatible", reason_codes=["shape_mismatch"]),
            None,
        )
        == "incompatible"
    )


def test_verdict_verified_derivative_when_values_differ():
    assert (
        synthesize_final_verdict(
            _support(),
            _arch(),
            _shapes(),
            _values(status="fail", compatibility="incompatible"),
        )
        == "verified_derivative"
    )


def test_verdict_verified_derivative_when_values_match():
    assert (
        synthesize_final_verdict(_support(), _arch(), _shapes(), _values()) == "verified_derivative"
    )


def test_verdict_insufficient_evidence_when_not_safetensors():
    assert (
        synthesize_final_verdict(
            _support(),
            _arch(),
            _shapes(
                status="pass",
                compatibility="insufficient_evidence",
                reason_codes=["not_safetensors"],
            ),
            None,
        )
        == "insufficient_evidence"
    )
