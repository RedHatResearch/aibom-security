"""Unit tests for FR-7 verdict synthesis mapping table."""

from aibom_verifier.nodes.verdict_synthesize import synthesize_final_verdict, verdict_message
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


def _arch(*, match=True, status="pass", reason_codes=None, differing_fields=None):
    if reason_codes is None:
        reason_codes = [] if match else ["arch_hash_mismatch"]
    return TestOutcome(
        test_id="arch_hash",
        status=status if match else "fail",
        compatibility="compatible" if match else "incompatible",
        reason_codes=reason_codes,
        detail={"match": match, "differing_fields": differing_fields or {}},
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


def test_verdict_incompatible_on_arch_hash_mismatch():
    # Config drift is not proof of fraud: vocabulary extension, context resize,
    # rope recalibration and tie flips all move canonical fields.
    assert synthesize_final_verdict(_support(), _arch(match=False), None, None) == "incompatible"


def test_message_arch_hash_mismatch_names_differing_fields():
    arch = _arch(
        match=False,
        differing_fields={"model_type": ["mixtral", "llama"], "hidden_size": [4096, 2048]},
    )

    message = verdict_message("incompatible", tests=[_support(), arch])

    assert "hidden_size" in message
    assert "model_type" in message
    assert "not adjudicated" in message


def test_message_arch_incompatible_without_reason_code_uses_arch_wording():
    # A fail/incompatible arch outcome is the cause even without a reason code,
    # as in synthesize_final_verdict; the message must not blame block 0.
    arch = _arch(
        match=False,
        reason_codes=[],
        differing_fields={"model_type": ["mixtral", "llama"]},
    )

    message = verdict_message("incompatible", tests=[arch])

    assert "model_type" in message
    assert "block-0" not in message.lower()


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
