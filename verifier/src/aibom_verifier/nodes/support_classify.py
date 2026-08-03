from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.support import classify_pair
from aibom_verifier.types import Compatibility, TestOutcome


def support_classify_node(inputs: dict, store: ArtifactStore) -> TestOutcome:
    """Classify the target/base pair for PoC weight-test eligibility.

    Pure config lookup; no cache reads/writes (`store` is accepted only to
    satisfy the `NodeFn` signature).
    """
    del store
    target_config: dict = inputs["target_config"]
    base_config: dict = inputs["base_config"]

    target_model_type = target_config.get("model_type", "")
    base_model_type = base_config.get("model_type", "")

    support_class = classify_pair(target_model_type, base_model_type)
    compatibility: Compatibility = (
        "compatible" if support_class == "dense_supported" else "unsupported"
    )

    return TestOutcome(
        test_id="support_classify",
        status="pass",
        compatibility=compatibility,
        detail={
            "support_class": support_class,
            "target_model_type": target_model_type,
            "base_model_type": base_model_type,
        },
    )
