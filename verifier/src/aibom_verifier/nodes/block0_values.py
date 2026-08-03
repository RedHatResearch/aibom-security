import json

from huggingface_hub import HfApi

from aibom_verifier.hf.safetensors_io import fetch_tensor_bytes
from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.slots.comparer import ExactBytesComparer
from aibom_verifier.types import TestOutcome

_COMPARER = ExactBytesComparer()


def _plan_key(target_repo: str, target_sha: str, base_repo: str, base_sha: str) -> str:
    return f"block0_plan:{target_repo}:{target_sha}:{base_repo}:{base_sha}"


def block0_values_node(inputs: dict, store: ArtifactStore) -> TestOutcome:
    """Test 2 — exact raw-byte comparison over the tensors planned by Test 1.

    Reads the `block0_plan` cached by `block0_shapes_node` from `store`
    (the planner only runs this node once shapes passed, so the plan is
    always present in practice; a missing plan is a planner wiring bug, not
    a business outcome, so it raises rather than returning a `TestOutcome`).
    """
    target_repo: str = inputs["target_repo"]
    target_sha: str = inputs["target_sha"]
    base_repo: str = inputs["base_repo"]
    base_sha: str = inputs["base_sha"]
    api: HfApi | None = inputs.get("api")

    plan_key = _plan_key(target_repo, target_sha, base_repo, base_sha)
    cached_plan = store.get(plan_key)
    if cached_plan is None:
        raise RuntimeError(f"block0_values_node ran without a cached plan at '{plan_key}'")
    plan = json.loads(cached_plan)

    try:
        for pair in plan["pairs"]:
            name = pair["name"]
            target_bytes = fetch_tensor_bytes(target_repo, target_sha, name, store, api=api)
            base_bytes = fetch_tensor_bytes(base_repo, base_sha, name, store, api=api)
            if not _COMPARER.equal(target_bytes, base_bytes):
                return TestOutcome(
                    test_id="block0_values",
                    status="fail",
                    compatibility="incompatible",
                    reason_codes=["byte_mismatch"],
                    detail={"first_mismatch": name},
                )
    except ValueError as exc:
        if str(exc) == "not_safetensors":
            return TestOutcome(
                test_id="block0_values",
                status="pass",
                compatibility="insufficient_evidence",
                reason_codes=["not_safetensors"],
                detail={"message": str(exc)},
            )
        raise

    return TestOutcome(
        test_id="block0_values",
        status="pass",
        compatibility="compatible",
        detail={"pair_count": len(plan["pairs"])},
    )
