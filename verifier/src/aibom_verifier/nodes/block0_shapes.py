import json

from huggingface_hub import HfApi

from aibom_verifier.hf.safetensors_io import list_tensor_names, tensor_shapes
from aibom_verifier.mapping.block0 import build_block0_plan
from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.types import TestOutcome


def _plan_key(target_repo: str, target_sha: str, base_repo: str, base_sha: str) -> str:
    return f"block0_plan:{target_repo}:{target_sha}:{base_repo}:{base_sha}"


def block0_shapes_node(inputs: dict, store: ArtifactStore) -> TestOutcome:
    """Test 1 — block-0 tensor inventory + shape/count gate.

    Builds and caches the `block0_plan` shared with `block0_values_node`, then
    gates on missing required tensors, one-sided optional tensors (design
    §4.5: a role present on only one side is a shape fail, not a skip), and
    per-tensor shape equality.
    """
    target_repo: str = inputs["target_repo"]
    target_sha: str = inputs["target_sha"]
    base_repo: str = inputs["base_repo"]
    base_sha: str = inputs["base_sha"]
    api: HfApi | None = inputs.get("api")

    plan_key = _plan_key(target_repo, target_sha, base_repo, base_sha)

    try:
        target_names = list_tensor_names(target_repo, target_sha, store, api=api)
        base_names = list_tensor_names(base_repo, base_sha, store, api=api)
        plan = build_block0_plan(target_names, base_names)
        store.put(plan_key, json.dumps(plan).encode("utf-8"))

        missing_target = plan["missing_required_target"]
        missing_base = plan["missing_required_base"]
        one_sided = plan["one_sided"]

        if missing_target or missing_base or one_sided:
            reason_codes = []
            if missing_target or missing_base:
                reason_codes.append("missing_required")
            if one_sided:
                reason_codes.append("one_sided_optional")
            return TestOutcome(
                test_id="block0_shapes",
                status="fail",
                compatibility="incompatible",
                reason_codes=reason_codes,
                artifacts=[plan_key],
                detail={
                    "missing_required_target": missing_target,
                    "missing_required_base": missing_base,
                    "one_sided": one_sided,
                },
            )

        target_shapes = tensor_shapes(target_repo, target_sha, store, api=api)
        base_shapes = tensor_shapes(base_repo, base_sha, store, api=api)
    except ValueError as exc:
        if str(exc) == "not_safetensors":
            return TestOutcome(
                test_id="block0_shapes",
                status="pass",
                compatibility="insufficient_evidence",
                reason_codes=["not_safetensors"],
                detail={"message": str(exc)},
            )
        raise

    mismatched = [
        pair["name"]
        for pair in plan["pairs"]
        if target_shapes.get(pair["name"]) != base_shapes.get(pair["name"])
    ]

    if mismatched:
        return TestOutcome(
            test_id="block0_shapes",
            status="fail",
            compatibility="incompatible",
            reason_codes=["shape_mismatch"],
            artifacts=[plan_key],
            detail={"mismatched_tensors": mismatched},
        )

    return TestOutcome(
        test_id="block0_shapes",
        status="pass",
        compatibility="compatible",
        artifacts=[plan_key],
        detail={"pair_count": len(plan["pairs"])},
    )
