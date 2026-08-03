from huggingface_hub import HfApi

from aibom_verifier.hf.card import resolve_presumed_base
from aibom_verifier.hf.config_io import load_config_json
from aibom_verifier.hf.pin import resolve_commit
from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.types import TestOutcome


def resolve_refs_node(inputs: dict, store: ArtifactStore) -> TestOutcome:
    """Pin target/base commit SHAs, resolve the presumed base, and load both configs.

    Any `CompareStartError` raised by the Hub-facing helpers below is a pre-DAG
    failure (gated/missing/not-found/etc.) and is intentionally left to
    propagate uncaught — the planner calls this node directly (not through
    `LocalWorker.run`, which would otherwise swallow it into a generic
    `status="error"` outcome) so the CLI sees the real error code/message.
    """
    target_repo: str = inputs["target_repo"]
    target_revision: str | None = inputs.get("target_revision")
    base_repo: str | None = inputs.get("base_repo")
    base_revision: str | None = inputs.get("base_revision")
    api: HfApi | None = inputs.get("api")

    target_sha = resolve_commit(target_repo, target_revision, api=api)

    if base_repo:
        base_source = "cli"
    else:
        base_repo = resolve_presumed_base(target_repo, target_sha, api=api)
        base_source = "card"

    base_sha = resolve_commit(base_repo, base_revision, api=api)

    target_config = load_config_json(target_repo, target_sha, store, api=api)
    base_config = load_config_json(base_repo, base_sha, store, api=api)

    return TestOutcome(
        test_id="resolve_refs",
        status="pass",
        detail={
            "target_sha": target_sha,
            "base_sha": base_sha,
            "base_repo": base_repo,
            "base_source": base_source,
            "target_config": target_config,
            "base_config": base_config,
        },
    )
