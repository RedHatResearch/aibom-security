from typing import Any

from huggingface_hub import HfApi

from aibom_verifier.hf._common import hub_errors
from aibom_verifier.types import CompareStartError


def _extract_base_model(card_data: Any) -> str | list[str] | None:
    if card_data is None:
        return None
    if isinstance(card_data, dict):
        return card_data.get("base_model")
    return getattr(card_data, "base_model", None)


def resolve_presumed_base(repo_id: str, sha: str, *, api: HfApi | None = None) -> str:
    """Resolve the presumed base model claimed by `repo_id`'s card at `sha`.

    This is a claim to verify, not proof of lineage.
    """
    hf_api = api or HfApi()
    with hub_errors(repo_id, f"revision={sha!r}"):
        info = hf_api.model_info(repo_id, revision=sha)

    base_model = _extract_base_model(getattr(info, "card_data", None))

    if isinstance(base_model, list):
        base_model = base_model[0] if base_model else None
    if isinstance(base_model, str):
        base_model = base_model.strip() or None

    if not base_model:
        raise CompareStartError(
            "missing_base_model",
            f"Repo '{repo_id}' card has no 'base_model' metadata (revision={sha!r}).",
        )
    return base_model
