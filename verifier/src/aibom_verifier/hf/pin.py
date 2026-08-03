from huggingface_hub import HfApi

from aibom_verifier.hf._common import hub_errors
from aibom_verifier.types import CompareStartError


def resolve_commit(repo_id: str, revision: str | None = None, *, api: HfApi | None = None) -> str:
    """Resolve `repo_id`@`revision` to a pinned commit SHA via the Hub API."""
    hf_api = api or HfApi()
    with hub_errors(repo_id, f"revision={revision!r}"):
        info = hf_api.repo_info(repo_id, revision=revision)

    sha = getattr(info, "sha", None)
    if not sha:
        raise CompareStartError(
            "resolve_failed",
            f"Hub did not return a commit SHA for '{repo_id}' (revision={revision!r}).",
        )
    return sha
