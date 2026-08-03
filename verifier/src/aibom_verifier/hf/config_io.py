import json

from huggingface_hub import HfApi

from aibom_verifier.hf._common import hub_errors
from aibom_verifier.slots.artifact_store import ArtifactStore
from aibom_verifier.types import CompareStartError


def load_config_json(
    repo_id: str,
    sha: str,
    store: ArtifactStore,
    *,
    api: HfApi | None = None,
) -> dict:
    """Load `config.json` for `repo_id`@`sha`, caching the raw bytes in `store`."""
    cache_key = f"config:{repo_id}:{sha}"
    cached = store.get(cache_key)
    if cached is not None:
        try:
            return json.loads(cached)
        except json.JSONDecodeError as exc:
            raise CompareStartError(
                "resolve_failed",
                f"Cached config.json for '{repo_id}' is not valid JSON.",
            ) from exc

    hf_api = api or HfApi()
    with hub_errors(repo_id, f"config.json at revision={sha!r}"):
        local_path = hf_api.hf_hub_download(repo_id, filename="config.json", revision=sha)

    try:
        with open(local_path, "rb") as f:
            data = f.read()
        parsed = json.loads(data)
    except OSError as exc:
        raise CompareStartError(
            "resolve_failed",
            f"Failed to read config.json for '{repo_id}': {exc}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise CompareStartError(
            "resolve_failed",
            f"config.json for '{repo_id}' is not valid JSON.",
        ) from exc

    store.put(cache_key, data)
    return parsed
