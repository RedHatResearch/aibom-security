import json

from huggingface_hub import HfApi

from aibom_verifier.hf._common import hub_errors
from aibom_verifier.slots.artifact_store import ArtifactStore


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
        return json.loads(cached)

    hf_api = api or HfApi()
    with hub_errors(repo_id, f"config.json at revision={sha!r}"):
        local_path = hf_api.hf_hub_download(repo_id, filename="config.json", revision=sha)

    with open(local_path, "rb") as f:
        data = f.read()

    store.put(cache_key, data)
    return json.loads(data)
