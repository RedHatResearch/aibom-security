import json
import struct

from huggingface_hub import HfApi, get_session, get_token, hf_hub_url
from huggingface_hub.errors import NotASafetensorsRepoError

from aibom_verifier.errors import NotSafetensorsError
from aibom_verifier.hf._common import hub_errors
from aibom_verifier.slots.artifact_store import ArtifactStore

_HEADER_LENGTH_BYTES = 8


def _range_headers(byte_range: str, *, token: str | None = None) -> dict[str, str]:
    """Build HTTP headers for an authenticated Range request.

    `get_session()` provides retry/timeout config but does NOT inject the
    HF auth token — gated repos return 401 without it. Prefer an explicit
    token (e.g. from an injected ``HfApi``) over the process-global login.
    """
    headers: dict[str, str] = {"Range": byte_range}
    auth_token = token if token is not None else get_token()
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers


def _api_token(api: HfApi | None) -> str | None:
    if api is None:
        return None
    token = getattr(api, "token", None)
    return token if isinstance(token, str) and token else None


def parse_header_length(first_8_bytes: bytes) -> int:
    """Parse the little-endian u64 header length from a safetensors file's first 8 bytes."""
    return struct.unpack("<Q", first_8_bytes[:_HEADER_LENGTH_BYTES])[0]


def data_byte_range(header_length: int, begin: int, end: int) -> tuple[int, int]:
    """Convert a tensor's header-relative `data_offsets` into an absolute file byte range.

    The safetensors layout is `[8-byte header length][header JSON][tensor data buffer]`,
    so `begin`/`end` (relative to the start of the data buffer) shift by `8 + header_length`.
    `end` is exclusive in the header but the returned end is inclusive, for HTTP `Range` headers.
    """
    absolute_start = _HEADER_LENGTH_BYTES + header_length + begin
    absolute_end = _HEADER_LENGTH_BYTES + header_length + end - 1
    return absolute_start, absolute_end


def _metadata_to_dict(metadata) -> dict:
    tensors: dict[str, dict] = {}
    for filename, file_metadata in metadata.files_metadata.items():
        for tensor_name, info in file_metadata.tensors.items():
            tensors[tensor_name] = {
                "dtype": info.dtype,
                "shape": list(info.shape),
                "data_offsets": list(info.data_offsets),
                "filename": filename,
            }
    return {"weight_map": dict(metadata.weight_map), "tensors": tensors}


def _fetch_metadata_dict(repo_id: str, sha: str, api: HfApi) -> dict:
    try:
        metadata = api.get_safetensors_metadata(repo_id, revision=sha)
    except NotASafetensorsRepoError as exc:
        raise NotSafetensorsError() from exc
    except Exception as exc:
        with hub_errors(repo_id, f"safetensors metadata at revision={sha!r}"):
            raise exc
    return _metadata_to_dict(metadata)


def _load_or_fetch_meta(repo_id: str, sha: str, store: ArtifactStore, api: HfApi) -> dict:
    cache_key = f"st_meta:{repo_id}:{sha}"
    cached = store.get(cache_key)
    if cached is not None:
        return json.loads(cached)

    meta = _fetch_metadata_dict(repo_id, sha, api)
    store.put(cache_key, json.dumps(meta).encode("utf-8"))
    return meta


def list_tensor_names(
    repo_id: str,
    sha: str,
    store: ArtifactStore,
    *,
    api: HfApi | None = None,
) -> list[str]:
    """Return the sorted list of tensor names for `repo_id`@`sha`.

    Raises :class:`~aibom_verifier.errors.NotSafetensorsError` if the repo has
    no safetensors weights.
    """
    hf_api = api or HfApi()
    meta = _load_or_fetch_meta(repo_id, sha, store, hf_api)
    return sorted(meta["weight_map"].keys())


def tensor_shapes(
    repo_id: str,
    sha: str,
    store: ArtifactStore,
    *,
    api: HfApi | None = None,
) -> dict[str, list[int]]:
    """Return `{tensor_name: shape}` for every tensor in `repo_id`@`sha`."""
    hf_api = api or HfApi()
    meta = _load_or_fetch_meta(repo_id, sha, store, hf_api)
    return {name: info["shape"] for name, info in meta["tensors"].items()}


def _load_or_fetch_shard_header(
    repo_id: str,
    sha: str,
    filename: str,
    store: ArtifactStore,
    *,
    token: str | None = None,
) -> tuple[dict, int]:
    cache_key = f"st_header:{repo_id}:{sha}:{filename}"
    cached = store.get(cache_key)
    if cached is not None:
        payload = json.loads(cached)
        return payload["header"], payload["header_length"]

    url = hf_hub_url(repo_id, filename, revision=sha)
    session = get_session()

    with hub_errors(repo_id, f"safetensors header {filename!r} at revision={sha!r}"):
        length_response = session.get(url, headers=_range_headers("bytes=0-7", token=token))
        length_response.raise_for_status()
        header_length = parse_header_length(length_response.content[:_HEADER_LENGTH_BYTES])

        header_end = _HEADER_LENGTH_BYTES + header_length - 1
        header_response = session.get(
            url, headers=_range_headers(f"bytes={_HEADER_LENGTH_BYTES}-{header_end}", token=token)
        )
        header_response.raise_for_status()
        header = json.loads(header_response.content)

    payload = json.dumps({"header": header, "header_length": header_length}).encode("utf-8")
    store.put(cache_key, payload)
    return header, header_length


def fetch_tensor_bytes(
    repo_id: str,
    sha: str,
    tensor_name: str,
    store: ArtifactStore,
    *,
    api: HfApi | None = None,
) -> bytes:
    """Fetch one tensor's raw bytes via HTTP Range requests (no full download)."""
    tensor_key = f"st_tensor:{repo_id}:{sha}:{tensor_name}"
    cached = store.get(tensor_key)
    if cached is not None:
        return cached

    hf_api = api or HfApi()
    token = _api_token(hf_api)
    meta = _load_or_fetch_meta(repo_id, sha, store, hf_api)
    weight_map = meta["weight_map"]
    if tensor_name not in weight_map:
        raise ValueError(f"unknown_tensor:{tensor_name}")
    filename = weight_map[tensor_name]

    header, header_length = _load_or_fetch_shard_header(repo_id, sha, filename, store, token=token)
    if tensor_name not in header:
        raise ValueError(f"tensor_not_in_header:{tensor_name}")
    begin, end = header[tensor_name]["data_offsets"]
    absolute_start, absolute_end = data_byte_range(header_length, begin, end)

    url = hf_hub_url(repo_id, filename, revision=sha)
    session = get_session()
    with hub_errors(repo_id, f"safetensors tensor {tensor_name!r} at revision={sha!r}"):
        response = session.get(
            url,
            headers=_range_headers(f"bytes={absolute_start}-{absolute_end}", token=token),
        )
        response.raise_for_status()
        data = response.content

    store.put(tensor_key, data)
    return data
