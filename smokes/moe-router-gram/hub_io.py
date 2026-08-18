from __future__ import annotations

import json
import re
from dataclasses import dataclass

import ml_dtypes  # noqa: F401
import numpy as np
from config import CACHE_DIR, GATE_SUFFIX
from huggingface_hub import HfApi

from aibom_verifier.hf.safetensors_io import fetch_tensor_bytes, list_tensor_names
from aibom_verifier.slots.artifact_store import FilesystemArtifactStore

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.gate\.weight$")
_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}


@dataclass(frozen=True)
class GateRef:
    name: str
    gate_rank: int
    physical_layer: int
    shape: tuple[int, ...]
    dtype: str


def resolve_sha(repo_id: str, *, api: HfApi | None = None) -> str:
    hf = api or HfApi()
    info = hf.model_info(repo_id)
    sha = info.sha
    if not isinstance(sha, str) or not sha:
        raise RuntimeError(f"no_sha:{repo_id}")
    return sha


def make_store() -> FilesystemArtifactStore:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return FilesystemArtifactStore(CACHE_DIR)


def gate_meta(repo_id: str, sha: str, store) -> dict[str, tuple[str, list[int]]]:
    cache_key = f"gate_meta_json:{repo_id}:{sha}"
    cached = store.get(cache_key)
    if cached is not None:
        raw = json.loads(cached.decode("utf-8"))
        return {k: (v[0], list(v[1])) for k, v in raw.items()}

    meta = HfApi().get_safetensors_metadata(repo_id, revision=sha)
    out: dict[str, tuple[str, list[int]]] = {}
    for fm in meta.files_metadata.values():
        for name, info in fm.tensors.items():
            if name.endswith(GATE_SUFFIX):
                out[name] = (str(info.dtype), list(info.shape))
    store.put(cache_key, json.dumps({k: [v[0], v[1]] for k, v in out.items()}).encode("utf-8"))
    return out


def list_gate_tensors(repo_id: str, sha: str, store) -> list[GateRef]:
    meta = gate_meta(repo_id, sha, store)
    _ = list_tensor_names(repo_id, sha, store)
    parsed: list[tuple[int, str]] = []
    for name in meta:
        m = _LAYER_RE.fullmatch(name)
        if not m:
            raise RuntimeError(f"unexpected_gate_name:{name}")
        parsed.append((int(m.group(1)), name))
    parsed.sort(key=lambda t: t[0])
    out: list[GateRef] = []
    for rank, (phys, name) in enumerate(parsed):
        dtype, shape = meta[name]
        out.append(
            GateRef(
                name=name,
                gate_rank=rank,
                physical_layer=phys,
                shape=tuple(shape),
                dtype=dtype,
            )
        )
    if not out:
        raise RuntimeError(f"no_gates:{repo_id}")
    return out


def decode_tensor(raw: bytes, dtype: str, shape: tuple[int, ...] | list[int]) -> np.ndarray:
    shape_t = tuple(shape)
    itemsize = _ITEMSIZE[dtype]
    expected = int(np.prod(shape_t)) * itemsize
    if len(raw) != expected:
        raise RuntimeError(
            f"byte_len_mismatch:got={len(raw)} expected={expected} dtype={dtype} shape={shape_t}"
        )
    if dtype == "BF16":
        arr = np.frombuffer(raw, dtype=ml_dtypes.bfloat16)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dtype == "F64":
        arr = np.frombuffer(raw, dtype=np.float64)
    else:
        raise RuntimeError(f"unsupported_dtype:{dtype}")
    return np.asarray(arr, dtype=np.float32).astype(np.float64).reshape(shape_t)


def load_gates(repo_id: str) -> list[np.ndarray]:
    store = make_store()
    sha = resolve_sha(repo_id)
    refs = list_gate_tensors(repo_id, sha, store)
    arrays: list[np.ndarray] = []
    for ref in refs:
        raw = fetch_tensor_bytes(repo_id, sha, ref.name, store)
        w = decode_tensor(raw, ref.dtype, ref.shape)
        if np.any(np.linalg.norm(w, axis=1) == 0):
            raise RuntimeError(f"zero_row:{ref.name}")
        arrays.append(w)
    return arrays
