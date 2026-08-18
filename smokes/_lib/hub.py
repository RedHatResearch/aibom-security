"""Shared Hub safetensors loading for smokes (not product code)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open


def load_config(repo_id: str, cache_dir: Path | None = None) -> dict:
    path = hf_hub_download(repo_id, filename="config.json", cache_dir=cache_dir)
    with open(path) as f:
        return json.load(f)


def load_tensors(
    repo_id: str,
    names: set[str],
    *,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    model_dir = snapshot_download(
        repo_id,
        allow_patterns=["*.safetensors", "model.safetensors.index.json"],
        cache_dir=cache_dir,
    )
    root = Path(model_dir)
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors in {model_dir}")

    out: dict[str, np.ndarray] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                if name in names and name not in out:
                    out[name] = handle.get_tensor(name).to(dtype=torch.float32).numpy()
    missing = names - out.keys()
    if missing:
        raise KeyError(f"Missing tensors for {repo_id}: {sorted(missing)[:5]}")
    return out


def load_all_tensors(repo_id: str, *, cache_dir: Path | None = None) -> dict[str, np.ndarray]:
    model_dir = snapshot_download(
        repo_id,
        allow_patterns=["*.safetensors", "model.safetensors.index.json"],
        cache_dir=cache_dir,
    )
    root = Path(model_dir)
    out: dict[str, np.ndarray] = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                if name not in out:
                    out[name] = handle.get_tensor(name).to(dtype=torch.float32).numpy()
    return out


def attn_tensor_names(num_layers: int) -> set[str]:
    names: set[str] = set()
    for i in range(num_layers):
        p = f"model.layers.{i}.self_attn"
        names.update(
            f"{p}.{r}_proj.weight"
            for r in ("q", "k", "v", "o")
        )
    return names


def mlp_tensor_names(num_layers: int) -> set[str]:
    names: set[str] = set()
    for i in range(num_layers):
        p = f"model.layers.{i}.mlp"
        names.update(f"{p}.{r}_proj.weight" for r in ("gate", "up", "down"))
    return names


def embed_name() -> str:
    return "model.embed_tokens.weight"
