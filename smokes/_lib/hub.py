"""Shared Hub safetensors loading for smokes (not product code)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open


@dataclass(frozen=True)
class ModelConfig:
    repo_id: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def load_config(repo_id: str, cache_dir: Path | None = None) -> dict:
    path = hf_hub_download(repo_id, filename="config.json", cache_dir=cache_dir)
    with open(path) as f:
        return json.load(f)


def parse_model_config(repo_id: str, raw: dict) -> ModelConfig:
    return ModelConfig(
        repo_id=repo_id,
        hidden_size=raw["hidden_size"],
        num_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
        intermediate_size=raw["intermediate_size"],
    )


def expand_kv_heads(
    weight: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    if n_kv_heads == n_heads:
        return weight
    n_rep = n_heads // n_kv_heads
    blocks = weight.reshape(n_kv_heads, head_dim, weight.shape[1])
    return np.repeat(blocks, n_rep, axis=0).reshape(n_heads * head_dim, weight.shape[1])


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
            for name in handle.keys():  # noqa: SIM118 — safe_open is not iterable
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
            for name in handle.keys():  # noqa: SIM118 — safe_open is not iterable
                if name not in out:
                    out[name] = handle.get_tensor(name).to(dtype=torch.float32).numpy()
    return out


def attn_tensor_names(num_layers: int) -> set[str]:
    names: set[str] = set()
    for i in range(num_layers):
        p = f"model.layers.{i}.self_attn"
        names.update(f"{p}.{r}_proj.weight" for r in ("q", "k", "v", "o"))
    return names


def mlp_tensor_names(num_layers: int) -> set[str]:
    names: set[str] = set()
    for i in range(num_layers):
        p = f"model.layers.{i}.mlp"
        names.update(f"{p}.{r}_proj.weight" for r in ("gate", "up", "down"))
    return names


def crs_layer_tensor_names(num_layers: int) -> set[str]:
    names: set[str] = set()
    for i in range(num_layers):
        p = f"model.layers.{i}"
        names.update(
            {
                f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight",
                f"{p}.self_attn.o_proj.weight",
                f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight",
            }
        )
    return names


def embed_name() -> str:
    return "model.embed_tokens.weight"
