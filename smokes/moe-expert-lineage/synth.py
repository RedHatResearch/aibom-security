"""Build Experiment A variants from a cached parent MoE checkpoint.

The parent is read from the local HF cache (no network). Each variant is written
to ``.scratch/moe-expert-lineage/<model>/<variant>/`` as ``model.safetensors``
plus a ``manifest.json`` that carries only checkpoint-card metadata (arch, seed,
layer -> expert count, projection names, tensor file) -- never ground truth. The
injected permutation / mixture is returned to the caller and kept in
``results.json``, so the detector cannot see it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from config import ARCHES, MERGE_ALPHA, SHALLOW_ALPHA, VARIANT_ORDINAL
from huggingface_hub import constants
from safetensors import safe_open
from safetensors.torch import save_file


# --------------------------------------------------------------------------- #
# parent checkpoint access
# --------------------------------------------------------------------------- #
def cached_snapshot(repo: str, revision: str) -> Path:
    """Local snapshot dir for a pinned repo/revision; never touches the network."""
    base = (
        Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    )
    if not base.is_dir():
        raise FileNotFoundError(f"{repo}@{revision} is not in the local HF cache ({base})")
    return base


def shard_index(snapshot: Path) -> dict[str, str]:
    """tensor name -> shard filename (empty for unsharded checkpoints)."""
    index = snapshot / "model.safetensors.index.json"
    if not index.is_file():
        return {}
    with index.open() as f:
        data = json.load(f)
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index} has no weight_map")
    return {str(k): str(v) for k, v in weight_map.items()}


def read_tensor(snapshot: Path, shards: dict[str, str], name: str) -> torch.Tensor:
    path = snapshot / shards[name] if name in shards else snapshot / "model.safetensors"
    with safe_open(str(path), framework="pt") as f:
        available = f.keys()
        if name not in available:
            raise KeyError(f"{name} absent from {path.name}")
        return f.get_tensor(name)


def discover_layers(
    shards: dict[str, str], snapshot: Path, arch: str, num_experts: int
) -> list[int]:
    """Layer indices that carry a full router + expert set for ``arch``."""
    spec = ARCHES[arch]
    if shards:
        names = list(shards)
    else:
        with safe_open(str(snapshot / "model.safetensors"), framework="pt") as f:
            names = list(f.keys())
    prefix, suffix = spec["router"].split("{layer}")
    pattern = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix))
    layers = sorted({int(m.group(1)) for n in names if (m := pattern.fullmatch(n))})
    complete = []
    for layer in layers:
        needed = [
            spec["expert"].format(layer=layer, expert=e, role=r)
            for e in range(num_experts)
            for r in spec["roles"]
        ]
        if all(n in names for n in needed):
            complete.append(layer)
    return complete


def load_layer(
    snapshot: Path,
    shards: dict[str, str],
    arch: str,
    layer: int,
    num_experts: int,
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    """Parent experts (list indexed by slot) and router rows for one layer."""
    spec = ARCHES[arch]
    experts: list[dict[str, torch.Tensor]] = []
    for e in range(num_experts):
        expert = {}
        for role in spec["roles"]:
            name = spec["expert"].format(layer=layer, expert=e, role=role)
            tensor = read_tensor(snapshot, shards, name)
            if tensor.ndim != 2 or min(tensor.shape) == 0:
                raise ValueError(f"{name} has unusable shape {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} has non-finite values")
            expert[role] = tensor
        experts.append(expert)
    router = read_tensor(snapshot, shards, spec["router"].format(layer=layer))
    if router.ndim != 2 or router.shape[0] != num_experts:
        raise ValueError(f"router for layer {layer} has shape {tuple(router.shape)}")
    return experts, router


def role_std(experts: list[dict[str, torch.Tensor]], role: str) -> float:
    """Population std of one projection across every expert in the layer."""
    total = 0.0
    total_sq = 0.0
    count = 0
    for expert in experts:
        values = expert[role].to(torch.float64)
        total += float(values.sum())
        total_sq += float((values * values).sum())
        count += values.numel()
    mean = total / count
    return float(np.sqrt(max(total_sq / count - mean * mean, 0.0)))


# --------------------------------------------------------------------------- #
# variant construction
# --------------------------------------------------------------------------- #
def _mix(parts: list[torch.Tensor], weights: tuple[float, ...], dtype: torch.dtype) -> torch.Tensor:
    acc = torch.zeros(parts[0].shape, dtype=torch.float32)
    for part, weight in zip(parts, weights, strict=True):
        acc += float(weight) * part.to(torch.float32)
    return acc.to(dtype)


def build_layer(
    variant: str,
    layer: int,
    experts: list[dict[str, torch.Tensor]],
    router: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, dict]:
    """Derive one layer of ``variant``; returns (experts, router, ground truth)."""
    n = len(experts)
    roles = list(experts[0])
    dtype = experts[0][roles[0]].dtype

    if variant == "permuted":
        perm = rng.permutation(n)
        out = [{r: experts[int(perm[s])][r] for r in roles} for s in range(n)]
        return out, router[torch.as_tensor(perm)].clone(), {"perm": [int(p) for p in perm]}

    if variant == "pruned":
        dropped = sorted(int(d) for d in rng.choice(n, size=2, replace=False))
        kept = [i for i in range(n) if i not in dropped]
        out = [{r: experts[i][r] for r in roles} for i in kept]
        return out, router[torch.as_tensor(kept)].clone(), {"kept": kept, "dropped": dropped}

    if variant in ("merged", "merged_shallow", "merged_permuted"):
        if n < 3:
            raise ValueError(f"{variant} needs at least 3 experts, got {n}")
        weights = SHALLOW_ALPHA if variant == "merged_shallow" else MERGE_ALPHA
        merged = {r: _mix([experts[0][r], experts[1][r]], weights, dtype) for r in roles}
        out = [merged] + [{r: experts[i][r] for r in roles} for i in range(2, n)]
        merged_router = _mix([router[0], router[1]], weights, router.dtype)
        out_router = torch.cat([merged_router.unsqueeze(0), router[2:]], dim=0)
        truth = {
            "merge_parents": [0, 1],
            "merge_weights": list(weights),
            "suspect_slot": 0,
            "kept": list(range(2, n)),
        }
        if variant != "merged_permuted":
            return out, out_router.clone(), truth
        perm = rng.permutation(len(out))
        truth["perm"] = [int(p) for p in perm]
        # ``perm`` maps output slot -> pre-permutation slot, over the E-1
        # post-merge slots: slot 0 is the merged expert and slot j>=1 is parent
        # j+1. Record the truth in the written checkpoint's slot space.
        truth["suspect_slot"] = int(np.flatnonzero(perm == 0)[0])
        truth["kept"] = [int(np.flatnonzero(perm == p - 1)[0]) for p in range(2, n)]
        return (
            [out[int(perm[s])] for s in range(len(out))],
            out_router[torch.as_tensor(perm)].clone(),
            truth,
        )

    if variant == "control":
        out = []
        stds = {role: role_std(experts, role) for role in roles}
        for i in range(n):
            expert = {}
            for role in roles:
                noise = rng.standard_normal(tuple(experts[i][role].shape))
                expert[role] = torch.from_numpy(noise * stds[role]).to(dtype)
            out.append(expert)
        return out, router.clone(), {"random": True, "std": stds}

    raise ValueError(f"unknown variant {variant!r}")


def save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_file({name: tensor.contiguous() for name, tensor in tensors.items()}, str(path))


def write_variant(
    out_dir: Path, arch: str, tensors: dict[str, torch.Tensor], layers: dict[int, int], seed: int
) -> None:
    """Write one variant as safetensors + manifest (tensors only, not a loadable model)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tensors(out_dir / "model.safetensors", tensors)
    manifest = {
        "arch": arch,
        "seed": seed,
        "roles": list(ARCHES[arch]["roles"]),
        "layers": {str(layer): {"num_experts": count} for layer, count in sorted(layers.items())},
        "tensor_file": "model.safetensors",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def tensor_name(arch: str, layer: int, expert: int, role: str) -> str:
    return ARCHES[arch]["expert"].format(layer=layer, expert=expert, role=role)


def router_name(arch: str, layer: int) -> str:
    return ARCHES[arch]["router"].format(layer=layer)


def layer_rng(variant: str, layer: int, seed: int) -> np.random.Generator:
    """Independent stream per (variant, layer), keyed by the frozen variant ordinal."""
    if variant not in VARIANT_ORDINAL:
        raise ValueError(f"unknown variant {variant!r}")
    return np.random.default_rng([seed, VARIANT_ORDINAL[variant], layer])
