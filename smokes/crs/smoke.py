"""CRS smoke — smokes/crs/ · arXiv:2608.14929 · #26 survey.

https://github.com/RedHatResearch/aibom-security/tree/main/smokes/crs
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from scipy.optimize import linear_sum_assignment

BRANCH_NAMES = ("mlp", "qk", "vo")


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


@dataclass
class BranchSignature:
    phi: np.ndarray
    s: float


@dataclass
class LayerBranches:
    branches: dict[str, BranchSignature]


@dataclass
class ModelCRS:
    config: ModelConfig
    layers: list[LayerBranches]
    load_seconds: float


def load_config(repo_id: str, cache_dir: Path | None) -> ModelConfig:
    config_path = hf_hub_download(repo_id, filename="config.json", cache_dir=cache_dir)
    with open(config_path) as f:
        raw = json.load(f)
    return ModelConfig(
        repo_id=repo_id,
        hidden_size=raw["hidden_size"],
        num_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
        intermediate_size=raw["intermediate_size"],
    )


def _expand_kv_heads(weight: np.ndarray, n_heads: int, n_kv_heads: int, head_dim: int) -> np.ndarray:
    """Repeat KV heads to match Q/O head count (GhostSpec-style GQA handling)."""
    if n_kv_heads == n_heads:
        return weight
    n_rep = n_heads // n_kv_heads
    blocks = weight.reshape(n_kv_heads, head_dim, weight.shape[1])
    expanded = np.repeat(blocks, n_rep, axis=0)
    return expanded.reshape(n_heads * head_dim, weight.shape[1])


def trace_concentration(matrix: np.ndarray) -> float:
    fro = np.linalg.norm(matrix, ord="fro")
    if fro == 0.0:
        return 0.0
    return abs(np.trace(matrix)) / fro


def centered_signature(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    d = matrix.shape[0]
    s = trace_concentration(matrix)
    trace_mean = np.trace(matrix) / d
    residual = matrix - trace_mean * np.eye(d, dtype=matrix.dtype)
    flat = residual.ravel()
    norm = np.linalg.norm(flat)
    if norm == 0.0:
        phi = np.zeros_like(flat)
    else:
        phi = flat / norm
    return phi, s


def branch_product_mlp(down: np.ndarray, up: np.ndarray) -> np.ndarray:
    return down @ up


def branch_product_qk(q: np.ndarray, k: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    k_exp = _expand_kv_heads(k, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
    return q @ k_exp.T


def branch_product_vo(o: np.ndarray, v: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    v_exp = _expand_kv_heads(v, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
    return o @ v_exp


def _tensor_names_for_layer(layer_idx: int) -> list[str]:
    prefix = f"model.layers.{layer_idx}"
    return [
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    ]


def load_model_crs(repo_id: str, cache_dir: Path | None) -> ModelCRS:
    t0 = time.perf_counter()
    cfg = load_config(repo_id, cache_dir)
    model_dir = snapshot_download(
        repo_id,
        allow_patterns=["*.safetensors", "model.safetensors.index.json"],
        cache_dir=cache_dir,
    )
    model_path = Path(model_dir)

    shard_files = sorted(model_path.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors shards in {model_dir}")

    needed: set[str] = set()
    for layer_idx in range(cfg.num_layers):
        needed.update(_tensor_names_for_layer(layer_idx))

    tensors: dict[str, np.ndarray] = {}
    for shard in shard_files:
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                if name in needed:
                    tensors[name] = handle.get_tensor(name).to(dtype=torch.float32).numpy()

    missing = needed - tensors.keys()
    if missing:
        raise KeyError(f"Missing tensors for {repo_id}: {sorted(missing)[:5]} ...")

    layers: list[LayerBranches] = []
    for layer_idx in range(cfg.num_layers):
        prefix = f"model.layers.{layer_idx}"
        q = tensors[f"{prefix}.self_attn.q_proj.weight"]
        k = tensors[f"{prefix}.self_attn.k_proj.weight"]
        v = tensors[f"{prefix}.self_attn.v_proj.weight"]
        o = tensors[f"{prefix}.self_attn.o_proj.weight"]
        up = tensors[f"{prefix}.mlp.up_proj.weight"]
        down = tensors[f"{prefix}.mlp.down_proj.weight"]

        products = {
            "mlp": branch_product_mlp(down, up),
            "qk": branch_product_qk(q, k, cfg),
            "vo": branch_product_vo(o, v, cfg),
        }
        branches: dict[str, BranchSignature] = {}
        for name, matrix in products.items():
            phi, s = centered_signature(matrix)
            branches[name] = BranchSignature(phi=phi, s=s)
        layers.append(LayerBranches(branches=branches))

    load_seconds = time.perf_counter() - t0
    return ModelCRS(config=cfg, layers=layers, load_seconds=load_seconds)


def compatibility(a: ModelCRS, b: ModelCRS) -> tuple[bool, str]:
    if a.config.hidden_size != b.config.hidden_size:
        return False, f"hidden_size mismatch {a.config.hidden_size} vs {b.config.hidden_size}"
    if a.config.num_layers != b.config.num_layers:
        return False, f"num_layers mismatch {a.config.num_layers} vs {b.config.num_layers}"
    if a.config.num_attention_heads != b.config.num_attention_heads:
        return (
            False,
            f"num_attention_heads mismatch "
            f"{a.config.num_attention_heads} vs {b.config.num_attention_heads}",
        )
    if a.config.head_dim != b.config.head_dim:
        return False, f"head_dim mismatch {a.config.head_dim} vs {b.config.head_dim}"
    return True, "compatible"


def _branch_dot(a: BranchSignature, b: BranchSignature) -> float:
    return float(np.dot(a.phi, b.phi))


def _gate_factor(s_a: float, s_b: float, tau_s: float) -> float:
    if tau_s <= 0.0:
        return 0.0
    return min(s_a / tau_s, s_b / tau_s, 1.0)


def compute_lineage(reference: ModelCRS, suspect: ModelCRS) -> dict:
    ok, reason = compatibility(reference, suspect)
    if not ok:
        return {
            "compatible": False,
            "reason": reason,
            "lineage_score": None,
            "per_branch_scores": None,
            "permutation": None,
            "tau_s": None,
            "compute_seconds": 0.0,
        }

    t0 = time.perf_counter()

    ref_s_values = [
        layer.branches[name].s for layer in reference.layers for name in BRANCH_NAMES
    ]
    tau_s = min(ref_s_values)

    layer_count = reference.config.num_layers
    gated_g = np.zeros((layer_count, layer_count), dtype=np.float64)
    raw_dots = np.zeros((layer_count, layer_count), dtype=np.float64)

    for i in range(layer_count):
        for j in range(layer_count):
            dots = []
            gated = []
            for name in BRANCH_NAMES:
                sig_a = reference.layers[i].branches[name]
                sig_b = suspect.layers[j].branches[name]
                dot = _branch_dot(sig_a, sig_b)
                gate = _gate_factor(sig_a.s, sig_b.s, tau_s)
                dots.append(dot)
                gated.append(dot * gate)
            raw_dots[i, j] = float(np.mean(dots))
            gated_g[i, j] = float(np.mean(gated))

    row_ind, col_ind = linear_sum_assignment(-gated_g)
    permutation = {int(i): int(col_ind[i]) for i in row_ind}

    aligned_scores: list[float] = []
    per_branch_aligned: dict[str, list[float]] = {name: [] for name in BRANCH_NAMES}
    for i, j in zip(row_ind, col_ind, strict=True):
        for name in BRANCH_NAMES:
            dot = _branch_dot(reference.layers[i].branches[name], suspect.layers[j].branches[name])
            per_branch_aligned[name].append(dot)
            aligned_scores.append(dot)

    lineage_score = float(np.mean(aligned_scores))
    compute_seconds = time.perf_counter() - t0

    identity_score = float(np.mean([raw_dots[i, i] for i in range(layer_count)]))

    return {
        "compatible": True,
        "reason": reason,
        "lineage_score": lineage_score,
        "identity_alignment_score": identity_score,
        "per_branch_mean": {name: float(np.mean(vals)) for name, vals in per_branch_aligned.items()},
        "permutation": permutation,
        "permutation_is_identity": permutation == {i: i for i in range(layer_count)},
        "tau_s": tau_s,
        "compute_seconds": compute_seconds,
    }


def run_pair(
    ref_repo: str,
    sus_repo: str,
    label: str,
    cache_dir: Path | None,
    ref_cache: dict[str, ModelCRS],
    sus_cache: dict[str, ModelCRS],
) -> dict:
    if ref_repo not in ref_cache:
        print(f"Loading reference {ref_repo} ...")
        ref_cache[ref_repo] = load_model_crs(ref_repo, cache_dir)
    if sus_repo not in sus_cache:
        print(f"Loading suspect {sus_repo} ...")
        sus_cache[sus_repo] = load_model_crs(sus_repo, cache_dir)

    ref = ref_cache[ref_repo]
    sus = sus_cache[sus_repo]
    result = compute_lineage(ref, sus)
    result.update(
        {
            "label": label,
            "reference": ref_repo,
            "suspect": sus_repo,
            "ref_load_seconds": ref.load_seconds,
            "sus_load_seconds": sus.load_seconds,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CRS smoke (smokes/crs/)")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    pairs = [
        (
            "positive",
            "HuggingFaceTB/SmolLM2-1.7B",
            "SultanR/SmolTulu-1.7b-Instruct",
        ),
        (
            "vocab_drift",
            "meta-llama/Llama-3.2-1B",
            "dphn/Dolphin3.0-Llama3.2-1B",
        ),
        (
            "negative",
            "HuggingFaceTB/SmolLM2-1.7B",
            "meta-llama/Llama-3.2-1B",
        ),
    ]

    ref_cache: dict[str, ModelCRS] = {}
    sus_cache: dict[str, ModelCRS] = {}
    results: list[dict] = []

    total_start = time.perf_counter()
    for label, ref_repo, sus_repo in pairs:
        print(f"\n=== {label}: {ref_repo} vs {sus_repo} ===")
        outcome = run_pair(ref_repo, sus_repo, label, args.cache_dir, ref_cache, sus_cache)
        results.append(outcome)
        if outcome["compatible"]:
            print(f"  L = {outcome['lineage_score']:.6f}")
            print(f"  identity-order L = {outcome['identity_alignment_score']:.6f}")
            print(f"  per-branch means: {outcome['per_branch_mean']}")
            print(f"  tau_s = {outcome['tau_s']:.6f}")
            print(f"  permutation identity? {outcome['permutation_is_identity']}")
            print(f"  compute_seconds = {outcome['compute_seconds']:.3f}")
        else:
            print(f"  INCOMPATIBLE: {outcome['reason']}")
        print(
            f"  load_seconds ref={outcome['ref_load_seconds']:.1f} "
            f"sus={outcome['sus_load_seconds']:.1f}"
        )

    total_seconds = time.perf_counter() - total_start
    print(f"\nTotal wall time: {total_seconds:.1f}s")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
