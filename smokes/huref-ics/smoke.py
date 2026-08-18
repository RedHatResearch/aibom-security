"""HuRef ICS smoke — smokes/huref-ics/

NeurIPS 2024 invariant terms (Llama-style, last 2 layers). Not stock HuRef repo.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _lib.hub import (  # noqa: E402
    attn_tensor_names,
    embed_name,
    load_config,
    load_tensors,
    mlp_tensor_names,
)
from _lib.pairs import default_pairs  # noqa: E402

K = 4096
LAYERS = 2
CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning models inherit weights from base checkpoints. "
    "Verification checks lineage without trusting metadata alone."
)


def expand_kv(k: np.ndarray, n_heads: int, n_kv: int, head_dim: int) -> np.ndarray:
    if n_kv == n_heads:
        return k
    n_rep = n_heads // n_kv
    blocks = k.reshape(n_kv, head_dim, k.shape[1])
    return np.repeat(blocks, n_rep, axis=0).reshape(n_heads * head_dim, k.shape[1])


def rare_token_ids(tokenizer, k: int) -> np.ndarray:
    counts: Counter[int] = Counter()
    for word in CORPUS.split():
        ids = tokenizer.encode(word, add_special_tokens=False)
        for tid in ids:
            counts[tid] += 1
    ranked = sorted(counts.keys(), key=lambda t: counts[t])
    if len(ranked) < k:
        extra = [i for i in range(tokenizer.vocab_size) if i not in counts][-k:]
        ranked = (ranked + extra)[:k]
    return np.asarray(ranked[-k:], dtype=np.int64)


def invariant_stack(
    tensors: dict[str, np.ndarray],
    *,
    embed: np.ndarray,
    token_ids: np.ndarray,
    layer: int,
    n_heads: int,
    n_kv: int,
    head_dim: int,
) -> np.ndarray:
    x = embed[token_ids].astype(np.float64)
    q = tensors[f"model.layers.{layer}.self_attn.q_proj.weight"]
    k = tensors[f"model.layers.{layer}.self_attn.k_proj.weight"]
    v = tensors[f"model.layers.{layer}.self_attn.v_proj.weight"]
    o = tensors[f"model.layers.{layer}.self_attn.o_proj.weight"]
    gate = tensors[f"model.layers.{layer}.mlp.gate_proj.weight"]
    up = tensors[f"model.layers.{layer}.mlp.up_proj.weight"]
    down = tensors[f"model.layers.{layer}.mlp.down_proj.weight"]

    k_exp = expand_kv(k, n_heads, n_kv, head_dim)
    v_exp = expand_kv(v, n_heads, n_kv, head_dim)

    ma = x @ (q.T @ k_exp) @ x.T
    mb = x @ (v_exp.T @ o) @ x.T
    mf = x @ (down @ up) @ x.T
    return np.stack([ma, mb, mf], axis=0)


def ics(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    channels = a.shape[0]
    flat_a, flat_b = [], []
    for c in range(channels):
        xa = a[c].astype(np.float64)
        xb = b[c].astype(np.float64)
        xa = (xa - xa.mean()) / (xa.std() or 1.0)
        xb = (xb - xb.mean()) / (xb.std() or 1.0)
        flat_a.append(xa.ravel())
        flat_b.append(xb.ravel())
    va = np.concatenate(flat_a)
    vb = np.concatenate(flat_b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)) * 100.0)


def extract(repo_id: str, cache_dir: Path | None) -> np.ndarray:
    cfg = load_config(repo_id, cache_dir)
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg["hidden_size"] // n_heads
    names = attn_tensor_names(n_layers) | mlp_tensor_names(n_layers) | {embed_name()}
    tensors = load_tensors(repo_id, names, cache_dir=cache_dir)
    embed = tensors[embed_name()]
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    token_ids = rare_token_ids(tokenizer, K)
    blocks = []
    for layer in range(n_layers - LAYERS, n_layers):
        blocks.append(
            invariant_stack(
                tensors,
                embed=embed,
                token_ids=token_ids,
                layer=layer,
                n_heads=n_heads,
                n_kv=n_kv,
                head_dim=head_dim,
            )
        )
    return np.concatenate(blocks, axis=0)


def compare(ref: str, sus: str, cache_dir: Path | None) -> float:
    return ics(extract(ref, cache_dir), extract(sus, cache_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="HuRef ICS smoke")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    for label, ref, sus in default_pairs():
        print(f"\n=== {label}: {ref} vs {sus} ===")
        t1 = time.perf_counter()
        score = compare(ref, sus, args.cache_dir)
        print(f"  ICS = {score:.2f}  ({time.perf_counter() - t1:.1f}s)")
    print(f"\nTotal {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
