"""Intrinsic Fingerprint σ-curve smoke — smokes/intrinsic-fingerprint/

Withdrawn arXiv:2507.03014 / PDF recipe: per-layer std of Q/K/V/O → z-scored Pearson curves.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _lib.hub import attn_tensor_names, load_config, load_tensors  # noqa: E402
from _lib.pairs import default_pairs  # noqa: E402

ROLES = ("q", "k", "v", "o")


def sigma_curves(tensors: dict[str, np.ndarray], num_layers: int) -> dict[str, np.ndarray]:
    curves: dict[str, list[float]] = {r: [] for r in ROLES}
    for layer in range(num_layers):
        for role in ROLES:
            key = f"model.layers.{layer}.self_attn.{role}_proj.weight"
            curves[role].append(float(np.std(tensors[key])))
    return {r: np.asarray(v, dtype=np.float64) for r, v in curves.items()}


def zscore(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    if std == 0:
        return x * 0.0
    return (x - float(np.mean(x))) / std


def interpolate_to(a: np.ndarray, length: int) -> np.ndarray:
    if len(a) == length:
        return a
    xs = np.linspace(0, len(a) - 1, num=length)
    idx = np.arange(len(a))
    return np.interp(xs, idx, a)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        shorter, longer = (a, b) if len(a) < len(b) else (b, a)
        a = interpolate_to(shorter, len(longer))
        b = longer
    za, zb = zscore(a), zscore(b)
    return float(np.corrcoef(za, zb)[0, 1])


def compare(repo_a: str, repo_b: str, cache_dir: Path | None) -> dict:
    cfg_a = load_config(repo_a, cache_dir)
    cfg_b = load_config(repo_b, cache_dir)
    la = cfg_a["num_hidden_layers"]
    lb = cfg_b["num_hidden_layers"]
    ta = load_tensors(repo_a, attn_tensor_names(la), cache_dir=cache_dir)
    tb = load_tensors(repo_b, attn_tensor_names(lb), cache_dir=cache_dir)
    ca = sigma_curves(ta, la)
    cb = sigma_curves(tb, lb)
    per_role = {r: pearson(ca[r], cb[r]) for r in ROLES}
    overall = float(np.mean(list(per_role.values())))
    return {"overall": overall, "per_role": per_role, "layers": (la, lb)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Intrinsic Fingerprint σ-curve smoke")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    for label, ref, sus in default_pairs():
        print(f"\n=== {label}: {ref} vs {sus} ===")
        out = compare(ref, sus, args.cache_dir)
        print(f"  overall = {out['overall']:.5f}  layers = {out['layers']}")
        print(f"  per_role = {out['per_role']}")
    print(f"\nTotal {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
