"""MoTHer adapted distance smoke — smokes/mother-distance/

Mean RMS ℓ_FT on matching square 2-D tensors + kurtosis direction hint (not full MDST).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _lib.hub import load_all_tensors  # noqa: E402
from _lib.pairs import default_pairs  # noqa: E402


def square_tensors(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: w
        for name, w in tensors.items()
        if w.ndim == 2 and w.shape[0] == w.shape[1] and "weight" in name
    }


def ku_sum(tensors: dict[str, np.ndarray]) -> tuple[float, int]:
    sq = square_tensors(tensors)
    if not sq:
        return 0.0, 0
    return float(sum(float(kurtosis(w.ravel(), fisher=False)) for w in sq.values())), len(sq)


def l_ft(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> tuple[float, int]:
    diffs: list[float] = []
    for name in sorted(set(a) & set(b)):
        if a[name].shape != b[name].shape:
            continue
        diffs.append(float(np.sqrt(np.mean((a[name] - b[name]) ** 2))))
    if not diffs:
        return float("nan"), 0
    return float(np.mean(diffs)), len(diffs)


def compare(ref: str, sus: str, cache_dir: Path | None) -> dict:
    ta = load_all_tensors(ref, cache_dir=cache_dir)
    tb = load_all_tensors(sus, cache_dir=cache_dir)
    sa = square_tensors(ta)
    sb = square_tensors(tb)
    mean_sq, n_sq = l_ft(sa, sb)
    mean_all, n_all = l_ft(
        {k: v for k, v in ta.items() if v.ndim == 2},
        {k: v for k, v in tb.items() if v.ndim == 2},
    )
    ku_ref, _ = ku_sum(ta)
    ku_sus, _ = ku_sum(tb)
    direction = "ref→sus (high-ku→low-ku)" if ku_ref >= ku_sus else "sus→ref"
    return {
        "l_ft_square": mean_sq,
        "n_square": n_sq,
        "l_ft_all2d": mean_all,
        "n_all2d": n_all,
        "ku_ref": ku_ref,
        "ku_sus": ku_sus,
        "direction_hint": direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MoTHer ℓ_FT smoke")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    for label, ref, sus in default_pairs():
        print(f"\n=== {label}: {ref} vs {sus} ===")
        t1 = time.perf_counter()
        out = compare(ref, sus, args.cache_dir)
        print(
            f"  ℓ_FT square={out['l_ft_square']:.5f} (n={out['n_square']}) "
            f"all2d={out['l_ft_all2d']:.5f} (n={out['n_all2d']})"
        )
        print(
            f"  ku_ref={out['ku_ref']:.2f} ku_sus={out['ku_sus']:.2f} "
            f"direction={out['direction_hint']}  ({time.perf_counter() - t1:.1f}s)"
        )
    print(f"\nTotal {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
