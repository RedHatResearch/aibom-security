"""MoE router Gram fingerprint smoke — smokes/moe-router-gram/

Gate-weight row Gram matrices + Hungarian expert alignment. Uses verifier ranged tensor I/O.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import FIXTURES, SEED  # noqa: E402
from hub_io import load_gates  # noqa: E402
from match import layer_scores  # noqa: E402


def mean_aligned(wa: list[np.ndarray], wb: list[np.ndarray]) -> float:
    n = min(len(wa), len(wb))
    scores = [float(layer_scores(wa[i], wb[i])["aligned"]) for i in range(n)]
    return float(np.mean(scores))


def main() -> None:
    parent = FIXTURES["parent"]
    child = FIXTURES["child"]
    negative = FIXTURES["negative"]

    t0 = time.perf_counter()
    print(f"loading {parent} …")
    wp = load_gates(parent)
    print(f"loading {child} …")
    wc = load_gates(child)
    print(f"loading {negative} …")
    wn = load_gates(negative)

    p = mean_aligned(wp, wc)
    h = mean_aligned(wp, wn)
    rng = np.random.default_rng(SEED)
    u = float(layer_scores(rng.normal(size=wp[0].shape), rng.normal(size=wp[0].shape))["aligned"])

    print(f"\npositive parent↔child  aligned={p:.6f}")
    print(f"hard_neg parent↔{negative.split('/')[-1]}  aligned={h:.6f}")
    print(f"null gaussian              aligned={u:.6f}")
    print(f"\nTotal {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
