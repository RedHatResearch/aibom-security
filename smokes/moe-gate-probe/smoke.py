"""MoE gate probe smoke — smokes/moe-gate-probe/ · TensorLock Eq. 1 · #17.

Gate-only probe of the TensorLock (doi:10.1145/3832119) Eq. 1
attention-statistic gate on native MoE models: s_MSA, the mean Spearman
correlation of the per-layer (std, skew, kurt) moment curves of the self_attn
q/k/v/o projections across two checkpoints (truncated to the shorter stack),
gated at tau = 0.6. The gate reads only the shared attention stack — MoE
experts and routers are invisible to it by design — so the question is
whether a gate validated on dense models still separates same-lineage from
cross-family pairs when one or both sides are MoE (OLMoE base <-> SFT).
Feeds #17 (moe-router-gram family of signals); QPS, FT-direction and
router/expert signals are follow-ups, not part of this probe.

https://github.com/RedHatResearch/aibom-security/issues/17
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
from scipy import stats

SMOKE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(SMOKE_DIR.parent))

from _lib.hub import load_config, load_tensors  # noqa: E402
from config import FIXTURES, REVISIONS  # noqa: E402

# Eq. 1 gate threshold: s_MSA >= tau counts as same-lineage.
GATE_TAU = 0.6

MSA_ROLES = ("q", "k", "v", "o")
ROLE_CURVES = {  # per-layer moment curves compared across models
    role: ("std", "skew", "kurt") for role in MSA_ROLES
}

# One role's moment curves: stat name -> per-layer values.
RoleCurves = dict[str, list[float]]
# One model's features: attention role -> that role's curves.
MsaFeatures = dict[str, RoleCurves]


# ---------------------------------------------------------------------------
# Eq. 1 gate, ported from smokes/tensorlock/smoke.py (same math).
# ---------------------------------------------------------------------------


def _safe_spearman(x: list[float], y: list[float]) -> float:
    """Spearman correlation; 0.0 when undefined (no variance or <2 layers)."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(stats.spearmanr(x, y).statistic)


def msa_features(mats: list[np.ndarray]) -> RoleCurves:
    """Per-layer (std, skew, kurt) moment curves for one attention role.

    mats: list of 2-D arrays, one per layer, in layer order.
    """
    return {
        "std": [float(np.std(m)) for m in mats],
        "skew": [float(stats.skew(m, axis=None)) for m in mats],
        "kurt": [float(stats.kurtosis(m, axis=None)) for m in mats],
    }


def role_similarities(features_a: MsaFeatures, features_b: MsaFeatures) -> dict[str, float]:
    """Per-role components of s_MSA: mean Spearman of each role's aligned
    moment curves, truncated to L_min. s_MSA is the mean of these values."""
    out = {}
    for role in MSA_ROLES:
        role_sims = []
        for stat_name in ROLE_CURVES[role]:
            a = features_a[role][stat_name]
            b = features_b[role][stat_name]
            n = min(len(a), len(b))
            role_sims.append(_safe_spearman(a[:n], b[:n]))
        out[role] = float(mean(role_sims))
    return out


def msa_similarity(features_a: MsaFeatures, features_b: MsaFeatures) -> float:
    """Eq. 1 s_MSA: mean Spearman of aligned moment curves, truncated to L_min."""
    return float(mean(role_similarities(features_a, features_b).values()))


# ---------------------------------------------------------------------------
# Hub loading — attention-role matrices only.
# ---------------------------------------------------------------------------


def attn_names(n_layers: int) -> list[str]:
    """Llama-style self_attn q/k/v/o projection names for every layer."""
    return [
        f"model.layers.{i}.self_attn.{r}_proj.weight" for i in range(n_layers) for r in MSA_ROLES
    ]


def load_attn_features(repo_id: str, cache_dir: Path | None) -> tuple[int, MsaFeatures]:
    """Moment-curve features of one checkpoint's shared attention stack.

    Loads only the self_attn projections at the pinned revision
    (config.REVISIONS); OLMoE's q_norm/k_norm, mlp.experts.* and mlp.gate are
    never requested.
    """
    n_layers = load_config(repo_id, cache_dir)["num_hidden_layers"]
    tensors = load_tensors(
        repo_id, set(attn_names(n_layers)), cache_dir=cache_dir, revision=REVISIONS[repo_id]
    )
    features = {
        role: msa_features(
            [tensors[f"model.layers.{i}.self_attn.{role}_proj.weight"] for i in range(n_layers)]
        )
        for role in MSA_ROLES
    }
    return n_layers, features


# ---------------------------------------------------------------------------
# Fixture runner.
# ---------------------------------------------------------------------------


def expectation_held(fixture: dict, score: float, gate: bool) -> bool:
    """Registered expectation: gate direction plus the score band, if any."""
    if gate != fixture["expect_gate"]:
        return False
    lo = fixture.get("expect_score_min")
    hi = fixture.get("expect_score_max")
    return (lo is None or score >= lo) and (hi is None or score <= hi)


def run_fixture(
    fixture: dict, cache_dir: Path | None, cache: dict[str, tuple[int, MsaFeatures]]
) -> dict:
    """Score one pair end-to-end and return its JSON entry."""
    label = fixture["label"]
    t0 = time.perf_counter()
    print(f"=== {label}: {fixture['a']} vs {fixture['b']} ===", file=sys.stderr)

    layers: dict[str, int] = {}
    features: dict[str, MsaFeatures] = {}
    for side in ("a", "b"):
        repo_id = fixture[side]
        if repo_id not in cache:
            print(f"loading attention stack of {repo_id} …", file=sys.stderr)
            cache[repo_id] = load_attn_features(repo_id, cache_dir)
        layers[side], features[side] = cache[repo_id]

    per_role = role_similarities(features["a"], features["b"])
    score = msa_similarity(features["a"], features["b"])
    gate = score >= GATE_TAU
    held = expectation_held(fixture, score, gate)

    print(f"{label}: s_msa={score:.4f} gate={gate} held={held}", file=sys.stderr)
    return {
        "label": label,
        "a": {
            "repo_id": fixture["a"],
            "revision": REVISIONS[fixture["a"]],
            "num_layers": layers["a"],
        },
        "b": {
            "repo_id": fixture["b"],
            "revision": REVISIONS[fixture["b"]],
            "num_layers": layers["b"],
        },
        "relation": fixture["relation"],
        "l_min": min(layers["a"], layers["b"]),
        "s_msa": {"overall": score, "per_role": per_role},
        "gate": gate,
        "expectation_held": held,
        "wall_seconds": time.perf_counter() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MoE gate probe smoke (smokes/moe-gate-probe/)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HF cache directory (default: the shared HF cache, reusing existing downloads)",
    )
    parser.add_argument("--only", default=None, help="run a single fixture set by label")
    args = parser.parse_args()

    # Validate the label before any Hub call so a typo cannot trigger a download.
    fixtures = [f for f in FIXTURES if args.only in (None, f["label"])]
    if args.only is not None and not fixtures:
        parser.error(f"unknown fixture label: {args.only}")

    total_start = time.perf_counter()
    cache: dict[str, tuple[int, MsaFeatures]] = {}
    results = []
    for fixture in fixtures:
        results.append(run_fixture(fixture, args.cache_dir, cache))

    output = {
        "method": "moe-gate-probe",
        "paper": "TensorLock (doi:10.1145/3832119), Eq. 1 s_MSA gate",
        "issue": 17,
        "gate_tau": GATE_TAU,
        "fixtures": results,
        "total_seconds": time.perf_counter() - total_start,
    }
    print(json.dumps(output, indent=2, default=float))
    sys.exit(0 if all(r["expectation_held"] for r in results) else 1)


if __name__ == "__main__":
    main()
