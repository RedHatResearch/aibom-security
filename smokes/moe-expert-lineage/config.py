"""Pinned constants for the MoE expert-lineage probe (Experiment A, issue #26).

Everything here is fixed before the run. Changing a value after a run is a
re-baseline: record the change and the new numbers in README.md.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]

# One seed for every permutation, prune choice, and Gaussian control tensor.
SEED = 20261029

PRIMARY = {
    "label": "tiny_mixtral",
    "repo": "Isotonic/TinyMixtral-4x248M-MoE",
    "revision": "1e3516176a6279ce93923060fcad848dc043af79",
    "arch": "mixtral",
    "num_experts": 4,
    "layers": None,  # every MoE layer found in the checkpoint
}

SECONDARY = {
    "label": "olmoe",
    "repo": "allenai/OLMoE-1B-7B-0924",
    "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    "arch": "olmoe",
    "num_experts": 64,
    "layers": [0, 8],
}

ARCHES = {
    "mixtral": {
        "expert": "model.layers.{layer}.block_sparse_moe.experts.{expert}.{role}.weight",
        "router": "model.layers.{layer}.block_sparse_moe.gate.weight",
        "roles": ("w1", "w2", "w3"),
    },
    "olmoe": {
        "expert": "model.layers.{layer}.mlp.experts.{expert}.{role}.weight",
        "router": "model.layers.{layer}.mlp.gate.weight",
        "roles": ("gate_proj", "up_proj", "down_proj"),
    },
}

VARIANTS = ("permuted", "pruned", "merged", "merged_shallow", "merged_permuted", "control")

# Descriptor: one BLOCK x BLOCK window per projection.
BLOCK = 32
# Pair classification floor, fixed a priori.
EXACT_COS = 0.9995
# "Close enough that matching alone would call it a copy" band, reported only.
NEAR_EXACT_COS = 0.995
# "A mixture explains this expert" floor, fixed a priori.
MERGE_GAIN_MIN = 0.30
# A component below this share is treated as fit noise, not a merge parent.
MIN_COMPONENT = 0.04
# Injected mixture weights for (parent 0, parent 1) and the pre-registered band.
MERGE_ALPHA = (0.60, 0.40)
# Shallow merge: 5% of parent 1 is small enough that cosine matching alone
# calls the result an exact copy. The mixture fit must still find it.
SHALLOW_ALPHA = (0.95, 0.05)
ALPHA_TOL = 0.02

SCRATCH = REPO_ROOT / ".scratch" / "moe-expert-lineage"
RESULTS = ROOT / "results.json"

# The scale repeat is gated on the primary run's wall time.
SECONDARY_BUDGET_S = 120.0

# Pre-registered expectations, stated before the first run. Claims are written
# for E experts; the TinyMixtral run instantiates E=4 (4/4, 2 kept, 3 experts).
EXPECTATIONS = (
    {
        "variant": "permuted",
        "claim": "every layer matches E/E experts at cos >= 0.9995 and the recovered "
        "permutation equals the injected one",
    },
    {
        "variant": "pruned",
        "claim": "every layer matches the E-2 survivors exactly, reports 2 missing "
        "experts, and yields no mixture fit reaching gain >= 0.30",
    },
    {
        "variant": "merged",
        "claim": "every layer matches the E-2 untouched experts exactly and reports one new "
        "expert explained by the 0.60/0.40 mixture of the two missing parents "
        "(alpha within +/-0.02, gain >= 0.30)",
    },
    {
        "variant": "merged_shallow",
        "claim": "every layer flags exactly one merged expert with alpha within +/-0.02 of "
        "0.95/0.05 and gain >= 0.30, whether or not cosine matching would have called it a "
        "copy; on models with near-duplicate experts the copy call is the failure mode this "
        "variant guards",
    },
    {
        "variant": "merged_permuted",
        "claim": "after alignment, identical to merged: E-2 exact matches, one new expert, "
        "alpha within +/-0.02 of 0.60/0.40, gain >= 0.30",
    },
    {
        "variant": "control",
        "claim": "zero exact matches and no mixture fit reaching gain >= 0.30",
    },
)
