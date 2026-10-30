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

VARIANTS = (
    "permuted",
    "pruned",
    "merged",
    "merged_shallow",
    "merged_permuted",
    "control",
    "merged_skewed",
    "merged_three",
    "merged_saliency",
    "merged_noisy",
    "merged_ties",
)

# RNG stream ordinal per variant, frozen. The stream is keyed on this map, not on
# the position in VARIANTS, so inserting or reordering a variant cannot re-key
# the streams of the existing ones. Changing an ordinal is a re-baseline.
# Ordinals 0-5 are the original six-variant suite; 6-10 are the harder merges
# added without moving any existing stream.
VARIANT_ORDINAL = {
    "permuted": 0,
    "pruned": 1,
    "merged": 2,
    "merged_shallow": 3,
    "merged_permuted": 4,
    "control": 5,
    "merged_skewed": 6,
    "merged_three": 7,
    "merged_saliency": 8,
    "merged_noisy": 9,
    "merged_ties": 10,
}
_missing = [variant for variant in VARIANTS if variant not in VARIANT_ORDINAL]
if _missing:
    raise RuntimeError(f"VARIANTS without an RNG ordinal: {_missing}")
if len(set(VARIANT_ORDINAL.values())) != len(VARIANT_ORDINAL):
    raise RuntimeError("VARIANT_ORDINAL has duplicate streams")

# Descriptor: one BLOCK x BLOCK window per projection.
BLOCK = 32
# Pair classification floor, fixed a priori.
EXACT_COS = 0.9995
# "Close enough that matching alone would call it a copy" band, reported only.
NEAR_EXACT_COS = 0.995
# "A mixture explains this expert" floor, fixed a priori.
MERGE_GAIN_MIN = 0.30
# A component below this share is treated as fit noise, not a merge parent.
# 0.03 == SHALLOW_ALPHA[1] - ALPHA_TOL, so every alpha inside the pre-registered
# band clears the component gate (re-baselined from 0.04; see README).
MIN_COMPONENT = 0.03
# Injected mixture weights for (parent 0, parent 1) and the pre-registered band.
MERGE_ALPHA = (0.60, 0.40)
# Shallow merge: 5% of parent 1 is small enough that cosine matching alone
# calls the result an exact copy. The mixture fit must still find it.
SHALLOW_ALPHA = (0.95, 0.05)
ALPHA_TOL = 0.02

# --- harder merge shapes (pre-registered 2026-10-29, before the suite ran) ---
# Unequal two-parent merge.
SKEWED_ALPHA = (0.80, 0.20)
# Three-parent convex merges: a plain equal-ish blend and a REAM-shaped
# saliency weighting (0.7/0.2/0.1). The weights are injected, not estimated.
THREE_ALPHA = (0.50, 0.30, 0.20)
SALIENCY_ALPHA = (0.70, 0.20, 0.10)
# Merge-then-finetune drift: per-element Gaussian noise at this fraction of the
# merged expert's own per-role population std.
NOISY_SIGMA = 0.01
# TIES-shaped elementwise merge: trim this fraction of each parent's
# delta-to-mean by magnitude, elect the sign of the largest surviving delta per
# element, then average the surviving deltas. Deliberately non-convex.
TIES_TRIM = 0.20
# TIES has no global mixture weights. 1/3 per parent is the summary target used
# to score the fitted alpha (the exchangeable equal-share mean), not an injected
# construction weight, so the band is wider.
TIES_ALPHA = (1 / 3, 1 / 3, 1 / 3)
TIES_ALPHA_TOL = 0.10

# Expected global weights and band for every new merged variant. The component
# floor must not silently narrow a pre-registered claim: every injected
# component has to clear the detector's MIN_COMPONENT gate.
MERGED_VARIANT_ALPHA = {
    "merged_skewed": (SKEWED_ALPHA, ALPHA_TOL),
    "merged_three": (THREE_ALPHA, ALPHA_TOL),
    "merged_saliency": (SALIENCY_ALPHA, ALPHA_TOL),
    "merged_noisy": (MERGE_ALPHA, ALPHA_TOL),
    "merged_ties": (TIES_ALPHA, TIES_ALPHA_TOL),
}
for _variant, (_weights, _tol) in MERGED_VARIANT_ALPHA.items():
    if min(_weights) < MIN_COMPONENT:
        raise RuntimeError(
            f"{_variant}: injected component {min(_weights)} is below the detector "
            f"component floor {MIN_COMPONENT}; the pre-registered claim would be "
            "narrower than the variant"
        )

SCRATCH = REPO_ROOT / ".scratch" / "moe-expert-lineage"
RESULTS = ROOT / "results.json"

# The scale repeat is gated on the primary run's wall time. The gate exists to
# bound runtime; it was 120 s for the six-variant suite, then 300 s for the
# eleven-variant suite, and was raised to 600 s in review (runtime only,
# disclosed in the README). It gates only the OLMoE repeat, never a verdict.
SECONDARY_BUDGET_S = 600.0

# Expectation registry. ``origin`` is ``pre_registered`` when the claim was in
# README.md before the first run, and ``post_hoc`` when it was restated after
# observing a miss; a post-hoc claim points at the original text with
# ``origin_ref``. Claims are written for E experts; the TinyMixtral run
# instantiates E=4 (4/4, 2 kept, 3 experts), and the harder merges write E-1 or
# E-2 experts as stated in each claim.
EXPECTATIONS = (
    {
        "variant": "permuted",
        "origin": "pre_registered",
        "claim": "every layer matches E/E experts at cos >= 0.9995 and the recovered "
        "permutation equals the injected one",
    },
    {
        "variant": "pruned",
        "origin": "pre_registered",
        "claim": "every layer matches the E-2 survivors exactly, reports 2 missing "
        "experts, and yields no mixture fit reaching gain >= 0.30",
    },
    {
        "variant": "merged",
        "origin": "pre_registered",
        "claim": "every layer matches the E-2 untouched experts exactly and reports one new "
        "expert explained by the 0.60/0.40 mixture of the two missing parents "
        "(alpha within +/-0.02, gain >= 0.30)",
    },
    {
        "variant": "merged_shallow",
        "origin": "post_hoc",
        "origin_ref": "README.md, 'Second pre-registration, after a miss': original claim "
        "was 'the E−2 untouched experts match exactly and the merge is left as an exact "
        "copy'",
        "claim": "every layer flags exactly one merged expert with alpha within +/-0.02 of "
        "0.95/0.05 and gain >= 0.30, whether or not cosine matching would have called it a "
        "copy; on models with near-duplicate experts the copy call is the failure mode this "
        "variant guards",
    },
    {
        "variant": "merged_permuted",
        "origin": "pre_registered",
        "claim": "after alignment, identical to merged: E-2 exact matches, one new expert, "
        "alpha within +/-0.02 of 0.60/0.40, gain >= 0.30",
    },
    {
        "variant": "control",
        "origin": "pre_registered",
        "claim": "zero exact matches and no mixture fit reaching gain >= 0.30",
    },
    {
        "variant": "merged_skewed",
        "origin": "pre_registered",
        "claim": "every layer flags exactly one merged expert with alpha within +/-0.02 of "
        "0.80/0.20 and gain >= 0.30, and the E-2 untouched experts match exactly; on a "
        "near-duplicate MoE the merged expert may also clear the exact-cosine floor, which "
        "is reported as shadowing rather than scored as a failure",
    },
    {
        "variant": "merged_three",
        "origin": "pre_registered",
        "claim": "every layer flags exactly one merged expert whose components are parents "
        "0, 1 and 2, with alpha within +/-0.02 of 0.50/0.30/0.20 and gain >= 0.30, and the "
        "E-3 untouched experts match exactly; a fit that spreads weight onto a third-party "
        "parent is a miss and is the identifiability limit this variant probes",
    },
    {
        "variant": "merged_saliency",
        "origin": "pre_registered",
        "claim": "every layer flags exactly one merged expert whose components are parents "
        "0, 1 and 2, with alpha within +/-0.02 of 0.70/0.20/0.10 and gain >= 0.30, and the "
        "E-3 untouched experts match exactly",
    },
    {
        "variant": "merged_noisy",
        "origin": "pre_registered",
        "claim": "every layer flags exactly one merged expert with alpha within +/-0.02 of "
        "0.60/0.40 and gain >= 0.30, and the E-2 untouched experts match exactly; the "
        "mixture is no longer exact, so gain is expected to drop from ~1.0 but stay well "
        "above the floor",
    },
    {
        "variant": "merged_ties",
        "origin": "pre_registered",
        "claim": "every layer flags exactly one merged expert whose components are parents "
        "0, 1 and 2, with each alpha within +/-0.10 of the 1/3 summary target and gain "
        ">= 0.30, and the E-3 untouched experts match exactly. This is an approximate "
        "summary of an elementwise non-convex merge. A legitimate miss is ANY other honest "
        "outcome - gain below 0.30, a different alpha, or a different label - because TIES "
        "is outside the global convex span; it is expected when the merged parents are "
        "near-orthogonal (OLMoE), where no global alpha exists to recover",
    },
)
