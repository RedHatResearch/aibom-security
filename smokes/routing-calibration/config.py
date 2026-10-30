"""Pinned fixtures and pre-registration for the routing-calibration smoke.

Everything here is frozen before the first run: model revisions, the probe
set and QDQ contract (reused by import from ``smokes/routing-agreement``),
the populations, the threshold rule, and the expectation bands. Thresholds
are never re-tuned after seeing results; a wrong band is recorded as a miss.

Question (issue 69): is routing overlap — the Jaccard overlap of per-layer
top-k expert sets — a usable lineage signal, i.e. does the same model after
4-bit requantization stay much closer to itself than a different checkpoint
of the same architecture? The same-architecture / different-checkpoint pair
is the hard negative; only it, never a label, can veto the signal.

Probe reuse: ``PROMPTS``/``SEQ_LEN``/``DOMAINS`` and the QDQ constants are
loaded from ``smokes/routing-agreement/config.py`` by file path (unique
module name, so this ``config`` does not shadow it). ``quant.py`` is imported
by ``smoke.py`` from the same directory; neither file is copied. Their SHA256
digests are recorded in ``results.json`` for provenance.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ISSUE = 69
ISSUE_URL = "https://github.com/RedHatResearch/aibom-security/issues/69"

# Not used to draw anything: the probe and QDQ are fully deterministic. It is
# recorded so a future run can prove it used the same seed.
SEED = 20261029

# ---------------------------------------------------------------------------
# Reuse the frozen probe + QDQ contract from smokes/routing-agreement
# ---------------------------------------------------------------------------

_RA_DIR = Path(__file__).resolve().parent.parent / "routing-agreement"
_RA_CONFIG_PATH = _RA_DIR / "config.py"
_QUANT_PATH = _RA_DIR / "quant.py"

_spec = importlib.util.spec_from_file_location("_routing_agreement_config", _RA_CONFIG_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - import wiring
    raise RuntimeError(f"cannot load probe source {_RA_CONFIG_PATH}")
_RA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_RA)

SEQ_LEN = _RA.SEQ_LEN
DOMAINS = _RA.DOMAINS
PROMPTS = _RA.PROMPTS

# The QDQ grid is defined once, in quant.py; the routing config no longer
# restates it, so read it from the module that applies it.
_spec_q = importlib.util.spec_from_file_location("_routing_agreement_quant", _QUANT_PATH)
if _spec_q is None or _spec_q.loader is None:  # pragma: no cover - import wiring
    raise RuntimeError(f"cannot load QDQ source {_QUANT_PATH}")
_QUANT = importlib.util.module_from_spec(_spec_q)
_spec_q.loader.exec_module(_QUANT)
QUANT_BITS = _QUANT.BITS
QUANT_BLOCK = _QUANT.BLOCK

_REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE_SOURCE = {
    "config_path": str(_RA_CONFIG_PATH.relative_to(_REPO_ROOT)),
    "quant_path": str(_QUANT_PATH.relative_to(_REPO_ROOT)),
    "prompts": len(PROMPTS),
    "seq_len": SEQ_LEN,
    "domains": list(DOMAINS),
    "note": "probe and QDQ contract imported, not copied",
}

# ---------------------------------------------------------------------------
# Checkpoints (all pinned; all loaded offline)
# ---------------------------------------------------------------------------
# ``mode``: "requant" runs base -> experts_q4 -> all_q4 in one load;
#           "base" captures the untouched checkpoint only.
# ``tokenizer`` is explicit because two snapshots ship no tokenizer files:
# OLMoE-...-SFT (same tokenizer as 0924) and TinyLLama-4x1.1B-MoE (Llama-2
# SentencePiece, byte-identical to the Mistral tokenizer shipped with
# TinyMixtral; verified by tokenizer.model SHA256 and special-token ids).

TINYMIXTRAL = {
    "label": "tinymixtral",
    "repo_id": "Isotonic/TinyMixtral-4x248M-MoE",
    "revision": "1e3516176a6279ce93923060fcad848dc043af79",
    "dtype": "float32",
    "n_layers": 12,
    "n_experts": 4,
    "top_k": 2,
    "tokenizer": {
        "repo_id": "Isotonic/TinyMixtral-4x248M-MoE",
        "revision": "1e3516176a6279ce93923060fcad848dc043af79",
    },
    "mode": "requant",
    "role": "same-model requantization + different-family reference (E=4, k=2)",
}

OLMOE_0924 = {
    "label": "olmoe_0924",
    "repo_id": "allenai/OLMoE-1B-7B-0924",
    "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    "dtype": "bfloat16",
    "n_layers": 16,
    "n_experts": 64,
    "top_k": 8,
    "tokenizer": {
        "repo_id": "allenai/OLMoE-1B-7B-0924",
        "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    },
    "mode": "requant",
    "role": "same-model requantization reference (E=64, k=8)",
}

OLMOE_0924_SFT = {
    "label": "olmoe_0924_sft",
    "repo_id": "allenai/OLMoE-1B-7B-0924-SFT",
    "revision": "215cc4f73147dd68bd11e7a7dcc56bac397f4221",
    "dtype": "bfloat16",
    "n_layers": 16,
    "n_experts": 64,
    "top_k": 8,
    "tokenizer": {
        "repo_id": "allenai/OLMoE-1B-7B-0924",
        "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    },
    "mode": "base",
    "role": "same-architecture different-checkpoint hard negative (E=64, k=8)",
}

TINYLLAMA_MOE = {
    "label": "tinyllama_moe",
    "repo_id": "s3nh/TinyLLama-4x1.1B-MoE",
    "revision": "e82b7308fe3a401995169eb99a85492a36df5b83",
    "dtype": "bfloat16",
    "n_layers": 22,
    "n_experts": 4,
    "top_k": 2,
    "tokenizer": {
        "repo_id": "Isotonic/TinyMixtral-4x248M-MoE",
        "revision": "1e3516176a6279ce93923060fcad848dc043af79",
    },
    "mode": "base",
    "role": "different family, matching E=4/k=2 (Mixtral-layout mergekit MoE)",
}

MODELS = {m["label"]: m for m in (TINYMIXTRAL, OLMOE_0924, OLMOE_0924_SFT, TINYLLAMA_MOE)}

# Execution order: the hard-negative pair is the decisive measurement, so the
# two OLMoE checkpoints run first. Each model runs in its own subprocess so
# peak RSS is per-checkpoint, never the sum of two 13.8 GB checkpoints.
WORKER_ORDER = ("olmoe_0924", "olmoe_0924_sft", "tinymixtral", "tinyllama_moe")

# ---------------------------------------------------------------------------
# Populations (all comparisons are base-vs-candidate, same probe rows)
# ---------------------------------------------------------------------------

REQUANT_PAIRS = (
    {
        "label": "tinymixtral_experts_q4",
        "model": "tinymixtral",
        "candidate": "experts_q4",
        "condition": "experts 4-bit block-32 QDQ, router kept at model precision",
    },
    {
        "label": "tinymixtral_all_q4",
        "model": "tinymixtral",
        "candidate": "all_q4",
        "condition": "experts + router 4-bit block-32 QDQ",
    },
    {
        "label": "olmoe_0924_experts_q4",
        "model": "olmoe_0924",
        "candidate": "experts_q4",
        "condition": "experts 4-bit block-32 QDQ, router kept at model precision",
    },
    {
        "label": "olmoe_0924_all_q4",
        "model": "olmoe_0924",
        "candidate": "all_q4",
        "condition": "experts + router 4-bit block-32 QDQ",
    },
)

CHECKPOINT_PAIRS = (
    {
        "label": "olmoe_0924_vs_0924_sft",
        "reference": ("olmoe_0924", "base"),
        "candidate": ("olmoe_0924_sft", "base"),
        "n_experts": 64,
        "top_k": 8,
        "role": "hard negative: same architecture, different checkpoint",
    },
)

FAMILY_PAIRS = (
    {
        "label": "tinymixtral_vs_tinyllama_moe",
        "reference": ("tinymixtral", "base"),
        "candidate": ("tinyllama_moe", "base"),
        "n_experts": 4,
        "top_k": 2,
        "role": "different family, matching E and k (layer index aligned for 12 layers)",
    },
)

# Different-family comparisons are only valid when E and k match: expert ids
# are indices, so a 64-expert router and a 60-expert router do not share a
# label space. No cached, weight-complete, non-OLMoE 64-expert MoE exists
# here, so that cell is an explicit skip rather than a silent 64-vs-other
# comparison.
FAMILY_SKIP_NOTES = (
    {
        "population": "different_family_E64",
        "reason": (
            "no cached weight-complete non-OLMoE 64-expert MoE: "
            "deepseek-ai/deepseek-moe-16b-base has only model.safetensors.index.json "
            "cached and would need ~32 GB bf16 (>20 GB budget); "
            "Qwen/Qwen1.5-MoE-A2.7B has 60 experts (E mismatch) and no weights cached"
        ),
    },
)

# ---------------------------------------------------------------------------
# Budgets (skip cleanly, never download)
# ---------------------------------------------------------------------------

RAM_BUDGET_GB = 20.0
TOTAL_TIME_BUDGET_S = 1200.0  # ~20 min for the whole smoke
PER_MODEL_TIMEOUT_S = 600.0
# 13.8 GB OLMoE + loader overhead is the largest fit; leave ~4 GB headroom.
MAX_WEIGHT_GB = 16.0

# ---------------------------------------------------------------------------
# Threshold rule (frozen before the first run; no tuning on negatives)
# ---------------------------------------------------------------------------

THRESHOLD_RULE = {
    "metric": "per-layer mean Jaccard over all masked (prompt, position) rows",
    "primary_population": "olmoe_0924_experts_q4",
    "statistic": "mean - 3 * sample_sd (ddof=1) across the 16 per-layer means",
    "sd_multiplier": 3.0,
    "secondary_population": "olmoe_0924_all_q4",
    "negative_population": "olmoe_0924_vs_0924_sft",
    "positive_direction": "higher Jaccard means 'same lineage'",
    "fpr_definition": (
        "fraction of negative-population layers (and, separately, masked rows) "
        "whose Jaccard is at or above the threshold: a different checkpoint "
        "wrongly accepted as the same lineage"
    ),
    "tpr_definition": (
        "fraction of the calibration population's layers (and rows) at or "
        "above its own threshold; by construction a one-sided 3-sd tail"
    ),
    "tuning": (
        "threshold uses only the same-model requant population; "
        "the negative population is never consulted"
    ),
}

# Numeric bands, checked in order; the first band whose fpr_layer_max and
# tpr_layer_min both hold names the verdict. The fallback is the registered
# "otherwise" case and also covers a run whose primary calibration is
# unavailable, so every published verdict is one of these names. smoke.py
# executes this object itself — it never restates the numbers.
VERDICT_RULES = {
    "bands": {
        "usable_lineage_signal": {"fpr_layer_max": 0.0, "tpr_layer_min": 0.90},
        "inference_tier_only": {"fpr_layer_max": 0.25, "tpr_layer_min": 0.80},
    },
    "fallback": "too_brittle",
    "note": (
        "the verdict is about the Jaccard fingerprint statistic; exact top-k "
        "set agreement is reported alongside but is not the calibrated metric"
    ),
}

# ---------------------------------------------------------------------------
# Pre-registered expectations (written before the first run)
# ---------------------------------------------------------------------------
# op: ge / le / lt / gt / between / eq (absolute tolerance) / max_lt (every
# element of a list is < value). Missing metric -> "skipped".

EXPECTATIONS = [
    {
        "id": "E1",
        "text": "TinyMixtral experts_q4 keeps overall mean Jaccard >= 0.95",
        "metric": "models.tinymixtral.variants.experts_q4.overall_mean_jaccard",
        "op": "ge",
        "value": 0.95,
    },
    {
        "id": "E2",
        "text": (
            "TinyMixtral experts_q4 layer 0 is exactly 1.0 "
            "(routing does not depend on expert weights)"
        ),
        "metric": "models.tinymixtral.variants.experts_q4.layer0_mean_jaccard",
        "op": "eq",
        "value": 1.0,
        "tol": 1e-9,
    },
    {
        "id": "E3",
        "text": "TinyMixtral router Q4 costs more than expert Q4 (all_q4 Jaccard < experts_q4)",
        "metric": "derived.tinymixtral_router_cost_jaccard",
        "op": "gt",
        "value": 0.0,
    },
    {
        "id": "E4",
        "text": "OLMoE experts_q4 overall mean Jaccard in [0.90, 0.98] (prior run 0.944)",
        "metric": "models.olmoe_0924.variants.experts_q4.overall_mean_jaccard",
        "op": "between",
        "value": [0.90, 0.98],
    },
    {
        "id": "E5",
        "text": "OLMoE experts_q4 layer 0 is exactly 1.0",
        "metric": "models.olmoe_0924.variants.experts_q4.layer0_mean_jaccard",
        "op": "eq",
        "value": 1.0,
        "tol": 1e-9,
    },
    {
        "id": "E6",
        "text": "OLMoE router Q4 costs more than expert Q4 (all_q4 Jaccard < experts_q4)",
        "metric": "derived.olmoe_router_cost_jaccard",
        "op": "gt",
        "value": 0.0,
    },
    {
        "id": "E7",
        "text": "OLMoE same-architecture different checkpoint is less similar than requantization",
        "metric": "derived.checkpoint_minus_requant_jaccard",
        "op": "lt",
        "value": 0.0,
    },
    {
        "id": "E8",
        "text": "OLMoE different checkpoint stays above chance (finite lineage signal retained)",
        "metric": "derived.checkpoint_minus_chance_jaccard",
        "op": "gt",
        "value": 0.0,
    },
    {
        "id": "E9",
        "text": "No different-checkpoint layer reaches the primary threshold (layer FPR = 0)",
        "metric": "calibration.primary.fpr_layer",
        "op": "le",
        "value": 0.0,
    },
    {
        "id": "E10",
        "text": "Different family (E=4, k=2) is within 0.10 of its analytic chance Jaccard",
        "metric": "derived.family_minus_chance_jaccard",
        "op": "between",
        "value": [-0.10, 0.10],
    },
    {
        "id": "E11",
        "text": "Primary threshold stays above 0.60 (usable margin)",
        "metric": "calibration.primary.threshold",
        "op": "ge",
        "value": 0.60,
    },
    {
        "id": "E12",
        "text": "At least 12 of 16 requant layers clear their own threshold (TPR >= 0.75)",
        "metric": "calibration.primary.tpr_layer",
        "op": "ge",
        "value": 0.75,
    },
    {
        "id": "E13",
        "text": (
            "OLMoE exact set agreement at k=8 is in [0.70, 0.85] (the brittle metric, prior 0.762)"
        ),
        "metric": "models.olmoe_0924.variants.experts_q4.overall_set_agreement",
        "op": "between",
        "value": [0.70, 0.85],
    },
]
