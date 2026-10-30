"""Pinned constants for the real-pair consistency smoke (issue #68).

The pair roster (repo ids and pinned revisions) is read from
``provenancebench/pairs.yaml`` so the pins have one source of truth. Only the
structural fields are read: ``expected``, ``evidence`` and ``challenge`` are
label-path data and are deliberately never loaded, here or anywhere downstream
of the harness. Thresholds are not defined here; they live frozen in
``smokes/moe-expert-lineage/detect.py`` and are echoed into ``results.json``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
PAIRS_YAML = REPO_ROOT / "provenancebench" / "pairs.yaml"
RESULTS = ROOT / "results.json"
SCRATCH = REPO_ROOT / ".scratch" / "moe-real-pairs"

# Descriptor seed: same seed as the synthetic Experiment A detector run.
SEED = 20261029

# Budget for the whole smoke, measured from process start. Downloads count.
BUDGET_S = 1200.0
RESERVE_S = 45.0
RSS_BUDGET_MB = 20 * 1024

# Download-rate assumption before this run has measured one. Measured on the
# run host's link (2.7 MB/s range GET), deliberately on the pessimistic side:
# an over-estimate would blow the budget, an under-estimate only skips a layer
# that is recorded as skipped.
DEFAULT_SPEED_BPS = 2_500_000
# Analysis-time assumption before the first layer of a pair has been scored.
# Measured on synthetic 128x96x3-projection tensors: 3.8 s for the mixture fit,
# plus descriptor reads from local disk.
ANALYSIS_DEFAULT_S = {"pb-core-granite-reap-025": 10.0, "pb-core-qwen3-30b-ream": 30.0}

# Tensor naming per architecture. ``layout: batched`` means one tensor holds
# every expert of a layer with the expert index in dim 0 (GraniteMoe); the
# driver presents it through a per-expert view so the detector's API is
# unchanged. ``layout: per_expert`` uses the detector's ``TensorSpec`` as is.
ARCHES = {
    "granitemoe": {
        "layout": "batched",
        "roles": ("input_linear", "output_linear"),
        "expert_template": "model.layers.{layer}.block_sparse_moe.experts.{expert}.{role}.weight",
        "router": "model.layers.{layer}.block_sparse_moe.router.layer.weight",
        "batched": {
            "input_linear": "model.layers.{layer}.block_sparse_moe.input_linear.weight",
            "output_linear": "model.layers.{layer}.block_sparse_moe.output_linear.weight",
        },
        "expert_keys": ("num_local_experts", "num_experts"),
    },
    "qwen3_moe": {
        "layout": "per_expert",
        "roles": ("gate_proj", "up_proj", "down_proj"),
        "expert_template": "model.layers.{layer}.mlp.experts.{expert}.{role}.weight",
        "router": "model.layers.{layer}.mlp.gate.weight",
        "batched": {},
        "expert_keys": ("num_experts", "num_local_experts"),
    },
}

# None = every MoE layer. The 30B pair is 48 layers of 128 experts; 0/24/47
# (first, middle, last) is the pre-registered sample when the budget allows.
SAMPLING = {
    "pb-core-granite-reap-025": None,
    "pb-core-qwen3-30b-ream": (0, 24, 47),
}

# Published recipe, quoted from the pinned model cards (read before the run).
RECIPE = {
    "pb-core-granite-reap-025": {
        "claim": "REAP expert drop 32 -> 24 (ratio 0.25); survivors copied, not merged",
        "parent_experts": 32,
        "subject_experts": 24,
        "source": "subject README 'Pruning Setup' / 'Checkpoint' sections",
    },
    "pb-core-qwen3-30b-ream": {
        "claim": (
            "REAM expert merge 128 -> 96; k=96 centroids by REAP saliency, "
            "32 non-centroid experts absorbed into groups, weighted average after "
            "permutation alignment; non-centroid gate rows removed"
        ),
        "parent_experts": 128,
        "subject_experts": 96,
        "source": "subject README + linked REAM method post (bknyaz.github.io, 2026)",
    },
}

# Pre-registration, written before the run. The machine checks live in
# smoke.py::_judge_*; these strings are what the README states.
PREREG = {
    "pb-core-granite-reap-025": (
        "subset of 24 verbatim survivors per layer, 8 missing parents, no new expert, "
        "and no mixture fit above gain 0.30"
    ),
    "pb-core-qwen3-30b-ream": (
        "per sampled layer: matched in [64, 96] (the 96-32 singleton centroids are "
        "copies), missing in [32, 64] (== 128 - matched), merged fits (gain >= 0.30) "
        "in [0, 32], every one of the 96 suspects classified; the REAM recipe permutes "
        "group members before averaging, so low merged counts are reported, not failed"
    ),
}


def load_pair(pair_id: str) -> dict:
    """Structural fields of one pinned pair; label-path fields are not touched."""
    with open(PAIRS_YAML) as handle:
        raw = yaml.safe_load(handle)
    for entry in raw["pairs"]:
        if entry.get("id") == pair_id:
            return {
                "id": pair_id,
                "stack": str(entry["stack"]),
                "subject": str(entry["subject"]),
                "subject_rev": str(entry["subject_rev"]),
                "reference": str(entry["reference"]),
                "reference_rev": str(entry["reference_rev"]),
            }
    raise KeyError(f"pair {pair_id!r} not in {PAIRS_YAML}")
