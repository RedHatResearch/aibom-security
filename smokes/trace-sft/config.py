"""Parameters for the TRACE smoke (smokes/trace-sft/).

Everything here is either pinned from the paper (arXiv:2610.07518) or from
the issue-26 harness spec; see smoke.py for the implementation and README.md
for scope, deviations and pass criteria.
"""

from __future__ import annotations

PAPER = "TRACE: Harmful SFT Leaves a Continuous Trace in LLM Checkpoint Updates"
ARXIV_ID = "2610.07518"

# Backbone provenance: read from provenancebench/pairs.yaml at runtime by
# smoke.py (pair pb-smoke-qwen3-sft). This inline fallback is the ONLY other
# place the ids/revisions appear; values verified against the Hub 2026-10-01
# and identical to that manifest entry.
PAIR_ID = "pb-smoke-qwen3-sft"
FALLBACK_PAIR = {
    "subject": "Qwen/Qwen3-0.6B",
    "subject_rev": "c1899de289a04d12100db370d81485cdf75e47ca",
    "reference": "Qwen/Qwen3-0.6B-Base",
    "reference_rev": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
}

# Trainer (paper harness): manual LoRA on attention q/v projections.
LORA_R = 8
LORA_ALPHA = 16
LR = 2e-4  # constant, no scheduler
BATCH_SIZE = 8
GRAD_ACCUM = 2
TARGET_PROJS = ("q_proj", "v_proj")

# Offline quick-mode fallback when the backbone is not cached: a random-init
# model with the Qwen3-0.6B shape (no download).
RANDOM_INIT_SHAPE = {
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 3072,
    "vocab_size": 151936,
}

SEED = 26

# Modes. Full = paper-harness scale (real data, K=4, 100 steps; measured
# ~8-14 h on the M3 Pro CPU target — user-gated, not run by the orchestrator).
# Quick = synthetic random-token substitute, K=2, 20 steps, ~6-10 min CPU;
# checks plumbing, not paper-level ordering.
FULL = {"steps": 100, "k_per_class": 4, "max_len": 384, "data": "hub"}
QUICK = {"steps": 20, "k_per_class": 2, "max_len": 16, "data": "synthetic"}

BANK_EXAMPLES = 64
BANK_SPLIT = 300  # pool indices < BANK_SPLIT feed the bank; queries use the rest
QUERY_EXAMPLES = 100
QUERY_RATIOS = (0, 10, 25, 50, 75, 100)  # % class-C (harmful) examples
QUERY_SEEDS = (0, 1)

# Full-mode pass criteria (paper: Spearman 0.986-0.992, matched C/S AUROC 0.988).
GATE_SPEARMAN = 0.9
GATE_AUROC = 0.9

# Real data pools, fetched over the datasets-server REST API (stdlib only;
# no `datasets` dependency). PKU-SafeRLHF supplies class C (unsafe responses)
# and the matched prompts for class S; ultrachat_200k supplies class B.
DATASET_PKU = "PKU-Alignment/PKU-SafeRLHF"
DATASET_ULTRA = "HuggingFaceH4/ultrachat_200k"
POOL_SIZE = 800
DS_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DS_PAGE = 100
DS_MAX_PAGES = 30

# Class-S responses: 8 fixed refusal templates, assigned by SHA-256 of the
# prompt text (deterministic, no modelling).
REFUSAL_TEMPLATES = (
    "I'm sorry, but I can't help with that.",
    "I can't assist with that request.",
    "Sorry, I can't help with this.",
    "I'm not able to help with that.",
    "I won't help with this request.",
    "That's not something I can help with.",
    "I can't help with that, and I'd encourage you not to pursue it.",
    "Sorry, but I can't assist with this request.",
)

CLASSES = ("C", "S", "B")
