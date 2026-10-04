# MoE gate probe smoke

- Issue: [#17](https://github.com/RedHatResearch/aibom-security/issues/17) MoE signal probe · sibling [`moe-router-gram/`](../moe-router-gram/)
- Code: [`smokes/moe-gate-probe/`](https://github.com/RedHatResearch/aibom-security/tree/main/smokes/moe-gate-probe)

## What this checks

The Eq. 1 s_MSA gate from [TensorLock](https://doi.org/10.1145/3832119), evaluated on the
shared attention stack of native MoE models. Per attention role (q/k/v/o), per-layer moment
curves (std, skew, kurtosis) of the projection weights are compared across two checkpoints
with Spearman correlation, truncated to the shorter stack; s_MSA is the mean of the twelve
curves and the gate fires at s_MSA ≥ τ = 0.6. The question: does a gate validated on dense
checkpoints still separate same-lineage from cross-family pairs when both sides are MoE
(OLMoE base ↔ SFT)? Feeds #17; this probe is gate-only — QPS, FT-direction and
router/expert signals are follow-ups.

## Fixtures

| label | pair | relation | layers | revisions |
|---|---|---|---|---|
| `olmoe_sft` | allenai/OLMoE-1B-7B-0924 ↔ allenai/OLMoE-1B-7B-0924-SFT | same_lineage_moe_sft | 16 / 16 | `6d84c485` / `215cc4f7` |
| `olmoe_neg_smollm` | allenai/OLMoE-1B-7B-0924 ↔ HuggingFaceTB/SmolLM2-1.7B-Instruct | cross_family_moe_dense | 16 / 24 | `6d84c485` / `31b70e2e` |
| `olmoe_neg_qwen3` | allenai/OLMoE-1B-7B-0924-SFT ↔ Qwen/Qwen3-0.6B | cross_family_moe_dense | 16 / 28 | `215cc4f7` / `c1899de6` |
| `tinyllama_upcycle` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 ↔ s3nh/TinyLLama-4x1.1B-MoE | card_claimed_dense_parent_upcycled_moe | 22 / 22 | `fe8a4ea1` / `e82b7308` |

Revisions pinned in [config.py](config.py) (`REVISIONS`) so a deleted or force-pushed Hub
repo cannot silently change what is scored.

## Pre-registration (2026-10-04, before the run)

| label | pair | relation | expectation |
|---|---|---|---|
| `olmoe_sft` | allenai/OLMoE-1B-7B-0924 ↔ allenai/OLMoE-1B-7B-0924-SFT | same_lineage_moe_sft | gate passes, s_MSA ≥ 0.9 |
| `olmoe_neg_smollm` | allenai/OLMoE-1B-7B-0924 ↔ HuggingFaceTB/SmolLM2-1.7B-Instruct | cross_family_moe_dense | gate rejects, s_MSA ≤ 0.30 |
| `olmoe_neg_qwen3` | allenai/OLMoE-1B-7B-0924-SFT ↔ Qwen/Qwen3-0.6B | cross_family_moe_dense | gate rejects, s_MSA ≤ 0.30 |
| `tinyllama_upcycle` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 ↔ s3nh/TinyLLama-4x1.1B-MoE | card_claimed_dense_parent_upcycled_moe | gate passes |

Reasoning, fixed before the run:

- `olmoe_sft` passes the gate: dense same-lineage analogs scored ≥ 0.99, and attention is
  untouched by MoE-ization, so we expect the dense band (≥ 0.9).
- `olmoe_neg_smollm` and `olmoe_neg_qwen3` are rejected: dense cross-family band (≤ 0.30).
  A pass on either is a finding, not a failure of the probe.
- Gate threshold τ = 0.6; moment curves truncated to L_min = 16 layers (OLMoE 16,
  SmolLM2-1.7B-Instruct 24, Qwen3-0.6B 28).

Upcycle fixture, added 2026-10-04 before its run:

- `tinyllama_upcycle` primary expectation is a gate pass: the card claims the TinyLlama
  parent, and an upcopy preserves the attention stack (MDGBench's corpus clustered this
  child into the TinyLlama family — weak paper-side support). No score band is registered
  because no dense analog exists for this cell; the reading is banded instead:
  >= 0.9 means attention survived upcycle plus finetune essentially untouched;
  0.6-0.9 is a drift finding (claim still verifiable at gate level, but upcycle finetuning
  moved attention more than SFT does); a rejection would mean the card claim is
  unverifiable from the attention stack alone.

Run 2, 2026-10-04 (`--only tinyllama_upcycle`, exit 0): expectation held. Warm-cache
compute 7.7 s (the first execution's 950 s wall was download lock-wait, not compute).

## Method notes

- The gate reads only self_attn q/k/v/o (the shared stack); MoE experts and the router are
  invisible to it by design.
- Negatives are MoE-vs-dense because the cheapest MoE-vs-MoE cross-family pair costs
  ≥ 29 GiB extra.
- Layer truncation means the negatives' stats vectors are cut to OLMoE's 16 layers.
- The upcycle fixture is the claim-verification shape: a card-declared dense parent vs a
  community upcycled MoE child. The child is MixtralForCausalLM with the parent's
  22 layers / 2048 hidden, so the attention stacks align 1:1 with no truncation. The
  experts — where the lineage claim actually lives — are invisible to the gate.

## Results

Run 2026-10-04 (exit 0): all three pre-registered expectations held. 20.5 s of compute
after the ~26 GiB OLMoE download (shared HF cache; negatives were cache hits).

| label | pair | relation | expectation | s_MSA | gate |
|---|---|---|---|---|---|
| `olmoe_sft` | allenai/OLMoE-1B-7B-0924 ↔ allenai/OLMoE-1B-7B-0924-SFT | same_lineage_moe_sft | pass, ≥ 0.9 | **0.9966** | pass |
| `olmoe_neg_smollm` | allenai/OLMoE-1B-7B-0924 ↔ HuggingFaceTB/SmolLM2-1.7B-Instruct | cross_family_moe_dense | reject, ≤ 0.30 | **0.1792** | reject |
| `olmoe_neg_qwen3` | allenai/OLMoE-1B-7B-0924-SFT ↔ Qwen/Qwen3-0.6B | cross_family_moe_dense | reject, ≤ 0.30 | **-0.3147** | reject |
| `tinyllama_upcycle` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 ↔ s3nh/TinyLLama-4x1.1B-MoE | card_claimed_dense_parent_upcycled_moe | pass (gate) | **1.0000** | pass |

Reading:

- The same-lineage MoE pair scores 0.9966 (per role: q 0.998, k 0.994, v 0.995, o 0.999),
  dead center in the dense same-lineage band (0.9921–0.9971 in `smokes/tensorlock/`). The
  Eq. 1 gate needs no MoE-specific work to serve as the connectivity stage.
- Both cross-family negatives are rejected with the full 0.82 separation margin. The 4-role
  averaging carries the decision: against SmolLM2, `o` alone correlates at 0.81 while `q`/`k`
  anticorrelate (-0.16 / -0.34).
- The upcycle fixture lands in the top band and then some: s_MSA exactly 1.0000, all four
  roles. Verified against the weights directly — all 88 attention tensors (22 layers x 4
  roles) are bitwise identical between parent and child (max abs diff 0.0). The upcycler
  copied the attention stack verbatim; no finetune touched it. The card claim is trivially
  verifiable at the connectivity stage, and the entire lineage question in this pair lives
  in the experts — the exact shape the gate cannot see.
- Follow-ups, scoped not run: expert-histogram QPS for quantized MoE children, and a
  MoE-vs-MoE cross-family negative. The open MoE cells are expert-side: verifying
  dense-to-MoE upcycle claims and quantized-MoE provenance beyond the shared stack.

## Ops

```bash
uv sync --all-packages --group smokes
uv run --group smokes python smokes/moe-gate-probe/smoke.py
```

`--only LABEL` runs a single fixture (the label is validated before any Hub call, so a
typo cannot trigger a download); `--cache-dir PATH` overrides the cache location — the
default is the shared HF cache, reusing anything already downloaded. Cold run downloads
~26 GiB for the OLMoE pair (negatives add ~5 GiB, the upcycle pair ~8.4 GiB); runtime is
minutes on CPU. Stdout is a
single JSON document (progress on stderr); exit 0 iff every fixture's registered
expectation held, 1 otherwise.
