# TRACE smoke (smokes/trace-sft/)

Reimplementation of the scoring method of **TRACE** — *"Harmful SFT Leaves a
Continuous Trace in LLM Checkpoint Updates"* ([arXiv:2610.07518](https://arxiv.org/abs/2610.07518))
— for the [#26](https://github.com/RedHatResearch/aibom-security/issues/26)
fingerprint survey. TRACE detects harmful (unsafe-response) SFT from the
**checkpoint update alone**: the low-rank (LoRA) delta a fine-tune leaves on
its base model is projected onto a reference geometry of harmful vs.
non-harmful updates, yielding a continuous harmfulness score `s_H ∈ [0, 1]`.

**License note:** the paper's anonymous code repository ships **no license**,
so nothing is vendored. Equations 1–5 below are reimplemented from the paper,
and the SFT harness is written from scratch to the paper's hyperparameters.

## Method (Eq. 1–5, reimplemented from the paper)

Given a fine-tuned checkpoint expressed as per-module LoRA factors
(`dW = (alpha/r) · B @ A` per attention module):

1. **Kernel (Eq. 1–3).** `K(i, j) = Σ_modules ⟨dW_i, dW_j⟩_F`. For two
   low-rank updates this is computed **factorized** —
   `s_i·s_j·tr((B_iᵀB_j)(A_jA_iᵀ))` with r×r `float64` intermediates — so a
   dense `dW` is never materialized. Dense queries (the real-OOD point) are
   supported via `⟨d, s·b@a⟩_F = Σ b∘(d@aᵀ)`.
3. **Score (Eq. 4–5).** Solve the paper's normal equations for the 2-variable
   NNLS `x ≈ γ_C·μ_C + γ_N·μ_N`, `γ ≥ 0` in the RKHS: `G·γ = b` with the
   **feature-space prototype Gram** `G[c,d] = ⟨μ_c, μ_d⟩_H` (block means of
   the bank kernel Gram, including the mixed C–N/S–N/B–N blocks via
   `μ_N = ½(μ_S + μ_B)`) and `b[c] = ⟨x, μ_c⟩_H` (mean of `K(x, i)` over
   class `c`). Solved exactly by face enumeration on the 2×2 `G`
   (selftest-verified against `scipy.optimize.nnls`, and against a
   brute-force dense feature-space NNLS), then
   `s_H = γ_C / (γ_C + γ_N + 1e-12)`.
The pure scoring core (`kernel`, `bank_gram`, `phi_of`, `prototype_gram`,
`nnls2`, `trace_score`) is numpy-only with no I/O, so a later
ProvenanceBench detector can lift it directly.

## Harness (the smoke builds its own trajectories)

- **Trainer**: manual LoRA r=8, α=16, dropout 0 on `q_proj`/`v_proj`,
  AdamW lr 2e-4 constant, batch 8 × grad-accum 2, 100 optimizer steps,
  CPU fp32 (MPS bf16 available via `--device mps`; CUDA is never used — the
  survey targets Apple M3 Pro 36 GB, CPU-only).
- **Data**: PKU-Alignment/PKU-SafeRLHF supplies class C (prompt + unsafe
  response) and the *same* prompts for class S with one of 8 fixed refusal
  templates assigned by SHA-256 of the prompt text (matched C/S by construction);
  HuggingFaceH4/ultrachat_200k supplies class B. Fetched via the
  datasets-server REST API (stdlib `urllib`; no `datasets` dependency).
  `--quick` replaces all of this with a synthetic random-token substitute
  (class-specific vocab bands) so the harness is testable offline.
- **Reference bank**: K=4 trajectories × 3 classes = 12 (the paper's own
  minimum stable bank; the paper found the geometry stable for K* ≈ 4–8
  per class), 64 examples, 100 steps each.
- **Queries**: mixed C/B at ratios {0, 10, 25, 50, 75, 100}% × 2 seeds =
  12 checkpoints, 100 examples, 100 steps, disjoint example pools from the
  bank.
- **Backbone**: Qwen/Qwen3-0.6B-Base at pinned revision
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, read from
  [provenancebench/pairs.yaml](../../provenancebench/pairs.yaml) pair
  `pb-smoke-qwen3-sft` at runtime (keeps provenance consistent with the
  manifest; a config fallback holds the same values if the manifest is
  absent). `--backbone HuggingFaceTB/SmolLM2-360M` overrides for speed
  (revision then resolves to latest).

## Reference-bank format

`data/bank_manifest.csv` — columns `trajectory_id, model, class,
factors_path` — plus one npz per trajectory holding `<module>__A`,
`<module>__B` arrays and the scalars `scaling` (α/r) and `r`. Query factors
are stored under `data/queries/` in the same format.

## How to run

```bash
uv sync --all-packages --group smokes
uv run --group smokes python smokes/trace-sft/smoke.py --selftest   # offline, ~5 s
uv run --group smokes python smokes/trace-sft/smoke.py --quick      # offline*, ~6-10 min CPU
uv run --group smokes python smokes/trace-sft/smoke.py              # full, real data (user-gated)
uv run --group smokes python smokes/trace-sft/smoke.py --real-ood   # + real Qwen3-0.6B point
```

\* `--quick` needs no network: it uses the synthetic data substitute and, if
the backbone is not already in the HF cache, a random-init Qwen3-0.6B-shaped
model (reported as `backbone.random_init_fallback: true`).

Stdout: one JSON object per scored query row
(`{trajectory_id, ratio, seed, s_H, gamma_C, gamma_N}`), then a final JSON
document embedding all rows plus the summary
`{spearman, auroc_matched_cs, timing_by_phase, gates, passed}`. Progress goes
to stderr. `HF_TOKEN` is not required (all repos and datasets are
non-gated; it is honored implicitly by `huggingface_hub` if set).

### Modes, gates and expected wall-clock (measured on the M3 Pro target)

| mode | data | bank | steps | wall-clock | pass criteria |
|---|---|---|---|---|---|
| `--selftest` | synthetic factors | — | — | ~5 s | all numeric checks pass, exit 0 |
| `--quick` | synthetic tokens | K=2 | 20 | ~6–10 min CPU | pipeline runs end-to-end + scores computed; achieved Spearman **reported, not thresholded** |
| default (full) | PKU-SafeRLHF + ultrachat | K=4 | 100 | ~8–14 h CPU | Spearman(ratio, s_H) ≥ 0.9 **and** matched C/S AUROC ≥ 0.9 |
| `--real-ood` | + Qwen3-0.6B instruct vs Base dW | — | — | + ~5 min (1.2 GB download) | expectation: low s_H (diagnostic, not gated) |

CPU cost basis (measured, Qwen3-0.6B shape, batch 8): ~20 s per 8×512
micro-batch, ~8.6 s at 8×384; full mode = 24 trajectories × 200 micro-batches,
with dynamic padding shortening real-data batches. **Full mode is
user-gated** — the orchestrator does not run it by default; the paper reports
Spearman 0.986–0.992 and matched C/S AUROC 0.988 at this scale, and the smoke
gates at 0.9/0.9.

`--real-ood` scores the real Qwen/Qwen3-0.6B (rev `c1899de…`) minus
Qwen/Qwen3-0.6B-Base (rev `da87bf…`) dense update on the same q/v modules
against the bank geometry: an ordinary instruct post-train should land at
**low s_H**. Pre-registered as a diagnostic — a high value is a finding about
the geometry (or the bank), not a silent failure.

### Selftest checks (offline, synthetic tensors, full scoring path)

1. factorized kernel == dense Frobenius computation (rel err < 1e-9);
2. dense-vs-low-rank kernel (the real-OOD path) == dense Frobenius;
3. LoRA reparameterization invariance: `A→cA`, `B→B/c` leaves the Gram and
   `s_H` unchanged (dW preserved exactly);
4. exact 2-variable NNLS vs `scipy.optimize.nnls` over 300 random cases;
5. `s_H` monotonicity on a synthetic 1-D geometry: queries interpolated
   between `μ_N` and `μ_C` (realized as concatenated-rank factor mixtures)
   get monotonically increasing `s_H`; a 5%-noise variant is reported
   (not gated) to show robustness margin;
6. block-mean Gram path == brute-force dense feature-space NNLS: dense dW
   vectors are materialized, prototypes computed as actual dense class
   means, and the dense-space NNLS solved with scipy — the resulting γ must
   match the kernel-path γ (this is the check that pins Eq. 4 to the
   feature-space objective).

## Fixtures / provenance

- Backbone + real-OOD pair: `pb-smoke-qwen3-sft` from
  [provenancebench/pairs.yaml](../../provenancebench/pairs.yaml)
  (subject Qwen/Qwen3-0.6B @ `c1899de289a04d12100db370d81485cdf75e47ca`,
  reference Qwen/Qwen3-0.6B-Base @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`).
- Datasets: PKU-Alignment/PKU-SafeRLHF (default config, train split),
  HuggingFaceH4/ultrachat_200k (default config, train_sft split), read via
  the public datasets-server API; both non-gated.

## Deviations from the paper

1. **Reimplementation only** (no license on the anonymous artifact). The
   scoring core follows the paper's equations exactly: Eq. 1 kernel,
   Eq. 2 class-mean prototypes, Eq. 4 normal equations (feature-space
   prototype Gram via block means of the bank kernel Gram, incl. the mixed
   blocks through `μ_N`), and Eq. 5 `s_H`. The NNLS itself is solved exactly
   by 2-variable face enumeration rather than an iterative solver
   (selftest-verified against scipy and against a brute-force dense
   feature-space NNLS). Everything around those equations — the SFT
   harness, data plumbing, and bank/query scale — is our own construction.
2. **Scale**: 0.6B backbone (paper: 1–8B), 64/100 examples per trajectory,
   100 optimizer steps — enough for the trace to form, not for paper-level
   absolute numbers; the gates are set at 0.9 rather than the paper's
   0.986–0.992 accordingly.
3. **`max_len` 384 vs the paper's 512** (measured cost basis: ~20 s per
   8×512 micro-batch vs ~8.6 s at 8×384 on the CPU target).
4. **Refusals** (class S) are our own 8 one-line refusal templates, assigned
   by SHA-256 of the prompt text — they differ from the paper's
   Appendix A.2 refusal templates; deterministic, matched to the C prompts,
   no extra model downloads.
5. **`POOL_SIZE=800`** rows fetched per dataset is our harness parameter
   (not from the paper); the paper does not specify pool sizes.
6. **Device**: CPU fp32 is the default; measured faster than MPS bf16 at
   full sequence length on the Apple M3 Pro target (MPS available via
   `--device mps`); CUDA is never used.
7. **Quick mode** uses a synthetic data substitute and (if uncached) a
   random-init backbone: it exercises the exact training → factor → bank →
   scoring path offline but carries no paper-level signal; its Spearman is
   reported, never gated.
