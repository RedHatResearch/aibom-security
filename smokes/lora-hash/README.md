# LoRA-Hash smoke (smokes/lora-hash/)

From-scratch reimplementation of **LoRA-Hash** (He, Chen, Li, Feng, Ji,
"LoRA-Hash: Unveiling Model Identity from Heavy-Tailed Weight Distributions",
ICME 2026; camera PDF
[nesa.zju.edu.cn/download/icme2026__camera.pdf](https://nesa.zju.edu.cn/download/icme2026__camera.pdf)).
Survey deep-dive for
[#26](https://github.com/RedHatResearch/aibom-security/issues/26); runnable-check
context in [../README.md](../README.md).

**No official code exists**, and the public camera PDF withholds Appendices
B, C and G, so three gaps are filled here and pinned (below). Everything is
implemented from the paper text only.

```bash
uv sync --all-packages --group smokes
uv run --group smokes python smokes/lora-hash/smoke.py --selftest   # offline, exits 0 on pass
uv run --group smokes python smokes/lora-hash/smoke.py              # synthetic corpus, full JSON
uv run --group smokes python smokes/lora-hash/smoke.py --corpus real  # real-delta corpus (downloads)
uv run --group smokes python smokes/lora-hash/smoke.py --adapter path/to/adapter_model.safetensors
```

`--cache-dir`, `--seed`, `--anchors N` are also supported. Stdout is a single
JSON document; progress goes to stderr. CPU-only (Apple M3 Pro 36GB); no
`.to('cuda')` anywhere, no GPU assumptions.

## Method (pinned from the paper)

Per LoRA adapter — a set of module factor pairs `{module: (A, B)}` with
`dW = B @ A`, loadable from a PEFT `adapter_model.safetensors`
(`{module}.lora_A.weight` / `.lora_B.weight`, optional `base_model.model.`
prefix) or an `.npz` of `{module}.lora_A` / `{module}.lora_B`:

1. **Reconstruct**: per module, `dW = B @ A`, flattened.
2. **Layer discard**: drop modules with negligible dynamic range
   (`max|dW| < epsilon`, `epsilon = 1e-4` default).
3. **Per-module Z-score normalization** *before* concatenation:
   `v ← (v − mean)/std`. This is the paper's ablation-confirmed critical
   step: Concat-Hash (no `dW` reconstruction) scores 0.55 AUC and Raw-Hash
   without Z-norm 0.8871 vs 0.9166 for the full method (closed-set) — the
   sanity targets our selftest checks against, in direction.
4. **Concatenate** retained modules in sorted-module order → `V`.
5. **Segment** `V` into `N+1` equal contiguous segments (`N = 64` default →
   hash length `L = 2N = 128`, the paper's stated minimum for FPR < 1e-6).
6. **Bits**: per adjacent segment pair, `b_max[i] = 1 if f_max[i] > f_max[i+1]`
   else 0 (i = 1..N), same for the min stream; `H = concat(b_max, b_min)`.
7. **Match**: normalized Hamming distance `D = mismatches / L`;
   plagiarized/same-adapter iff `D <= tau`.
8. **Tau calibration**: on the negative pairs (unrelated adapters) to the
   paper's 1% FPR target. Tau is *calibrated, never hardcoded* (the paper's
   own protocol; we do not use 0.04).

Scoring core (`lora_hash`, `hamming`, `calibrate_tau`, `auc_score`) is pure
numpy/scipy on factor dicts — no I/O, no fixtures — so a ProvenanceBench
detector can lift it directly. The adapter loaders additionally cover the
PEFT safetensors and `.npz` layouts bit-exactly (checked by a round-trip in
`--selftest`).

## Withheld-appendix gaps and our resolutions

The camera PDF withholds **Appendices B, C and G**. The three gaps we had to
fill, and the pinned choices:

| Gap | Withheld in | Our resolution |
|---|---|---|
| Segment count / hash length | App. B (implementation detail) | `N = 64` segments of comparison → `L = 2N = 128` bits, the paper's stated minimum hash length for FPR < 1e-6. `N+1` equal contiguous segments over the concatenated vector, adjacent-segment max/min comparisons per stream. |
| Attack severities | App. C (attack suite) | Gaussian noise at 0.05/0.1/0.2 per-module std on both factors; magnitude pruning 10/30/50%; rank truncation to r/2, r/4 (SVD of `dW`); reparameterization `A → cA, B → B/c` for `c ∈ {0.5, 2.0}`; global rescale ×0.7/×1.5. Plus one addition of ours (below). |
| Threshold / calibration protocol detail | App. G | Calibrate `tau` on the unrelated-adapter negatives to a 1% FPR (the paper's stated protocol), never hardcode. |

**Our one attack addition — `finetune_drift`:** the issue's attack list
contains no attack that perturbs *relative* module scales (noise/prune are
within-module, reparameterization/global-rescale are within-module or exactly
scale-invariant under Z-norm, rank truncation changes spectrum). But relative
module scale is the *only* thing per-module Z-score normalization protects —
without such an attack the Z-norm ablation *cannot* show degradation and the
paper's 0.8871-vs-0.9166 ordering would be untestable. We therefore add a
"continued fine-tuning" drift attack: per-module lognormal scale drift
(`σ = 0.5, 1.0`) plus a small additive update — the realistic plagiarism
channel of re-finetuning a stolen adapter a few steps. This is documented as
ours, not the paper's.

## Corpus (self-generated — the pbench adapter gap)

**ProvenanceBench has no LoRA-adapter pairs at all** (every pair is a full
checkpoint pair). This gap is noted here and feeds
[#48](https://github.com/RedHatResearch/aibom-security/issues/48). The corpus
is therefore self-generated, in two modes:

- **`synthetic` (default, offline)** — random-factor adapters with the
  heavy-tailed per-module weight distributions the paper hashes: per-module
  spectral decay `α ~ exp(0.8·N(0,1))` in `A`, lognormal per-module scale
  `~ exp(0.6·N(0,1))`, 28 layers × {q_proj, v_proj} at rank 8, with two
  near-dead modules injected to exercise the epsilon-discard path. 12
  unrelated anchors by default (`--anchors`), each with the full attack grid
  as variants. Positives = anchor vs self and vs its attacked variants;
  negatives = all cross-anchor pairs (unrelated adapters).
- **`real` (`--corpus real`)** — REAL lineage in adapter form: for the pinned
  pbench pair **pb-smoke-qwen3-sft** (`Qwen/Qwen3-0.6B` →
  `Qwen/Qwen3-0.6B-Base`), compute the true delta `W_subject − W_reference`
  for all q_proj/v_proj modules and SVD-truncate to r=8 → pseudo-adapter
  `(A, B)` factors; same for **pb-smoke-smoltulu** (`SultanR/SmolTulu-1.7b-Instruct`
  → `HuggingFaceTB/SmolLM2-1.7B`) as the cross-family negative anchor, plus
  synthetic random-factor adapters as additional negatives. Model ids and
  pinned revisions are read at runtime from
  [provenancebench/pairs.yaml](../../provenancebench/pairs.yaml) (the
  orchestrator's manifest) via a small loader — never inlined. **Download
  size: both checkpoints are safetensors-only, ~1.2 GB + ~3.4 GB ≈ 4.6 GB**
  into the shared HF cache on a cold run (both non-gated; a missing
  `HF_TOKEN` is fine). Both anchors get the full attack grid as positives.

**Corpus-strength caveat (read before citing the numbers):** the real
mode's AUC 1.0 / recall 1.0 are pipeline validation, not method
validation. The positives include anchor-vs-self identity pairs
(distance 0), and the negatives are cross-family or random — an easy
separation. The "adapters" are SVD truncations of dense SFT deltas,
not trained LoRAs, so the paper's heavy-tailed Sparsity Hypothesis is
only partially exercised. A meaningful evaluation needs a corpus of
real Hub adapters (the same gap as #48).

## Acceptance criteria (`--selftest`)

Pre-registered, offline, synthetic corpus only, fixed seed 0. All must hold
for exit 0:

| Check | Criterion |
|---|---|
| Hash length | `L = 128` bits exactly |
| Layer discard exercised | at least one module dropped by the epsilon rule |
| Recall | anchor-vs-variant positives: `D ≤ tau` for ≥ 90% (target ≥ 0.9) |
| Negatives | unrelated anchors above calibrated `tau` (eval-split FPR ≤ 10%, i.e. negatives sit above tau with margin) |
| Concat-Hash ablation | degrades *sharply*: recall ≥ 5 points and AUC ≥ 5 points below the full method (paper: 0.55 vs 0.9166 AUC) |
| No-Z-norm ablation | degrades: AUC not above the full method, and mean drift-attack distance ≥ 5× the full method's (paper: 0.8871 vs 0.9166 AUC) |
| Adapter I/O | `.npz` and PEFT safetensors round-trips reproduce the hash bit-exactly |

Rationale for the ≥5× drift-distance criterion instead of a raw AUC gap for
the no-Z-norm ablation: at 12 anchors the closed-set negatives are so far
above tau (D ≈ 0.5 vs tau ≈ 0.35) that the Z-norm loss cannot move global
AUC/recall measurably — its damage is concentrated exactly on the
relative-scale (drift) attacks, where the without-Z-norm distance is ~30×
the full method's (0.19 vs 0.006). The criterion pins the *direction* of the
paper's 0.8871-vs-0.9166 ordering on the attack family that carries it.

## Output contract

One JSON document on stdout:

- `params` (`n_segments`, `hash_bits`, `epsilon`, `target_fpr`, `rank`, `seed`)
- `corpus` (mode, anchors, attack list, pbench provenance for `real`)
- `tau` (calibrated on the negative calibration split), `recall_at_1pct_fpr`,
  `fpr_eval` (held-out negative split)
- `pairs`: per-positive `{anchor, variant, attack, D_hamming, match}`
- `negatives`: per-negative `{a, b, D_hamming, match, split}`
- `ablations`: per-pipeline (`full`, `no_z_norm`, `concat_only`) — tau,
  recall, FPR, AUC, per-attack mean D, dropped modules, timing; each pipeline
  calibrates its own tau on the same fixed negative split
- `timing`: corpus build, per-pipeline hashing, total (wall clock)
- `--selftest` adds `selftest.checks` (each criterion above) and
  `selftest.pass`

`--adapter PATH` prints a single JSON line with the adapter's hash (hex),
module count and dropped-module count, then exits.

## Provenance / license notes

- Method: LoRA-Hash, ICME 2026 (He, Chen, Li, Feng, Ji, Zhejiang
  University/NESA Lab). No official code exists as of 2026-10-12; this smoke
  is a from-scratch reimplementation from the camera PDF. Appendix
  B/C/G gap-fills are ours and documented above.
- Real-corpus checkpoints: Qwen (Apache-2.0), SmolLM2 (Apache-2.0),
  SmolTulu (apache-2.0) — all non-gated; `HF_TOKEN` not required.
- Synthetic tensors: generated in-process with `numpy.random.default_rng`;
  no network, no HF downloads in `--selftest`.

## Timing

Apple M3 Pro 36GB, CPU-only, numpy. `--selftest` (12 anchors × 14 attack
variants, three pipelines): a few seconds end-to-end, dominated by corpus
generation and the three hash passes. `--corpus real` adds the two
checkpoint downloads (~4.6 GB cold) and ~2 minutes of delta+SVD work over
56 modules, then the same scoring path. Per-pipeline wall-clock is recorded
in `timing.hash_seconds` in the JSON output.
